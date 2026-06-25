#!/usr/bin/env python3
"""
summarize_batch.py — Aggregate metrics from a batch of 25 experiments.

Reads metrics.json from each subdirectory of --batch_dir and writes:
  summary.csv     — all metrics in tabular form
  summary.txt     — pretty-printed table
  summary_panel.png — tiled result panels (if result_panel.png files exist)
"""
from __future__ import annotations
import argparse, csv, json, math, os, sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def vec_mag(x, y):
    if x is None or y is None:
        return None
    return math.sqrt(x**2 + y**2)


def vec_err(gt_dx, gt_dy, det_dx, det_dy):
    if any(v is None for v in [gt_dx, gt_dy, det_dx, det_dy]):
        return None
    return math.sqrt((gt_dx - det_dx)**2 + (gt_dy - det_dy)**2)


def load_metrics(batch_dir: Path) -> list[dict]:
    rows = []
    for metrics_file in sorted(batch_dir.glob("*/metrics.json")):
        exp_name = metrics_file.parent.name
        m = json.load(open(metrics_file))
        gt_dx  = m.get("gt_pixel_dx")
        gt_dy  = m.get("gt_pixel_dy")
        det_dx = m.get("detected_pixel_dx")
        det_dy = m.get("detected_pixel_dy")
        rows.append({
            "experiment":       exp_name,
            "object":           exp_name.rsplit("_", 1)[0] if "_" in exp_name else exp_name,
            "move":             exp_name.rsplit("_", 1)[1] if "_" in exp_name else "",
            "gt_dx":            gt_dx,
            "gt_dy":            gt_dy,
            "gt_mag":           vec_mag(gt_dx, gt_dy),
            "det_dx":           det_dx,
            "det_dy":           det_dy,
            "det_mag":          vec_mag(det_dx, det_dy),
            "error_px":         vec_err(gt_dx, gt_dy, det_dx, det_dy),
            "n_gaussians":      m.get("n_gaussians_modified"),
            "psnr":             m.get("psnr_updated_vs_gt_after"),
            "ssim":             m.get("ssim_updated_vs_gt_after"),
        })
    return rows


def print_table(rows: list[dict]) -> str:
    header = (
        f"{'Experiment':<22}  {'GT (dx,dy) px':<18}  "
        f"{'Det (dx,dy) px':<18}  {'Err px':>6}  {'PSNR':>7}  {'SSIM':>6}"
    )
    sep = "─" * len(header)
    lines = [sep, header, sep]

    prev_obj = None
    for r in rows:
        obj = r["experiment"].rsplit("_", 1)[0] if "_" in r["experiment"] else r["experiment"]
        if prev_obj and obj != prev_obj:
            lines.append("")
        prev_obj = obj

        gt   = f"({r['gt_dx']:+.1f}, {r['gt_dy']:+.1f})" if r["gt_dx"] is not None else "—"
        det  = f"({r['det_dx']:+.1f}, {r['det_dy']:+.1f})" if r["det_dx"] is not None else "—"
        err  = f"{r['error_px']:>6.1f}" if r["error_px"] is not None else "     —"
        psnr = f"{r['psnr']:>7.2f}" if r["psnr"] is not None else "     —"
        ssim = f"{r['ssim']:>6.3f}" if r["ssim"] is not None else "    —"
        lines.append(
            f"{r['experiment']:<22}  {gt:<18}  {det:<18}  {err}  {psnr}  {ssim}"
        )

    lines.append(sep)

    # Aggregate stats
    valid = [r for r in rows if r["psnr"] is not None and r["error_px"] is not None]
    if valid:
        avg_err  = sum(r["error_px"] for r in valid) / len(valid)
        avg_psnr = sum(r["psnr"]     for r in valid) / len(valid)
        avg_ssim = sum(r["ssim"]     for r in valid) / len(valid)
        med_err  = sorted(r["error_px"] for r in valid)[len(valid)//2]
        lines.append(
            f"{'MEAN':<22}  {'':<18}  {'':<18}  {avg_err:>6.1f}  {avg_psnr:>7.2f}  {avg_ssim:>6.3f}"
        )
        lines.append(
            f"{'MEDIAN error':<22}  {'':<18}  {'':<18}  {med_err:>6.1f}"
        )
        lines.append(sep)

    return "\n".join(lines)


def make_summary_panel(batch_dir: Path, rows: list[dict], out_path: Path):
    """Tile all result_panel.png images into one large grid."""
    panels = []
    for r in rows:
        p = batch_dir / r["experiment"] / "result_panel.png"
        if p.exists():
            panels.append((r["experiment"], np.array(Image.open(p).convert("RGB"))))

    if not panels:
        return

    n = len(panels)
    ncols = 5
    nrows = math.ceil(n / ncols)

    ph, pw = panels[0][1].shape[:2]
    label_h = 28
    cell_h = ph + label_h
    cell_w = pw

    canvas = np.full((nrows * cell_h, ncols * cell_w, 3), 30, dtype=np.uint8)

    for idx, (name, img) in enumerate(panels):
        row, col = divmod(idx, ncols)
        y0 = row * cell_h + label_h
        x0 = col * cell_w
        h, w = img.shape[:2]
        canvas[y0:y0+h, x0:x0+w] = img

    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    for idx, (name, _) in enumerate(panels):
        row, col = divmod(idx, ncols)
        x0 = col * cell_w + 4
        y0 = row * cell_h + 4
        draw.text((x0, y0), name, fill=(255, 220, 80))

    pil.save(out_path)
    print(f"  Summary panel → {out_path}  ({ncols}×{nrows} grid, {n} experiments)")


def run(args):
    batch_dir = Path(args.batch_dir)
    rows = load_metrics(batch_dir)

    if not rows:
        print(f"No metrics.json files found under {batch_dir}")
        sys.exit(1)

    print(f"\nFound {len(rows)} completed experiments.\n")
    table = print_table(rows)
    print(table)

    # Save text summary
    txt_path = batch_dir / "summary.txt"
    txt_path.write_text(table + "\n")
    print(f"\n  Summary table → {txt_path}")

    # Save CSV
    csv_path = batch_dir / "summary.csv"
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV → {csv_path}")

    # Save per-object stats
    objects = {}
    for r in rows:
        obj = r["object"]
        objects.setdefault(obj, []).append(r)

    print("\n── Per-object averages ──────────────────────────────")
    for obj, obj_rows in sorted(objects.items()):
        valid = [r for r in obj_rows if r["psnr"] is not None]
        if valid:
            print(f"  {obj:<14}  PSNR {sum(r['psnr'] for r in valid)/len(valid):.2f} dB"
                  f"  err {sum(r['error_px'] for r in valid)/len(valid):.1f} px"
                  f"  ({len(valid)}/{len(obj_rows)} trials)")

    # Summary panel image
    make_summary_panel(batch_dir, rows, batch_dir / "summary_panel.png")

    # Save JSON summary
    summary = {
        "n_total": len(rows),
        "n_valid": len([r for r in rows if r["psnr"] is not None]),
        "mean_psnr": (sum(r["psnr"] for r in rows if r["psnr"]) /
                      max(1, len([r for r in rows if r["psnr"]]))),
        "mean_ssim": (sum(r["ssim"] for r in rows if r["ssim"]) /
                      max(1, len([r for r in rows if r["ssim"]]))),
        "mean_error_px": (sum(r["error_px"] for r in rows if r["error_px"]) /
                          max(1, len([r for r in rows if r["error_px"]]))),
        "experiments": rows,
    }
    with open(batch_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_dir", required=True,
                        help="Directory containing per-experiment subdirs with metrics.json")
    run(parser.parse_args())
