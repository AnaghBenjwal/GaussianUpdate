#!/usr/bin/env python3
"""
leaderboard.py — Compare multiple batch experiment runs side-by-side.

Usage:
    python scripts/sim/leaderboard.py \
        --runs output/sim_experiments/FloorPlan1/iter10000 \
                output/sim_experiments/FloorPlan1/iter30000 \
        --labels "10k iters" "30k iters" \
        --out output/sim_experiments/FloorPlan1/leaderboard

Produces:
    leaderboard.txt   — side-by-side table per experiment
    leaderboard.csv   — one row per (run, experiment) for plotting
    leaderboard.png   — bar charts: PSNR and detection error per run
"""
from __future__ import annotations
import argparse, csv, json, math, os
from pathlib import Path

import numpy as np


def load_run(run_dir: Path) -> list[dict]:
    rows = []
    for mf in sorted(run_dir.glob("*/metrics.json")):
        m = json.load(open(mf))
        rows.append({
            "experiment": mf.parent.name,
            "gt_dx":      m.get("gt_pixel_dx"),
            "gt_dy":      m.get("gt_pixel_dy"),
            "det_dx":     m.get("detected_pixel_dx"),
            "det_dy":     m.get("detected_pixel_dy"),
            "error_px":   m.get("error_px") or _err(m),
            "psnr":       m.get("psnr_updated_vs_gt_after"),
            "ssim":       m.get("ssim_updated_vs_gt_after"),
            "n_gaussians": m.get("n_gaussians_modified"),
        })
    return rows


def _err(m):
    gt_dx  = m.get("gt_pixel_dx")
    gt_dy  = m.get("gt_pixel_dy")
    det_dx = m.get("detected_pixel_dx")
    det_dy = m.get("detected_pixel_dy")
    if any(v is None for v in [gt_dx, gt_dy, det_dx, det_dy]):
        return None
    return math.sqrt((gt_dx - det_dx)**2 + (gt_dy - det_dy)**2)


def print_leaderboard(runs: list[list[dict]], labels: list[str]) -> str:
    # Collect all experiment names (union)
    all_exps = []
    seen = set()
    for run in runs:
        for r in run:
            if r["experiment"] not in seen:
                all_exps.append(r["experiment"])
                seen.add(r["experiment"])

    # Index runs by experiment name
    indexed = [{r["experiment"]: r for r in run} for run in runs]

    col_w = 20
    hdr = f"{'Experiment':<22}"
    for label in labels:
        hdr += f"  {'PSNR':>6}  {'Err px':>7}  {'SSIM':>5}  |"
    sep = "─" * len(hdr)

    lines = [sep, hdr, sep]
    prev_obj = None

    for exp in all_exps:
        obj = exp.rsplit("_", 1)[0]
        if prev_obj and obj != prev_obj:
            lines.append("")
        prev_obj = obj

        line = f"{exp:<22}"
        for idx_run in indexed:
            r = idx_run.get(exp)
            if r and r["psnr"] is not None:
                line += f"  {r['psnr']:>6.2f}  {r['error_px'] or 0:>7.1f}  {r['ssim']:>5.3f}  |"
            else:
                line += f"  {'—':>6}  {'—':>7}  {'—':>5}  |"
        lines.append(line)

    lines.append(sep)

    # Summary row
    summary_line = f"{'MEAN':<22}"
    for run in runs:
        valid = [r for r in run if r["psnr"] is not None]
        if valid:
            avg_psnr = sum(r["psnr"] for r in valid) / len(valid)
            avg_err  = sum(r["error_px"] for r in valid if r["error_px"]) / max(1, len([r for r in valid if r["error_px"]]))
            avg_ssim = sum(r["ssim"] for r in valid) / len(valid)
            summary_line += f"  {avg_psnr:>6.2f}  {avg_err:>7.1f}  {avg_ssim:>5.3f}  |"
        else:
            summary_line += f"  {'—':>6}  {'—':>7}  {'—':>5}  |"
    lines.append(summary_line)
    lines.append(sep)

    # Per-label header legend
    lines.append("")
    for i, label in enumerate(labels):
        valid = [r for r in runs[i] if r["psnr"] is not None]
        n = len(valid)
        if n:
            avg_psnr = sum(r["psnr"] for r in valid) / n
            avg_err  = sum(r["error_px"] for r in valid if r["error_px"]) / max(1, len([r for r in valid if r["error_px"]]))
            good = sum(1 for r in valid if r["error_px"] is not None and r["error_px"] < 20)
            lines.append(
                f"  [{label}]  {n} trials  "
                f"PSNR {avg_psnr:.2f} dB  "
                f"mean err {avg_err:.1f} px  "
                f"good detections (<20px err): {good}/{n}"
            )
    lines.append(sep)

    return "\n".join(lines)


def save_csv(runs, labels, out_path):
    rows = []
    for label, run in zip(labels, runs):
        for r in run:
            rows.append({"run": label, **r})
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run"] + list(rows[0].keys())[1:])
        w.writeheader()
        w.writerows(rows)


def save_charts(runs, labels, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping chart")
        return

    objects = ["apple", "book", "bowl", "bread", "coffeemachine"]
    moves   = ["right", "left", "forward", "back", "diagonal"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(len(objects))
    width = 0.8 / len(runs)
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

    for metric, ax, title, better in [
        ("psnr", axes[0], "PSNR (dB) — higher is better", "high"),
        ("error_px", axes[1], "Detection error (px) — lower is better", "low"),
    ]:
        for i, (label, run) in enumerate(zip(labels, runs)):
            vals = []
            for obj in objects:
                obj_rows = [r for r in run if r["experiment"].startswith(obj + "_") and r[metric] is not None]
                vals.append(sum(r[metric] for r in obj_rows) / len(obj_rows) if obj_rows else 0)
            offset = (i - len(runs) / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=label, color=colors[i % len(colors)])

        ax.set_xticks(x)
        ax.set_xticklabels(objects, rotation=15, ha="right")
        ax.set_title(title)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Charts → {out_path}")


def run(args):
    run_dirs  = [Path(p) for p in args.runs]
    labels    = args.labels if args.labels else [d.name for d in run_dirs]
    out_dir   = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = [load_run(d) for d in run_dirs]

    print(f"\nLoaded {len(runs)} runs:")
    for label, run in zip(labels, runs):
        print(f"  [{label}]  {len(run)} experiments")

    table = print_leaderboard(runs, labels)
    print("\n" + table)

    txt_path = out_dir / "leaderboard.txt"
    txt_path.write_text(table + "\n")
    print(f"\n  Leaderboard table → {txt_path}")

    save_csv(runs, labels, out_dir / "leaderboard.csv")
    print(f"  CSV → {out_dir / 'leaderboard.csv'}")

    save_charts(runs, labels, out_dir / "leaderboard.png")

    # Save JSON for easy programmatic access
    summary = {}
    for label, run in zip(labels, runs):
        valid = [r for r in run if r["psnr"] is not None]
        summary[label] = {
            "n_trials": len(valid),
            "mean_psnr": sum(r["psnr"] for r in valid) / len(valid) if valid else None,
            "mean_ssim": sum(r["ssim"] for r in valid) / len(valid) if valid else None,
            "mean_error_px": sum(r["error_px"] for r in valid if r["error_px"]) / max(1, len([r for r in valid if r["error_px"]])) if valid else None,
            "good_detections": sum(1 for r in valid if r["error_px"] is not None and r["error_px"] < 20),
            "experiments": run,
        }
    with open(out_dir / "leaderboard.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  JSON → {out_dir / 'leaderboard.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs",   nargs="+", required=True, help="Batch result directories")
    parser.add_argument("--labels", nargs="+", help="Display label for each run (same order as --runs)")
    parser.add_argument("--out",    required=True, help="Output directory for leaderboard files")
    run(parser.parse_args())
