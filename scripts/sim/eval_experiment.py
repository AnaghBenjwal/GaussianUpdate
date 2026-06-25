#!/usr/bin/env python3
"""
eval_experiment.py — Run one experiment end-to-end and compute PSNR.

After detect_and_update runs:
  - GG renders the scene (before update) from cam0
  - GG renders the scene (after update) from cam0
  - AI2THOR renders the "after" ground truth from cam0
  - We compute PSNR(updated_render, gt_after)

Run in gaussian_grouping env:
    conda run -n gaussian_grouping python scripts/sim/eval_experiment.py \
        --model_path output/sim/FloorPlan1 \
        --source_path data/sim/FloorPlan1 \
        --before_after_dir data/sim/FloorPlan1/before_after/apple_right \
        --iteration 10000 \
        --num_classes 32 \
        --output output/sim_experiments/FloorPlan1/apple_right
"""
from __future__ import annotations
import argparse, json, math, os, sys
from pathlib import Path

import numpy as np
import torch
import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from argparse import ArgumentParser
from scene import Scene
from scene.gaussian_model import GaussianModel
from arguments import ModelParams, PipelineParams
sys.path.insert(0, str(ROOT / "scripts"))
from detect_and_update import render_frame, detect_and_update


def psnr(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """PSNR between two uint8 RGB images."""
    mse = np.mean((img_a.astype(np.float64) - img_b.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return float("inf")
    return float(20.0 * math.log10(255.0) - 10.0 * math.log10(mse))


def ssim_simple(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """Approximate SSIM (no skimage dependency)."""
    a = img_a.astype(np.float64)
    b = img_b.astype(np.float64)
    mu_a, mu_b = a.mean(), b.mean()
    sig_a  = a.std()
    sig_b  = b.std()
    sig_ab = ((a - mu_a) * (b - mu_b)).mean()
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return float(
        (2 * mu_a * mu_b + c1) * (2 * sig_ab + c2) /
        ((mu_a ** 2 + mu_b ** 2 + c1) * (sig_a ** 2 + sig_b ** 2 + c2))
    )


def run(args):
    ba_dir = Path(args.before_after_dir)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    after_path = str(ba_dir / "after.png")
    ba_meta = json.load(open(ba_dir / "meta.json"))
    cls_idx  = ba_meta["class_idx"]
    gt_dx    = ba_meta.get("gt_pixel_dx")
    gt_dy    = ba_meta.get("gt_pixel_dy")

    print(f"Running detect_and_update...")
    print(f"  Object class: {cls_idx}  GT shift: ({gt_dx:.1f}, {gt_dy:.1f}) px")

    report = detect_and_update(
        model_path       = args.model_path,
        source_path      = args.source_path,
        iteration        = args.iteration,
        cam_idx          = 0,
        output_dir       = str(out_dir),
        after_image_path = after_path,
        force_obj_id     = cls_idx,
        num_classes      = args.num_classes,
        sh_degree        = 3,
        use_sam2         = False,
    )

    # ── Load result images ───────────────────────────────────────────────────
    # detect_and_update saves to subdirs of out_dir:
    before_render_file = out_dir / "01_gaussian_map_render" / "before_render.png"
    updated_render_file = out_dir / "05_updated_gaussian_render" / "updated_render.png"

    gt_after = np.array(Image.open(after_path).convert("RGB"))

    if updated_render_file.exists():
        updated_render = np.array(Image.open(updated_render_file).convert("RGB"))
        if updated_render.shape[:2] != gt_after.shape[:2]:
            updated_render = np.array(
                Image.fromarray(updated_render).resize(
                    (gt_after.shape[1], gt_after.shape[0]), Image.LANCZOS
                )
            )
        p = psnr(updated_render, gt_after)
        s = ssim_simple(updated_render, gt_after)
        print(f"\n  PSNR(updated render vs GT after): {p:.2f} dB")
        print(f"  SSIM(updated render vs GT after): {s:.4f}")
    else:
        p, s, updated_render = None, None, None
        print("  No updated render found for PSNR")

    # ── Build result panel ────────────────────────────────────────────────────
    panels = {}

    gt_before_path = ba_dir / "before.png"
    if gt_before_path.exists():
        panels["gt_before"] = np.array(Image.open(gt_before_path).convert("RGB"))
    if before_render_file.exists():
        panels["before_render"] = np.array(Image.open(before_render_file).convert("RGB"))
    panels["gt_after"] = gt_after
    if updated_render is not None:
        panels["updated_render"] = updated_render

    if len(panels) >= 2:
        H = max(v.shape[0] for v in panels.values())
        W = max(v.shape[1] for v in panels.values())
        order = ["gt_before", "before_render", "gt_after", "updated_render"]
        imgs = []
        labels = []
        for key in order:
            if key in panels:
                img = panels[key]
                if img.shape[:2] != (H, W):
                    img = np.array(Image.fromarray(img).resize((W, H), Image.LANCZOS))
                imgs.append(img)
                labels.append(key.replace("_", " ").title())

        # Add text labels
        labeled = []
        for img, label in zip(imgs, labels):
            canvas = img.copy()
            cv2.putText(canvas, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 0), 2, cv2.LINE_AA)
            labeled.append(canvas)

        panel = np.concatenate(labeled, axis=1)
        panel_path = out_dir / "result_panel.png"
        Image.fromarray(panel).save(panel_path)
        print(f"  Result panel → {panel_path}")

    # ── Save metrics ─────────────────────────────────────────────────────────
    metrics = {
        "object_class": cls_idx,
        "gt_pixel_dx": gt_dx,
        "gt_pixel_dy": gt_dy,
        "detected_pixel_dx": report.get("px_shift", [None])[0],
        "detected_pixel_dy": report.get("px_shift", [None, None])[1],
        "n_gaussians_modified": report.get("n_gaussians_modified"),
        "psnr_updated_vs_gt_after": p,
        "ssim_updated_vs_gt_after": s,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== METRICS ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")
    print(f"All results in {out_dir}/")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",       required=True)
    parser.add_argument("--source_path",      required=True)
    parser.add_argument("--before_after_dir", required=True)
    parser.add_argument("--iteration",        type=int, default=10000)
    parser.add_argument("--num_classes",      type=int, default=32)
    parser.add_argument("--output",           required=True)
    args = parser.parse_args()
    run(args)
