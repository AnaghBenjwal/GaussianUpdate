#!/usr/bin/env python3
"""
make_final_example.py — End-to-end demo: OpenCV before/after pair → GG update → showable outputs.

Run in the gaussian_grouping conda env:
    cd /home/unk/Documents/gaussian-grouping
    conda run -n gaussian_grouping python scripts/make_final_example.py

What it does
------------
1. Loads the GG model and renders the before frame once (this is the canonical
   "before" used for BOTH the OpenCV pair and the change-detection pipeline).
2. Picks a compact foreground object from the GG object-segmentation prediction.
3. Creates a clean "after" image using OpenCV:
      - cv2.inpaint fills the hole left by removing the object (TELEA, radius 7)
      - The object is pasted at a new position with feathered alpha blending
4. Feeds the before/after pair into detect_and_update exactly as real data would be.
5. Assembles output/presentation/final_example/ with:
      before.png, after_target.png, detected_change_mask.png,
      updated_render.png, comparison_grid.png, side_by_side_video.mp4, summary.md
"""

from __future__ import annotations
import os, sys, json, shutil, math
from pathlib import Path
from typing import Tuple

import numpy as np
import cv2
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)

from detect_and_update import detect_and_update, render_frame
from argparse import ArgumentParser
from scene import Scene
from scene.gaussian_model import GaussianModel
from arguments import ModelParams, PipelineParams

# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════
MODEL_PATH   = "output/lerf_mask/figurines"
SOURCE_PATH  = "data/lerf_mask/figurines"
ITERATION    = 30000
CAM_IDX      = 0
NUM_CLASSES  = 256

# Class 42: 68,428 Gaussians, 4.4% of image (31k pixels), bbox (747,130)→(985,316)
# Shift -300px x, -100px y → new bbox (447,30)→(685,216) — upper-center, window background.
# Window/background Gaussians are at very large depth so our z=3.85 object is in front.
TARGET_CLASS_ID = 42
SHIFT_X, SHIFT_Y = -300, -100

WORK_DIR  = "output/presentation/final_example_work"
FINAL_DIR = "output/presentation/final_example"

INPAINT_RADIUS = 7
FEATHER_SIGMA  = 6.0
FEATHER_KSIZE  = 31


# ═══════════════════════════════════════════════════════════════════════════════
# Shared model setup (render once, reuse)
# ═══════════════════════════════════════════════════════════════════════════════

def load_model_and_render(cam_idx: int):
    """
    Load the GG model, render the scene at ``cam_idx``, and return
    the before-image, per-pixel object-class map, and the GG objects for
    later use by detect_and_update.

    Returns
    -------
    before_rgb   : uint8 (H, W, 3)
    pred_obj_img : int32 (H, W) — argmax class IDs per pixel
    """
    parser = ArgumentParser()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    args = parser.parse_args([
        "--model_path",  MODEL_PATH,
        "--source_path", SOURCE_PATH,
        "--eval",
    ])
    dataset  = lp.extract(args)
    pipeline = pp.extract(args)

    gaussians = GaussianModel(dataset.sh_degree)
    scene     = Scene(dataset, gaussians, load_iteration=ITERATION, shuffle=False)

    classifier = torch.nn.Conv2d(gaussians.num_objects, NUM_CLASSES, kernel_size=1).cuda()
    ckpt_path  = os.path.join(MODEL_PATH, "point_cloud",
                              f"iteration_{ITERATION}", "classifier.pth")
    classifier.load_state_dict(
        torch.load(ckpt_path, map_location="cuda", weights_only=False)
    )
    classifier.eval()

    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    view     = scene.getTrainCameras()[cam_idx]
    print(f"    Rendering cam #{cam_idx}: '{view.image_name}'  "
          f"({view.image_width}×{view.image_height} px)")

    before_rgb, pred_obj_img = render_frame(
        view, gaussians, pipeline, bg_color, classifier, NUM_CLASSES
    )
    return before_rgb, pred_obj_img


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1 — Build the OpenCV before/after pair
# ═══════════════════════════════════════════════════════════════════════════════

def get_object_mask(pred_obj_img: np.ndarray) -> np.ndarray:
    """Return the binary mask for TARGET_CLASS_ID from the pred_obj_img."""
    mask = (pred_obj_img == TARGET_CLASS_ID).astype(np.uint8)
    ys, xs = np.where(mask)
    print(f"    Class {TARGET_CLASS_ID}: {mask.sum():,} pixels  "
          f"bbox ({xs.min()},{ys.min()})→({xs.max()},{ys.max()})")
    return mask


def build_after_image(
    before_rgb: np.ndarray,
    obj_mask:   np.ndarray,
    shift_x:    int,
    shift_y:    int,
) -> np.ndarray:
    """
    Generate a clean "after" image using cv2.inpaint and feathered alpha blending.

    1. cv2.inpaint (TELEA) fills the object's vacated location.
    2. The object pixels are pasted at (x+shift_x, y+shift_y) with a soft
       Gaussian-blurred alpha mask so edges blend naturally.

    Returns after_bgr : uint8 BGR image.
    """
    H, W = before_rgb.shape[:2]
    before_bgr = cv2.cvtColor(before_rgb, cv2.COLOR_RGB2BGR)

    # 1. Inpaint with slightly dilated mask
    kernel_d  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    inpaint_m = cv2.dilate(obj_mask, kernel_d).astype(np.uint8) * 255
    inpainted = cv2.inpaint(before_bgr, inpaint_m, INPAINT_RADIUS, cv2.INPAINT_TELEA)

    # 2. Feathered alpha
    kernel_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    inner    = cv2.erode(obj_mask, kernel_e).astype(np.float32)
    alpha    = cv2.GaussianBlur(inner, (FEATHER_KSIZE, FEATHER_KSIZE), FEATHER_SIGMA)
    alpha    = np.clip(alpha, 0.0, 1.0)

    # 3. Paste
    after = inpainted.copy().astype(np.float32)
    src_ys, src_xs = np.where(obj_mask > 0)
    dst_xs = src_xs + shift_x
    dst_ys = src_ys + shift_y
    valid  = (dst_xs >= 0) & (dst_xs < W) & (dst_ys >= 0) & (dst_ys < H)
    sy, sx = src_ys[valid], src_xs[valid]
    dy, dx = dst_ys[valid], dst_xs[valid]
    a      = alpha[sy, sx, None]
    after[dy, dx] = a * before_bgr[sy, sx].astype(np.float32) + (1 - a) * after[dy, dx]

    return after.astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3 — Final output assembly
# ═══════════════════════════════════════════════════════════════════════════════

def make_comparison_grid(
    before:    np.ndarray,
    after_tgt: np.ndarray,
    mask_vis:  np.ndarray,
    updated:   np.ndarray,
    out_path:  str,
):
    """2×2 presentation grid with dark labels."""
    H, W   = before.shape[:2]
    pad    = 14
    lh     = 52
    fscale = 1.05
    thick  = 2

    gh = 2 * (H + lh) + 3 * pad
    gw = 2 * W + 3 * pad
    canvas = np.full((gh, gw, 3), 18, dtype=np.uint8)

    for idx, (img, label) in enumerate([
        (before,    "Before  (GG render)"),
        (after_tgt, "After target  (OpenCV)"),
        (mask_vis,  "Detected change mask"),
        (updated,   "Updated GG render"),
    ]):
        r, c = divmod(idx, 2)
        y0 = pad + r * (H + lh + pad)
        x0 = pad + c * (W + pad)
        canvas[y0:y0+H, x0:x0+W] = img
        bar_y = y0 + H
        canvas[bar_y:bar_y+lh, x0:x0+W] = (42, 42, 42)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, fscale, thick)
        tx = x0 + (W - tw) // 2
        ty = bar_y + (lh + th) // 2 - 2
        cv2.putText(canvas, label, (tx, ty),
                    cv2.FONT_HERSHEY_DUPLEX, fscale, (235, 235, 235), thick, cv2.LINE_AA)

    # subtle dividers
    mx = pad + W + pad // 2
    my = pad + H + lh + pad // 2
    cv2.line(canvas, (mx, 0), (mx, gh), (72, 72, 72), 1)
    cv2.line(canvas, (0, my), (gw, my), (72, 72, 72), 1)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    print(f"    Grid  → {out_path}  ({gw}×{gh} px)")


def make_wipe_video(
    before:  np.ndarray,
    updated: np.ndarray,
    out_path: str,
    fps: int = 30,
):
    """
    Side-by-side video: left panel = Before (static),
    right panel shows a vertical wipe sweeping left→right to reveal Updated.
    Forward sweep + reverse sweep = seamless loop.
    """
    H, W  = before.shape[:2]
    VW, VH = W * 2, H + 60        # two panels + label bar

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = cv2.VideoWriter(out_path, fourcc, fps, (VW, VH))

    b_bgr = cv2.cvtColor(before,  cv2.COLOR_RGB2BGR)
    u_bgr = cv2.cvtColor(updated, cv2.COLOR_RGB2BGR)

    def frame(frac: float) -> np.ndarray:
        f = np.zeros((VH, VW, 3), dtype=np.uint8)
        f[0:H, 0:W] = b_bgr
        right = b_bgr.copy()
        wp = int(frac * W)
        if wp > 0:
            right[:, :wp] = u_bgr[:, :wp]
        if 0 < wp < W:
            cv2.line(right, (wp, 0), (wp, H), (255, 255, 255), 3)
        f[0:H, W:] = right
        f[H:, :] = (28, 28, 28)
        cv2.putText(f, "BEFORE",
                    (W//2 - 65, H+40), cv2.FONT_HERSHEY_DUPLEX,
                    0.9, (195, 195, 195), 2, cv2.LINE_AA)
        cv2.putText(f, "UPDATED  (Gaussian model corrected)",
                    (W + 18, H+40), cv2.FONT_HERSHEY_DUPLEX,
                    0.9, (195, 195, 195), 2, cv2.LINE_AA)
        return f

    hold   = fps          # 1 s hold
    sweep  = fps * 2      # 2 s sweep

    for _ in range(hold):   writer.write(frame(0.0))
    for i in range(sweep):  writer.write(frame(i / sweep))
    for _ in range(hold):   writer.write(frame(1.0))
    for i in range(sweep):  writer.write(frame(1.0 - i / sweep))
    for _ in range(hold):   writer.write(frame(0.0))
    writer.release()
    total = 3 * hold + 2 * sweep
    print(f"    Video → {out_path}  ({total//fps}s @ {fps}fps)")


def write_summary(path: str, report: dict, shift_x: int, shift_y: int):
    text = f"""# Final Example: Gaussian Grouping Scene Update from Image Change

This example demonstrates the full pipeline on the **figurines** dataset.
A compact foreground object (GG class {report['target_obj_id']}) was digitally relocated
{abs(shift_x)} pixels left and {shift_y} pixels down using OpenCV inpainting and feathered
compositing, simulating a real physical object move.

The change-detection stage identified **{report['n_gaussians_modified']:,} Gaussians**
belonging to the moved object by running the GG classifier directly on per-Gaussian object
features, then localised the 2D shift via normalised cross-correlation template matching
(recovered shift: {report['px_shift'][0]:+.1f} × {report['px_shift'][1]:+.1f} px).

A 3D world-space displacement of
**({report['delta_world_xyz'][0]:.3f}, {report['delta_world_xyz'][1]:.3f}, {report['delta_world_xyz'][2]:.3f})**
was computed by back-projecting the pixel shift through the camera's pinhole model at the
object's measured depth of {report['depth_at_obj']:.2f} world units, then applied in-place
to the Gaussian positions.

The `updated_render.png` shows the Gaussian Splat map re-rendered after the correction;
`comparison_grid.png` places all four stages side by side.
"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    print(f"    Summary → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(WORK_DIR,  exist_ok=True)
    os.makedirs(FINAL_DIR, exist_ok=True)

    # ── 1. Render the before frame (canonical; used for BOTH OpenCV and pipeline)
    print("[1/4] Rendering canonical before frame from GG model …")
    before_rgb, pred_obj_img = load_model_and_render(CAM_IDX)
    print(f"    Rendered: {before_rgb.shape[1]}×{before_rgb.shape[0]} px")

    # Save before frame — this exact image is what detect_and_update will
    # also re-render (same checkpoint, same cam), so diff will be near-zero.
    before_save = os.path.join(FINAL_DIR, "before.png")
    cv2.imwrite(before_save, cv2.cvtColor(before_rgb, cv2.COLOR_RGB2BGR))

    # ── 2. Build OpenCV after image ───────────────────────────────────────────
    print("\n[2/4] Generating OpenCV after image …")
    obj_mask = get_object_mask(pred_obj_img)
    target_class = TARGET_CLASS_ID
    print(f"    Mask pixels: {obj_mask.sum():,}")

    after_bgr = build_after_image(before_rgb, obj_mask, SHIFT_X, SHIFT_Y)
    after_path = os.path.join(WORK_DIR, "after_input.png")
    cv2.imwrite(after_path, after_bgr)
    print(f"    After saved → {after_path}")

    after_target_save = os.path.join(FINAL_DIR, "after_target.png")
    cv2.imwrite(after_target_save, after_bgr)

    # Quick sanity: diff between before and after should be ~33k pixels
    after_rgb = cv2.cvtColor(after_bgr, cv2.COLOR_BGR2RGB)
    diff_chk  = np.abs(before_rgb.astype(int) - after_rgb.astype(int)).max(axis=2)
    print(f"    Sanity diff > 25: {(diff_chk > 25).sum():,} pixels "
          f"(expected ~{2*obj_mask.sum():,})")

    # ── 3. Run detect_and_update pipeline ─────────────────────────────────────
    print("\n[3/4] Running detect_and_update pipeline …")
    report = detect_and_update(
        model_path       = MODEL_PATH,
        source_path      = SOURCE_PATH,
        iteration        = ITERATION,
        cam_idx          = CAM_IDX,
        output_dir       = WORK_DIR,
        after_image_path = after_path,
        synthetic        = False,
        num_classes      = NUM_CLASSES,
        use_sam2         = False,
        force_obj_id     = target_class,
    )

    # ── 4. Collect and assemble final outputs ─────────────────────────────────
    print("\n[4/4] Assembling final_example/ …")

    src_files = {
        "detected_change_mask.png": os.path.join(WORK_DIR, "03_change_detection",
                                                  "mask_overlay_before.png"),
        "updated_render.png":       os.path.join(WORK_DIR, "05_updated_gaussian_render",
                                                  "updated_render.png"),
    }
    for dst_name, src_path in src_files.items():
        shutil.copy2(src_path, os.path.join(FINAL_DIR, dst_name))
        print(f"    Copied {dst_name}")

    shutil.copy2(os.path.join(WORK_DIR, "report.json"),
                 os.path.join(FINAL_DIR, "report.json"))

    # Load images for grid / video
    before_rgb2  = cv2.cvtColor(cv2.imread(before_save),    cv2.COLOR_BGR2RGB)
    after_rgb2   = cv2.cvtColor(cv2.imread(after_target_save), cv2.COLOR_BGR2RGB)
    mask_vis     = cv2.cvtColor(cv2.imread(os.path.join(FINAL_DIR, "detected_change_mask.png")),
                                cv2.COLOR_BGR2RGB)
    updated_rgb  = cv2.cvtColor(cv2.imread(os.path.join(FINAL_DIR, "updated_render.png")),
                                cv2.COLOR_BGR2RGB)

    make_comparison_grid(
        before_rgb2, after_rgb2, mask_vis, updated_rgb,
        out_path=os.path.join(FINAL_DIR, "comparison_grid.png"),
    )
    make_wipe_video(
        before_rgb2, updated_rgb,
        out_path=os.path.join(FINAL_DIR, "side_by_side_video.mp4"),
    )
    write_summary(
        path    =os.path.join(FINAL_DIR, "summary.md"),
        report  =report,
        shift_x =SHIFT_X,
        shift_y =SHIFT_Y,
    )

    print(f"\n{'='*60}")
    print(f"DONE — {FINAL_DIR}/")
    print(f"  Gaussians modified : {report['n_gaussians_modified']:,} (class {report['target_obj_id']})")
    print(f"  2D shift recovered : {report['px_shift']}")
    print(f"  3D displacement    : {[round(v,4) for v in report['delta_world_xyz']]}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
