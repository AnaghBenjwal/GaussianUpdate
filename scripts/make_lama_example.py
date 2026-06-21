#!/usr/bin/env python3
"""
make_lama_example.py — End-to-end demo using LaMa inpainting instead of cv2.inpaint.

Identical pipeline to make_final_example.py, except the background reconstruction
after object removal uses LaMa (photo-realistic) instead of OpenCV TELEA (texture copy).

Outputs go to output/presentation/final_example_lama/
(kept completely separate from the other two pipelines)

Run in the gaussian_grouping conda env:
    cd /home/unk/Documents/gaussian-grouping
    conda run -n gaussian_grouping python scripts/make_lama_example.py
"""

from __future__ import annotations
import os, sys, json, shutil, subprocess, tempfile
from pathlib import Path

import numpy as np
import cv2
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)

from detect_and_update import render_frame, detect_and_update
from argparse import ArgumentParser
from scene import Scene
from scene.gaussian_model import GaussianModel
from arguments import ModelParams, PipelineParams

# ═══════════════════════════════════════════════════════════════════════════════
# Config — same object/shift as make_final_example.py for direct comparison
# ═══════════════════════════════════════════════════════════════════════════════
MODEL_PATH  = "output/lerf_mask/figurines"
SOURCE_PATH = "data/lerf_mask/figurines"
ITERATION   = 30000
CAM_IDX     = 0
NUM_CLASSES = 256

TARGET_CLASS_ID  = 42
SHIFT_X, SHIFT_Y = -300, -100

FEATHER_SIGMA = 6.0
FEATHER_KSIZE = 31
LAMA_DILATE   = 15   # px dilation on mask before LaMa (catches border fringe)

WORK_DIR  = "output/presentation/final_example_lama_work"
FINAL_DIR = "output/presentation/final_example_lama"


# ═══════════════════════════════════════════════════════════════════════════════
# Load GG model and render
# ═══════════════════════════════════════════════════════════════════════════════

def load_model_and_render(cam_idx: int):
    parser = ArgumentParser()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    args = parser.parse_args([
        "--model_path",  MODEL_PATH,
        "--source_path", SOURCE_PATH,
        "--eval",
    ])
    dataset   = lp.extract(args)
    pipeline  = pp.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene     = Scene(dataset, gaussians, load_iteration=ITERATION, shuffle=False)
    clf       = torch.nn.Conv2d(gaussians.num_objects, NUM_CLASSES, kernel_size=1).cuda()
    clf.load_state_dict(torch.load(
        os.path.join(MODEL_PATH, "point_cloud", f"iteration_{ITERATION}", "classifier.pth"),
        map_location="cuda", weights_only=False,
    ))
    clf.eval()
    bg   = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    view = scene.getTrainCameras()[cam_idx]
    print(f"    Rendering cam #{cam_idx}: '{view.image_name}'  "
          f"({view.image_width}×{view.image_height} px)")
    return render_frame(view, gaussians, pipeline, bg, clf, NUM_CLASSES)


# ═══════════════════════════════════════════════════════════════════════════════
# LaMa inpainting via subprocess (dev env)
# ═══════════════════════════════════════════════════════════════════════════════

def run_lama(image_path: str, mask_path: str, output_path: str,
             dilate: int = LAMA_DILATE) -> bool:
    """Call lama_inpaint.py in the dev conda env."""
    cmd = [
        "conda", "run", "-n", "dev", "--no-capture-output",
        "python3", str(ROOT / "scripts" / "lama_inpaint.py"),
        "--image",  image_path,
        "--mask",   mask_path,
        "--output", output_path,
        "--dilate", str(dilate),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    [LaMa error] {result.stderr.strip()[-300:]}")
        return False
    for line in result.stdout.splitlines():
        print(f"    [LaMa] {line}")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# Build after image: LaMa background + feathered object paste
# ═══════════════════════════════════════════════════════════════════════════════

def build_after_image_lama(
    before_rgb: np.ndarray,
    obj_mask:   np.ndarray,
    lama_bg_path: str,
    shift_x: int,
    shift_y: int,
) -> np.ndarray:
    """
    Paste the object at its new position onto the LaMa-inpainted background.
    Uses feathered alpha blending (same as make_final_example.py).
    """
    H, W = before_rgb.shape[:2]
    before_bgr = cv2.cvtColor(before_rgb, cv2.COLOR_RGB2BGR)

    # Load LaMa background (BGR)
    lama_bg = cv2.imread(lama_bg_path)

    # Feathered alpha for the object
    kernel_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    inner    = cv2.erode(obj_mask, kernel_e).astype(np.float32)
    alpha    = cv2.GaussianBlur(inner, (FEATHER_KSIZE, FEATHER_KSIZE), FEATHER_SIGMA)
    alpha    = np.clip(alpha, 0.0, 1.0)

    after = lama_bg.copy().astype(np.float32)
    src_ys, src_xs = np.where(obj_mask > 0)
    dst_xs = src_xs + shift_x
    dst_ys = src_ys + shift_y
    valid  = (dst_xs >= 0) & (dst_xs < W) & (dst_ys >= 0) & (dst_ys < H)
    sy, sx = src_ys[valid], src_xs[valid]
    dy, dx = dst_ys[valid], dst_xs[valid]
    a = alpha[sy, sx, None]
    after[dy, dx] = (
        a * before_bgr[sy, sx].astype(np.float32)
        + (1 - a) * after[dy, dx]
    )
    return after.astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
# Comparison grid
# ═══════════════════════════════════════════════════════════════════════════════

def make_comparison_grid(
    before:   np.ndarray,   # RGB
    cv2_after: np.ndarray,  # BGR (cv2 inpaint)
    lama_bg:  np.ndarray,   # BGR (LaMa background only)
    lama_after: np.ndarray, # BGR (LaMa + object pasted)
    updated:  np.ndarray,   # RGB
    out_path: str,
) -> None:
    """
    3-row grid:
      Row 1: Before | cv2 after (baseline)
      Row 2: LaMa background | LaMa after (with object)
      Row 3: Updated GG render (full width)
    """
    H, W   = before.shape[:2]
    pad    = 12
    lh     = 48
    fscale = 0.95
    thick  = 2

    # Convert everything to BGR for OpenCV
    before_bgr  = cv2.cvtColor(before,  cv2.COLOR_RGB2BGR)
    updated_bgr = cv2.cvtColor(updated, cv2.COLOR_RGB2BGR)

    panels = [
        (before_bgr,  "Before  (GG render)"),
        (cv2_after,   "After: cv2.inpaint  (baseline)"),
        (lama_bg,     "LaMa inpainted background"),
        (lama_after,  "After: LaMa  (new method)"),
        (updated_bgr, "Updated GG render"),
    ]

    # Layout: 2 columns × 2 rows + 1 full-width row
    gw = 2 * W + 3 * pad
    gh = 3 * (H + lh) + 4 * pad
    canvas = np.full((gh, gw, 3), 18, dtype=np.uint8)

    positions = [
        (0, 0), (0, 1),   # row 0: before | cv2 after
        (1, 0), (1, 1),   # row 1: lama bg | lama after
    ]
    for i, (r, c) in enumerate(positions):
        img, label = panels[i]
        y0 = pad + r * (H + lh + pad)
        x0 = pad + c * (W + pad)
        canvas[y0 : y0 + H, x0 : x0 + W] = img
        bar_y = y0 + H
        canvas[bar_y : bar_y + lh, x0 : x0 + W] = (42, 42, 42)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, fscale, thick)
        cv2.putText(canvas, label, (x0 + (W - tw) // 2, bar_y + (lh + th) // 2 - 2),
                    cv2.FONT_HERSHEY_DUPLEX, fscale, (235, 235, 235), thick, cv2.LINE_AA)

    # Divider between rows 0 and 1
    my1 = pad + H + lh + pad // 2
    my2 = pad + 2 * (H + lh) + 2 * pad - pad // 2
    mx  = pad + W + pad // 2
    cv2.line(canvas, (mx, 0), (mx, 2 * (H + lh + pad) + pad), (72, 72, 72), 1)
    cv2.line(canvas, (0, my1), (gw, my1), (72, 72, 72), 1)
    cv2.line(canvas, (0, my2), (gw, my2), (72, 72, 72), 1)

    # Full-width bottom panel: Updated GG render
    img, label = panels[4]
    y0 = pad + 2 * (H + lh + pad)
    canvas[y0 : y0 + H, pad : pad + W] = img  # left half
    # Right half: diff heatmap (lama_after vs updated)
    diff = cv2.absdiff(lama_after, updated_bgr)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    diff_color = cv2.applyColorMap(diff_gray * 3, cv2.COLORMAP_INFERNO)
    canvas[y0 : y0 + H, pad + W + pad : pad + 2 * W + pad] = diff_color
    for i2, (img2, lbl2) in enumerate([(img, "Updated GG render"),
                                        (diff_color, "Diff: LaMa after vs GG render")]):
        x0 = pad + i2 * (W + pad)
        bar_y = y0 + H
        canvas[bar_y : bar_y + lh, x0 : x0 + W] = (42, 42, 42)
        (tw, th), _ = cv2.getTextSize(lbl2, cv2.FONT_HERSHEY_DUPLEX, fscale, thick)
        cv2.putText(canvas, lbl2, (x0 + (W - tw) // 2, bar_y + (lh + th) // 2 - 2),
                    cv2.FONT_HERSHEY_DUPLEX, fscale, (235, 235, 235), thick, cv2.LINE_AA)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, canvas)
    print(f"    Grid  → {out_path}  ({gw}×{gh} px)")


def make_wipe_video(before: np.ndarray, updated: np.ndarray,
                    out_path: str, fps: int = 30) -> None:
    H, W   = before.shape[:2]
    VW, VH = W * 2, H + 60
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = cv2.VideoWriter(out_path, fourcc, fps, (VW, VH))
    b_bgr  = cv2.cvtColor(before,  cv2.COLOR_RGB2BGR)
    u_bgr  = cv2.cvtColor(updated, cv2.COLOR_RGB2BGR)

    def frame(frac):
        f = np.zeros((VH, VW, 3), dtype=np.uint8)
        f[0:H, 0:W] = b_bgr
        right = b_bgr.copy()
        wp = int(frac * W)
        if wp > 0:
            right[:, :wp] = u_bgr[:, :wp]
        if 0 < wp < W:
            cv2.line(right, (wp, 0), (wp, H), (255, 255, 255), 3)
        f[0:H, W:] = right
        f[H:, :]   = (28, 28, 28)
        cv2.putText(f, "BEFORE", (W // 2 - 65, H + 40),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (195, 195, 195), 2, cv2.LINE_AA)
        cv2.putText(f, "UPDATED  (LaMa inpainting + GG update)",
                    (W + 18, H + 40), cv2.FONT_HERSHEY_DUPLEX,
                    0.9, (195, 195, 195), 2, cv2.LINE_AA)
        return f

    hold, sweep = fps, fps * 2
    for _ in range(hold):  writer.write(frame(0.0))
    for i in range(sweep): writer.write(frame(i / sweep))
    for _ in range(hold):  writer.write(frame(1.0))
    for i in range(sweep): writer.write(frame(1.0 - i / sweep))
    for _ in range(hold):  writer.write(frame(0.0))
    writer.release()
    print(f"    Video → {out_path}  ({(3*hold+2*sweep)//fps}s @ {fps}fps)")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    os.makedirs(WORK_DIR,  exist_ok=True)
    os.makedirs(FINAL_DIR, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="gg_lama_")

    # ── 1. Render GG before frame ─────────────────────────────────────────────
    print("[1/6] Rendering canonical before frame …")
    before_rgb, pred_obj_img = load_model_and_render(CAM_IDX)
    print(f"    {before_rgb.shape[1]}×{before_rgb.shape[0]} px")

    before_save = os.path.join(FINAL_DIR, "before.png")
    cv2.imwrite(before_save, cv2.cvtColor(before_rgb, cv2.COLOR_RGB2BGR))

    # ── 2. Object mask ────────────────────────────────────────────────────────
    print("\n[2/6] Building object mask …")
    obj_mask = (pred_obj_img == TARGET_CLASS_ID).astype(np.uint8)
    ys, xs   = np.where(obj_mask)
    print(f"    Class {TARGET_CLASS_ID}: {obj_mask.sum():,} px  "
          f"bbox ({xs.min()},{ys.min()})→({xs.max()},{ys.max()})")

    # Save mask as npy for LaMa
    mask_path = os.path.join(tmp, "obj_mask.npy")
    np.save(mask_path, obj_mask)

    # Save before image for LaMa input
    before_for_lama = os.path.join(tmp, "before.png")
    cv2.imwrite(before_for_lama, cv2.cvtColor(before_rgb, cv2.COLOR_RGB2BGR))

    # ── 3. LaMa inpainting (runs in dev env, GPU freed after step 1) ─────────
    print("\n[3/6] Running LaMa inpainting …")
    torch.cuda.empty_cache()
    lama_bg_path = os.path.join(WORK_DIR, "lama_background.png")
    ok = run_lama(before_for_lama, mask_path, lama_bg_path)
    if not ok:
        raise RuntimeError("LaMa inpainting failed — check dev env installation.")

    # ── 4. Build after images (cv2 baseline + LaMa) ───────────────────────────
    print("\n[4/6] Building after images …")

    # cv2 baseline (for comparison)
    before_bgr  = cv2.cvtColor(before_rgb, cv2.COLOR_RGB2BGR)
    kernel_d    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    inpaint_m   = cv2.dilate(obj_mask, kernel_d).astype(np.uint8) * 255
    cv2_bg      = cv2.inpaint(before_bgr, inpaint_m, 7, cv2.INPAINT_TELEA)

    def paste_object(bg_bgr):
        H, W   = before_rgb.shape[:2]
        kernel_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        inner    = cv2.erode(obj_mask, kernel_e).astype(np.float32)
        alpha    = cv2.GaussianBlur(inner, (FEATHER_KSIZE, FEATHER_KSIZE), FEATHER_SIGMA)
        alpha    = np.clip(alpha, 0.0, 1.0)
        result   = bg_bgr.copy().astype(np.float32)
        src_ys, src_xs = np.where(obj_mask > 0)
        dst_xs = src_xs + SHIFT_X
        dst_ys = src_ys + SHIFT_Y
        valid  = (dst_xs >= 0) & (dst_xs < W) & (dst_ys >= 0) & (dst_ys < H)
        sy, sx = src_ys[valid], src_xs[valid]
        dy, dx = dst_ys[valid], dst_xs[valid]
        a = alpha[sy, sx, None]
        result[dy, dx] = (
            a * before_bgr[sy, sx].astype(np.float32)
            + (1 - a) * result[dy, dx]
        )
        return result.astype(np.uint8)

    cv2_after  = paste_object(cv2_bg)
    lama_bg    = cv2.imread(lama_bg_path)
    # LaMa pads internally to multiples of 8 — resize back to original dims
    H_orig, W_orig = before_rgb.shape[:2]
    if lama_bg.shape[:2] != (H_orig, W_orig):
        lama_bg = cv2.resize(lama_bg, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
    lama_after = paste_object(lama_bg)

    # Save for pipeline and display
    cv2.imwrite(os.path.join(WORK_DIR, "cv2_after.png"),  cv2_after)
    cv2.imwrite(os.path.join(WORK_DIR, "lama_after.png"), lama_after)
    cv2.imwrite(os.path.join(FINAL_DIR, "after_lama.png"),   lama_after)
    cv2.imwrite(os.path.join(FINAL_DIR, "after_cv2.png"),    cv2_after)
    cv2.imwrite(os.path.join(FINAL_DIR, "lama_background.png"), lama_bg)

    # ── 5. Gaussian update using LaMa after image ─────────────────────────────
    print("\n[5/6] Running detect_and_update with LaMa after image …")
    lama_after_path = os.path.join(WORK_DIR, "lama_after.png")
    report = detect_and_update(
        model_path       = MODEL_PATH,
        source_path      = SOURCE_PATH,
        iteration        = ITERATION,
        cam_idx          = CAM_IDX,
        output_dir       = WORK_DIR,
        after_image_path = lama_after_path,
        synthetic        = False,
        num_classes      = NUM_CLASSES,
        use_sam2         = False,
        force_obj_id     = TARGET_CLASS_ID,
    )

    # ── 6. Assemble final_example_lama/ ──────────────────────────────────────
    print("\n[6/6] Assembling final_example_lama/ …")
    updated_src = os.path.join(WORK_DIR, "05_updated_gaussian_render", "updated_render.png")
    updated_dst = os.path.join(FINAL_DIR, "updated_render.png")
    shutil.copy2(updated_src, updated_dst)

    updated_rgb = cv2.cvtColor(cv2.imread(updated_dst), cv2.COLOR_BGR2RGB)

    make_comparison_grid(
        before_rgb, cv2_after, lama_bg, lama_after, updated_rgb,
        out_path=os.path.join(FINAL_DIR, "comparison_grid.png"),
    )
    make_wipe_video(
        before_rgb, updated_rgb,
        out_path=os.path.join(FINAL_DIR, "side_by_side_video.mp4"),
    )

    # Summary
    sx, sy = report["px_shift"]
    summary = f"""# LaMa Inpainting Example: Gaussian Grouping Scene Update

Object GG class {report['target_obj_id']} ({report['n_gaussians_modified']:,} Gaussians)
shifted {abs(int(sx))} px left / {abs(int(sy))} px up.

**Background reconstruction:** LaMa (Large Mask inpainting) — photo-realistic
hole-filling compared to the cv2.inpaint TELEA baseline.

**Pipeline:** detect_and_update with LaMa-generated after image.
2-D shift recovered: {sx:+.1f} × {sy:+.1f} px.
3-D displacement: ({report['delta_world_xyz'][0]:.3f}, {report['delta_world_xyz'][1]:.3f},
{report['delta_world_xyz'][2]:.3f}) world units at depth {report['depth_at_obj']:.2f}.
"""
    with open(os.path.join(FINAL_DIR, "summary.md"), "w") as f:
        f.write(summary)

    print(f"\n{'='*60}")
    print(f"DONE — {FINAL_DIR}/")
    print(f"  Gaussians modified: {report['n_gaussians_modified']:,}")
    print(f"  2D shift recovered: {report['px_shift']}")
    print(f"  3D displacement:    {[round(v,4) for v in report['delta_world_xyz']]}")
    print(f"\nKey outputs:")
    for fn in sorted(os.listdir(FINAL_DIR)):
        print(f"  {fn}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
