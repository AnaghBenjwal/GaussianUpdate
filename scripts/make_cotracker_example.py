#!/usr/bin/env python3
"""
make_cotracker_example.py — End-to-end demo using SAM2 + CoTracker for object tracking.

This is the SAM2+CoTracker counterpart to make_final_example.py.
It uses the same simulated object move but replaces pixel-diff + template
matching with:
  • A smooth N-frame motion clip (object moves gradually from A to B)
  • SAM2 to cleanly segment the object in frame 0
  • CoTracker to track a grid of points through the motion clip
  • Tracked displacement → 3D Gaussian update

Outputs go to output/presentation/final_example_cotracker/
(kept completely separate from output/presentation/final_example/)

Run in the gaussian_grouping conda env:
    cd /home/unk/Documents/gaussian-grouping
    conda run -n gaussian_grouping python scripts/make_cotracker_example.py
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

import subprocess, tempfile
from detect_and_update import render_frame
from track_and_update import track_and_update, _run_sam2, _run_cotracker
from argparse import ArgumentParser
from scene import Scene
from scene.gaussian_model import GaussianModel
from arguments import ModelParams, PipelineParams

# ═══════════════════════════════════════════════════════════════════════════════
# Config — identical to make_final_example.py so results are directly comparable
# ═══════════════════════════════════════════════════════════════════════════════
MODEL_PATH  = "output/lerf_mask/figurines"
SOURCE_PATH = "data/lerf_mask/figurines"
ITERATION   = 30000
CAM_IDX     = 0
NUM_CLASSES = 256

# Same object + shift as make_final_example.py for a fair comparison
TARGET_CLASS_ID = 42
SHIFT_X, SHIFT_Y = -300, -100

N_FRAMES = 16          # frames in the smooth motion clip
INPAINT_RADIUS = 7
FEATHER_SIGMA  = 6.0
FEATHER_KSIZE  = 31

WORK_DIR  = "output/presentation/final_example_cotracker_work"
FINAL_DIR = "output/presentation/final_example_cotracker"


# ═══════════════════════════════════════════════════════════════════════════════
# Shared model load + render (same as make_final_example.py)
# ═══════════════════════════════════════════════════════════════════════════════

def load_model_and_render(cam_idx: int) -> Tuple[np.ndarray, np.ndarray]:
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
    ckpt = os.path.join(MODEL_PATH, "point_cloud",
                        f"iteration_{ITERATION}", "classifier.pth")
    classifier.load_state_dict(torch.load(ckpt, map_location="cuda", weights_only=False))
    classifier.eval()

    bg    = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    view  = scene.getTrainCameras()[cam_idx]
    print(f"    Rendering cam #{cam_idx}: '{view.image_name}'  "
          f"({view.image_width}×{view.image_height} px)")
    return render_frame(view, gaussians, pipeline, bg, classifier, NUM_CLASSES)


# ═══════════════════════════════════════════════════════════════════════════════
# Generate smooth N-frame motion clip
# ═══════════════════════════════════════════════════════════════════════════════

def make_motion_clip(
    before_rgb: np.ndarray,
    obj_mask:   np.ndarray,
    shift_x:    int,
    shift_y:    int,
    frames_dir: str,
    n_frames:   int,
) -> list[str]:
    """
    Generate a smooth N-frame video of the object moving from its original
    position to (orig + shift_x, orig + shift_y).

    Frame 0  = before (object at original position)
    Frame N-1 = after  (object at shifted position)

    The inpainted background (no object) is pre-computed once; at each frame t
    the object is pasted at a linearly interpolated offset.

    Returns list of saved frame paths.
    """
    os.makedirs(frames_dir, exist_ok=True)
    H, W = before_rgb.shape[:2]
    before_bgr = cv2.cvtColor(before_rgb, cv2.COLOR_RGB2BGR)

    # Pre-compute inpainted background (object hole filled)
    kernel_d  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    inpaint_m = cv2.dilate(obj_mask, kernel_d).astype(np.uint8) * 255
    background = cv2.inpaint(before_bgr, inpaint_m, INPAINT_RADIUS, cv2.INPAINT_TELEA)

    # Pre-compute feathered alpha for the object
    kernel_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    inner    = cv2.erode(obj_mask, kernel_e).astype(np.float32)
    alpha    = cv2.GaussianBlur(inner, (FEATHER_KSIZE, FEATHER_KSIZE), FEATHER_SIGMA)
    alpha    = np.clip(alpha, 0.0, 1.0)

    src_ys, src_xs = np.where(obj_mask > 0)
    frame_paths: list[str] = []

    for t in range(n_frames):
        frac = t / max(n_frames - 1, 1)
        dx   = int(round(frac * shift_x))
        dy   = int(round(frac * shift_y))

        frame = background.copy().astype(np.float32)
        dst_xs = src_xs + dx
        dst_ys = src_ys + dy
        valid  = (dst_xs >= 0) & (dst_xs < W) & (dst_ys >= 0) & (dst_ys < H)
        sy, sx = src_ys[valid], src_xs[valid]
        vy, vx = dst_ys[valid], dst_xs[valid]
        a      = alpha[sy, sx, None]
        frame[vy, vx] = (
            a * before_bgr[sy, sx].astype(np.float32)
            + (1 - a) * frame[vy, vx]
        )

        path = os.path.join(frames_dir, f"frame_{t:04d}.png")
        cv2.imwrite(path, frame.astype(np.uint8))
        frame_paths.append(path)

    return frame_paths


# ═══════════════════════════════════════════════════════════════════════════════
# Assemble final outputs (comparison grid + wipe video + summary)
# ═══════════════════════════════════════════════════════════════════════════════

def make_comparison_grid(
    before:   np.ndarray,
    after:    np.ndarray,
    sam2_vis: np.ndarray,
    updated:  np.ndarray,
    out_path: str,
) -> None:
    """2×2 grid: Before | After | SAM2 mask | Updated."""
    H, W   = before.shape[:2]
    pad    = 14
    lh     = 52
    fscale = 1.05
    thick  = 2
    gh = 2 * (H + lh) + 3 * pad
    gw = 2 * W + 3 * pad
    canvas = np.full((gh, gw, 3), 18, dtype=np.uint8)

    panels = [
        (before,   "Before  (GG render)"),
        (after,    "After target  (OpenCV)"),
        (sam2_vis, "SAM2 object mask"),
        (updated,  "Updated GG render"),
    ]
    for idx, (img, label) in enumerate(panels):
        r, c  = divmod(idx, 2)
        y0    = pad + r * (H + lh + pad)
        x0    = pad + c * (W + pad)
        canvas[y0 : y0 + H, x0 : x0 + W] = img
        bar_y = y0 + H
        canvas[bar_y : bar_y + lh, x0 : x0 + W] = (42, 42, 42)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, fscale, thick)
        tx = x0 + (W - tw) // 2
        ty = bar_y + (lh + th) // 2 - 2
        cv2.putText(canvas, label, (tx, ty),
                    cv2.FONT_HERSHEY_DUPLEX, fscale, (235, 235, 235), thick, cv2.LINE_AA)

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
) -> None:
    """Side-by-side wipe video (same as make_final_example.py)."""
    H, W  = before.shape[:2]
    VW, VH = W * 2, H + 60
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = cv2.VideoWriter(out_path, fourcc, fps, (VW, VH))

    b_bgr = cv2.cvtColor(before,  cv2.COLOR_RGB2BGR)
    u_bgr = cv2.cvtColor(updated, cv2.COLOR_RGB2BGR)

    def frame(frac: float) -> np.ndarray:
        f = np.zeros((VH, VW, 3), dtype=np.uint8)
        f[0:H, 0:W] = b_bgr
        right = b_bgr.copy()
        wp    = int(frac * W)
        if wp > 0:
            right[:, :wp] = u_bgr[:, :wp]
        if 0 < wp < W:
            cv2.line(right, (wp, 0), (wp, H), (255, 255, 255), 3)
        f[0:H, W:] = right
        f[H:, :]   = (28, 28, 28)
        cv2.putText(f, "BEFORE",
                    (W // 2 - 65, H + 40), cv2.FONT_HERSHEY_DUPLEX,
                    0.9, (195, 195, 195), 2, cv2.LINE_AA)
        cv2.putText(f, "UPDATED  (SAM2 + CoTracker)",
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
    total = 3 * hold + 2 * sweep
    print(f"    Video → {out_path}  ({total // fps}s @ {fps}fps)")


def write_summary(path: str, report: dict) -> None:
    sx, sy = report["px_shift"]
    text = f"""# SAM2 + CoTracker Example: Gaussian Grouping Scene Update

This example uses the **SAM2 + CoTracker** pipeline (as opposed to
pixel-diff + template matching in the baseline) on the **figurines** dataset.

**Object:** GG class {report['target_obj_id']} — moved {abs(sx):.0f} px left /
{abs(sy):.0f} px up over a {report['n_frames']}-frame synthetic motion clip.

**SAM2** segmented the object cleanly in frame 0 ({report['mask_source']}).
**CoTracker** tracked {report['n_tracks']} point queries through the clip;
{report['n_visible_last']} remained visible in the last frame,
recovering a displacement of {sx:+.1f} × {sy:+.1f} px.

The 2-D shift was back-projected to a 3-D world displacement of
**({report['delta_world_xyz'][0]:.3f}, {report['delta_world_xyz'][1]:.3f},
{report['delta_world_xyz'][2]:.3f})** at object depth {report['depth_at_obj']:.2f} units,
then applied in-place to {report['n_gaussians_modified']:,} Gaussians.
"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    print(f"    Summary → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    os.makedirs(WORK_DIR,  exist_ok=True)
    os.makedirs(FINAL_DIR, exist_ok=True)

    # ── 1. Render canonical before frame ─────────────────────────────────────
    print("[1/5] Rendering canonical before frame …")
    before_rgb, pred_obj_img = load_model_and_render(CAM_IDX)
    print(f"    {before_rgb.shape[1]}×{before_rgb.shape[0]} px")

    before_save = os.path.join(FINAL_DIR, "before.png")
    cv2.imwrite(before_save, cv2.cvtColor(before_rgb, cv2.COLOR_RGB2BGR))

    # ── 2. Build object mask ──────────────────────────────────────────────────
    print("\n[2/5] Building object mask …")
    obj_mask = (pred_obj_img == TARGET_CLASS_ID).astype(np.uint8)
    ys, xs   = np.where(obj_mask)
    print(f"    Class {TARGET_CLASS_ID}: {obj_mask.sum():,} px  "
          f"bbox ({xs.min()},{ys.min()})→({xs.max()},{ys.max()})")

    # Save after_target.png for the presentation (same as make_final_example.py)
    after_bgr = _build_after_single_frame(before_rgb, obj_mask, SHIFT_X, SHIFT_Y)
    cv2.imwrite(os.path.join(FINAL_DIR, "after_target.png"), after_bgr)

    # ── 3. Generate smooth N-frame motion clip ────────────────────────────────
    print(f"\n[3/5] Generating {N_FRAMES}-frame motion clip …")
    frames_dir = os.path.join(WORK_DIR, "motion_frames")
    make_motion_clip(before_rgb, obj_mask, SHIFT_X, SHIFT_Y, frames_dir, N_FRAMES)
    print(f"    Frames saved to {frames_dir}")

    # ── 3b. Run SAM2 + CoTracker NOW (before GG model reload) ────────────────
    # Running these as subprocesses while GG model is in GPU VRAM causes OOM.
    # By running here (after load_model_and_render returned + GG freed),
    # the GPU is available for CoTracker's CUDA allocations.
    print("\n[3b] Running SAM2 + CoTracker (GPU freed from step 1) …")
    tmp = tempfile.mkdtemp(prefix="gg_ct_pre_")

    frame0_path = os.path.join(frames_dir, "frame_0000.png")
    ys, xs_obj  = np.where(obj_mask)
    bbox        = (int(xs_obj.min()), int(ys.min()), int(xs_obj.max()), int(ys.max()))

    # SAM2: segment object in frame 0
    torch.cuda.empty_cache()  # release any lingering GG tensors from step 1
    sam2_mask_path = os.path.join(tmp, "sam2_mask.npy")
    sam2_ok = _run_sam2(frame0_path, bbox, sam2_mask_path, dev_env="dev")
    if sam2_ok:
        initial_mask = np.load(sam2_mask_path).astype(np.uint8)
        mask_source  = "SAM2"
    else:
        initial_mask = obj_mask.copy()
        mask_source  = "GG pred_obj_img (SAM2 fallback)"
    print(f"    SAM2 mask: {mask_source}  |  {initial_mask.sum():,} px")

    # CoTracker: track displacement through motion clip
    np.save(os.path.join(tmp, "initial_mask.npy"), initial_mask)
    tracks_mp4 = os.path.join(WORK_DIR, "cotracker_tracks.mp4")
    ct_result  = _run_cotracker(
        frames_dir, os.path.join(tmp, "initial_mask.npy"),
        os.path.join(tmp, "displacement.json"),
        viz_path=tracks_mp4, n_grid=6, dev_env="dev",
    )
    if ct_result is None:
        raise RuntimeError("CoTracker failed — see error above.")
    ct_result["mask_source"] = mask_source
    print(f"    CoTracker shift: ({ct_result['shift_x']:+.1f}, {ct_result['shift_y']:+.1f}) px  "
          f"[{ct_result['n_visible_last']}/{ct_result['n_tracks']} visible]")

    # ── 4. Run track_and_update pipeline (Gaussian update only) ──────────────
    print("\n[4/5] Updating Gaussians …")
    report = track_and_update(
        model_path        = MODEL_PATH,
        source_path       = SOURCE_PATH,
        iteration         = ITERATION,
        cam_idx           = CAM_IDX,
        output_dir        = WORK_DIR,
        frames_dir        = frames_dir,
        force_obj_id      = TARGET_CLASS_ID,
        num_classes       = NUM_CLASSES,
        dev_env           = "dev",
        n_grid            = 6,
        use_sam2          = False,            # already done above
        precomputed_shift = (ct_result["shift_x"], ct_result["shift_y"]),
        precomputed_mask  = initial_mask,
        cotracker_meta    = ct_result,
    )

    # ── 5. Assemble final_example_cotracker/ ─────────────────────────────────
    print("\n[5/5] Assembling final_example_cotracker/ …")

    # Updated render
    shutil.copy2(os.path.join(WORK_DIR, "updated_render.png"),
                 os.path.join(FINAL_DIR, "updated_render.png"))

    # SAM2 mask visualisation (replaces change_mask in the comparison grid)
    sam2_vis_path = os.path.join(WORK_DIR, "sam2_mask.png")
    shutil.copy2(sam2_vis_path, os.path.join(FINAL_DIR, "sam2_mask.png"))

    # CoTracker track video
    ct_mp4 = os.path.join(WORK_DIR, "cotracker_tracks.mp4")
    if os.path.exists(ct_mp4):
        shutil.copy2(ct_mp4, os.path.join(FINAL_DIR, "cotracker_tracks.mp4"))
        print("    Copied cotracker_tracks.mp4")

    # Report JSON
    shutil.copy2(os.path.join(WORK_DIR, "report.json"),
                 os.path.join(FINAL_DIR, "report.json"))

    # Load images for grid + video
    before_np  = cv2.cvtColor(cv2.imread(before_save),     cv2.COLOR_BGR2RGB)
    after_np   = cv2.cvtColor(cv2.imread(os.path.join(FINAL_DIR, "after_target.png")),
                               cv2.COLOR_BGR2RGB)
    sam2_np    = cv2.cvtColor(cv2.imread(sam2_vis_path),    cv2.COLOR_BGR2RGB)
    updated_np = cv2.cvtColor(cv2.imread(os.path.join(FINAL_DIR, "updated_render.png")),
                               cv2.COLOR_BGR2RGB)

    make_comparison_grid(
        before_np, after_np, sam2_np, updated_np,
        out_path=os.path.join(FINAL_DIR, "comparison_grid.png"),
    )
    make_wipe_video(
        before_np, updated_np,
        out_path=os.path.join(FINAL_DIR, "side_by_side_video.mp4"),
    )
    write_summary(os.path.join(FINAL_DIR, "summary.md"), report)

    print(f"\n{'='*60}")
    print(f"DONE — {FINAL_DIR}/")
    print(f"  Pipeline:          {report['pipeline']}")
    print(f"  Gaussians modified: {report['n_gaussians_modified']:,} (class {report['target_obj_id']})")
    print(f"  2D shift recovered: {report['px_shift']}")
    print(f"  CoTracker tracks:   {report['n_visible_last']}/{report['n_tracks']} visible")
    print(f"  3D displacement:    {[round(v, 4) for v in report['delta_world_xyz']]}")
    print(f"{'='*60}")


def _build_after_single_frame(
    before_rgb: np.ndarray, obj_mask: np.ndarray, shift_x: int, shift_y: int
) -> np.ndarray:
    """Generate the after frame (same as make_final_example.py) for display."""
    H, W = before_rgb.shape[:2]
    before_bgr = cv2.cvtColor(before_rgb, cv2.COLOR_RGB2BGR)
    kernel_d  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    inpaint_m = cv2.dilate(obj_mask, kernel_d).astype(np.uint8) * 255
    inpainted = cv2.inpaint(before_bgr, inpaint_m, INPAINT_RADIUS, cv2.INPAINT_TELEA)
    kernel_e  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    inner     = cv2.erode(obj_mask, kernel_e).astype(np.float32)
    alpha     = cv2.GaussianBlur(inner, (FEATHER_KSIZE, FEATHER_KSIZE), FEATHER_SIGMA)
    alpha     = np.clip(alpha, 0.0, 1.0)
    after     = inpainted.copy().astype(np.float32)
    src_ys, src_xs = np.where(obj_mask > 0)
    dst_xs = src_xs + shift_x
    dst_ys = src_ys + shift_y
    valid  = (dst_xs >= 0) & (dst_xs < W) & (dst_ys >= 0) & (dst_ys < H)
    sy, sx = src_ys[valid], src_xs[valid]
    dy, dx = dst_ys[valid], dst_xs[valid]
    a = alpha[sy, sx, None]
    after[dy, dx] = a * before_bgr[sy, sx].astype(np.float32) + (1 - a) * after[dy, dx]
    return after.astype(np.uint8)


if __name__ == "__main__":
    main()
