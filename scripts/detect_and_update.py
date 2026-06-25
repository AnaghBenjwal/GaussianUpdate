#!/usr/bin/env python3
"""
detect_and_update.py — Core inference script for the GG Change Detection Pipeline.

Given a trained Gaussian Grouping model and a new "after" RGB frame (at a known
camera pose from the training set), this script:

  1. Renders the existing model at the chosen camera → "before" image
  2. Detects the changed region between before/after via pixel-diff (+ optional
     SAM2 refinement via subprocess into the dev conda env)
  3. Identifies which Gaussians belong to the changed object using the GG
     per-object classifier
  4. Estimates the 2D centroid shift between before/after masks
  5. Back-projects the 2D shift to a 3D world-space displacement using the
     camera intrinsics and the depth of the object Gaussians
  6. Applies the 3D displacement to those Gaussians
  7. Renders the updated model for visual/quantitative comparison
  8. Saves: updated PLY checkpoint, comparison renders, JSON report, and a
     presentation-ready output folder

Environments
------------
This script MUST run in the ``gaussian_grouping`` conda env (Python 3.8) because
``diff-gaussian-rasterization`` is only compiled there.  SAM2 (Python ≥ 3.10) is
called optionally via subprocess into the ``dev`` env.

Typical usage — synthetic self-test (generates its own after-frame):
    cd /home/unk/Documents/gaussian-grouping
    conda run -n gaussian_grouping python scripts/detect_and_update.py \\
        --model_path  output/lerf_mask/figurines \\
        --source_path data/lerf_mask/figurines   \\
        --cam_idx 0                               \\
        --synthetic                               \\
        --synthetic_shift_x 60                   \\
        --synthetic_shift_y 30                    \\
        --output output/presentation

Usage with a real after-image (swap in tomorrow):
    conda run -n gaussian_grouping python scripts/detect_and_update.py \\
        --model_path  output/lerf_mask/figurines \\
        --source_path data/lerf_mask/figurines   \\
        --cam_idx 0                              \\
        --after  /path/to/real_after.png         \\
        --output output/presentation
"""

from __future__ import annotations
import os, sys, json, math, shutil, argparse, subprocess
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import cv2
from PIL import Image

# ── Make the repo root importable ────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from argparse import ArgumentParser
from scene import Scene
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render as gg_render
from arguments import ModelParams, PipelineParams


# ═══════════════════════════════════════════════════════════════════════════
# Depth rendering
# ═══════════════════════════════════════════════════════════════════════════

def render_depth(view, gaussians: GaussianModel, pipeline) -> np.ndarray:
    """
    Render a depth map by alpha-compositing each Gaussian's camera-space z-depth.

    Returns
    -------
    depth_map : float32 ndarray (H, W) — depth in scene world-units (same
                scale as the Gaussian xyz coordinates).
    """
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings, GaussianRasterizer,
    )
    import math

    means3D = gaussians.get_xyz                                   # (N, 3)
    ones    = torch.ones(means3D.shape[0], 1, device="cuda")
    pts_hom = torch.cat([means3D, ones], dim=1)                   # (N, 4)

    # world_view_transform stores W2V^T, so pts @ W2V^T = (W2V @ pts^T)^T
    pts_cam = pts_hom @ view.world_view_transform                 # (N, 4)
    z = pts_cam[:, 2].clamp(min=0.01)                             # (N,)

    z_min = z.min().item()
    z_max = z.max().item()
    z_norm = (z - z_min) / (z_max - z_min + 1e-8)                # (N,) in [0,1]
    depth_color = z_norm.unsqueeze(1).expand(-1, 3).contiguous()  # (N, 3)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(view.image_height),
        image_width =int(view.image_width),
        tanfovx     =math.tan(view.FoVx * 0.5),
        tanfovy     =math.tan(view.FoVy * 0.5),
        bg          =torch.zeros(3, device="cuda"),
        scale_modifier=1.0,
        viewmatrix  =view.world_view_transform,
        projmatrix  =view.full_proj_transform,
        sh_degree   =0,
        campos      =view.camera_center,
        prefiltered =False,
        debug       =pipeline.debug,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means2D = torch.zeros_like(means3D, requires_grad=False)

    rendered_rgb, _, _ = rasterizer(
        means3D       =means3D,
        means2D       =means2D,
        shs           =None,
        sh_objs       =None,
        colors_precomp=depth_color,
        opacities     =gaussians.get_opacity,
        scales        =gaussians.get_scaling,
        rotations     =gaussians.get_rotation,
        cov3D_precomp =None,
    )
    # Un-normalise: first channel holds the normalised depth
    depth_norm = rendered_rgb[0].cpu().numpy()                    # (H, W) in [0,1]
    depth_m    = depth_norm * (z_max - z_min) + z_min            # (H, W) world units
    return depth_m


# ═══════════════════════════════════════════════════════════════════════════
# Change detection
# ═══════════════════════════════════════════════════════════════════════════

def detect_changed_region(
    before_rgb: np.ndarray,
    after_rgb:  np.ndarray,
    diff_thresh: float = 25.0,
    sam2_helper: Optional[str] = None,
    output_dir:  Optional[str] = None,
    dev_env: str = "dev",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect which pixels changed between ``before_rgb`` and ``after_rgb``.

    Algorithm
    ---------
    1. Compute per-pixel max-channel absolute difference.
    2. Threshold to a binary mask, then morphologically close + dilate to fill
       small holes and smooth the boundary.
    3. Keep the single largest connected component.
    4. (Optional) Refine with SAM2 image predictor via subprocess.

    Parameters
    ----------
    before_rgb, after_rgb : uint8 (H, W, 3) arrays
    diff_thresh           : pixels with max-channel diff > this are flagged
    sam2_helper           : path to sam2_segment.py (None → skip SAM2)
    output_dir            : directory for SAM2 temp files
    dev_env               : conda env name that has SAM2 installed

    Returns
    -------
    mask_before : uint8 (H, W) — change mask aligned to the before render
    mask_after  : uint8 (H, W) — change mask aligned to the after image
                  (same as mask_before when SAM2 is skipped)
    """
    diff      = np.abs(before_rgb.astype(np.float32) - after_rgb.astype(np.float32))
    diff_gray = diff.max(axis=2)                                  # (H, W)

    binary = (diff_gray > diff_thresh).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed  = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,  kernel)
    dilated = cv2.dilate(closed, kernel, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        dilated, connectivity=8
    )
    if num_labels <= 1:
        # No foreground found — return the whole image as a fallback
        mask_before = np.ones_like(binary)
        return mask_before, mask_before

    areas   = stats[1:, cv2.CC_STAT_AREA]          # skip background label 0
    largest = int(areas.argmax()) + 1
    mask_before = (labels == largest).astype(np.uint8)

    # --- optional SAM2 refinement ---
    mask_after = mask_before
    if sam2_helper and output_dir:
        refined = _try_sam2_refinement(
            mask_before, after_rgb, sam2_helper, output_dir, dev_env
        )
        if refined is not None:
            mask_after = refined

    return mask_before, mask_after


def _try_sam2_refinement(
    rough_mask: np.ndarray,
    after_rgb:  np.ndarray,
    sam2_helper: str,
    output_dir:  str,
    dev_env:     str,
) -> Optional[np.ndarray]:
    """Call sam2_segment.py in the dev conda env; return refined mask or None."""
    tmp_after = os.path.join(output_dir, "_tmp_after_for_sam2.png")
    tmp_out   = os.path.join(output_dir, "_tmp_sam2_mask.npy")

    cv2.imwrite(tmp_after, cv2.cvtColor(after_rgb, cv2.COLOR_RGB2BGR))

    ys, xs = np.where(rough_mask > 0)
    if len(ys) == 0:
        return None
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    cmd = [
        "conda", "run", "--no-capture-output", "-n", dev_env,
        "python3", sam2_helper,
        "--image", tmp_after,
        "--bbox", str(x1), str(y1), str(x2), str(y2),
        "--output", tmp_out,
    ]
    try:
        subprocess.run(cmd, check=True, timeout=120)
        refined = np.load(tmp_out).astype(np.uint8)
        print("[info] SAM2 refinement applied.")
        return refined
    except Exception as e:
        print(f"[warn] SAM2 refinement skipped ({e}); using pixel-diff mask.")
        return None


def mask_centroid(mask: np.ndarray) -> Tuple[float, float]:
    """Return (col, row) centroid of nonzero pixels."""
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        h, w = mask.shape
        return w / 2.0, h / 2.0
    return float(xs.mean()), float(ys.mean())


def estimate_2d_shift_template(
    before_rgb:  np.ndarray,
    after_rgb:   np.ndarray,
    obj_mask:    np.ndarray,   # (H, W) uint8 — original object footprint
) -> Tuple[float, float]:
    """
    Estimate the 2-D pixel shift of an object between two images using
    normalised cross-correlation (template matching).

    The template is the object patch from ``before_rgb`` (non-object pixels
    neutralised to mid-grey).  The best match position in ``after_rgb`` gives
    the shift.

    Returns (shift_x, shift_y) in pixels.
    """
    ys, xs = np.where(obj_mask > 0)
    if len(ys) == 0:
        return 0.0, 0.0
    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()), int(ys.max())

    template = before_rgb[y1:y2+1, x1:x2+1].copy().astype(np.float32)
    crop_mask = obj_mask[y1:y2+1, x1:x2+1, None].astype(np.float32)
    # Replace non-object pixels with 128 (neutral grey) so they don't attract matches
    template = template * crop_mask + 128.0 * (1.0 - crop_mask)
    template_u8 = template.astype(np.uint8)

    # Ensure the template is not larger than the search image
    H, W = after_rgb.shape[:2]
    th, tw = template_u8.shape[:2]
    if th >= H or tw >= W:
        # Object too big to template-match; fall back to centroid of diff
        return 0.0, 0.0

    result = cv2.matchTemplate(after_rgb, template_u8, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)
    shift_x = float(max_loc[0] - x1)
    shift_y = float(max_loc[1] - y1)
    return shift_x, shift_y


# ═══════════════════════════════════════════════════════════════════════════
# Gaussian selection
# ═══════════════════════════════════════════════════════════════════════════

def get_changed_gaussian_indices(
    gaussians:       GaussianModel,
    classifier:      torch.nn.Module,
    pred_obj_img:    np.ndarray,   # (H, W) int, argmax class IDs per pixel
    changed_mask:    np.ndarray,   # (H, W) uint8 — 1 = changed region
    removal_thresh:  float = 0.3,
) -> Tuple[torch.Tensor, int]:
    """
    Find the Gaussians that belong to the object in the changed region.

    Strategy
    --------
    1. Find the dominant non-background class ID inside ``changed_mask`` on the
       per-pixel prediction image from the before render.
    2. Use the classifier's weight matrix to predict a class for every Gaussian
       directly (no rendering needed).
    3. Return indices of Gaussians whose predicted class matches the target.

    Parameters
    ----------
    pred_obj_img : (H, W) int array — per-pixel argmax class ID (0 = background)
    changed_mask : (H, W) uint8 — 1 where change was detected

    Returns
    -------
    indices      : LongTensor of Gaussian indices to move
    target_obj_id: int — the class ID identified as the changed object
    """
    region_ids = pred_obj_img[changed_mask > 0]
    if len(region_ids) == 0:
        raise ValueError("Changed-region mask is empty.")

    # Score classes by the FRACTION of their total image pixels that fall in the
    # changed region.  A class that moved will have most of its pixels inside the
    # changed area; a background class has the same density inside and outside.
    # This correctly picks the moved foreground object over a large background class
    # even when the background dominates the raw pixel count in the changed region.
    all_ids, all_counts = np.unique(pred_obj_img, return_counts=True)
    total_per_class = dict(zip(all_ids.tolist(), all_counts.tolist()))

    unique, counts_in_region = np.unique(region_ids, return_counts=True)
    scores = {}
    for cls, cnt in zip(unique.tolist(), counts_in_region.tolist()):
        if cls > 0:  # skip invalid/background (class 0)
            total = total_per_class.get(cls, 1)
            scores[cls] = cnt / total  # fraction of this class captured by the diff

    if not scores:
        raise ValueError("No non-background classes in changed region.")

    target_obj_id = max(scores, key=scores.get)
    best_frac = scores[target_obj_id]
    print(f"    Class selection scores (top 3): "
          + ", ".join(f"cls {c}={scores[c]:.2f}" for c in
                      sorted(scores, key=scores.get, reverse=True)[:3]))

    # Per-Gaussian class prediction using the classifier's linear transform.
    # _objects_dc shape: (N, 1, 16); permute(2,0,1) → (16, N, 1) which Conv2d
    # treats as (in_channels=16, H=N, W=1) producing (num_classes, N, 1).
    with torch.no_grad():
        logits3d = classifier(gaussians._objects_dc.permute(2, 0, 1))  # (C, N, 1)
        pred_ids = logits3d.argmax(dim=0).squeeze()                     # (N,)

    indices = torch.where(pred_ids == target_obj_id)[0]
    print(f"[info] Target object class: {target_obj_id}  |  "
          f"Gaussians selected: {len(indices):,} / {pred_ids.shape[0]:,}")
    return indices, target_obj_id


# ═══════════════════════════════════════════════════════════════════════════
# Back-projection: 2-D centroid shift → 3-D world displacement
# ═══════════════════════════════════════════════════════════════════════════

def backproject_displacement(
    view,
    gaussian_indices: torch.Tensor,
    gaussians:        GaussianModel,
    centroid_before:  Tuple[float, float],   # (col, row) in before-render px
    centroid_after:   Tuple[float, float],   # (col, row) in after-image px
) -> Tuple[torch.Tensor, float]:
    """
    Convert a 2-D image-space centroid shift to a 3-D world-space displacement.

    Approach
    --------
    Rather than rendering a depth map (which can be slow and has boundary
    artefacts), we compute the object depth directly as the mean camera-space
    z-coordinate of the selected Gaussians.  This is exact and fast.

    The pinhole back-projection formula is then:
        X_cam = (u - cx) / fx * z
        Y_cam = (v - cy) / fy * z
        Z_cam = z  (unchanged — we assume planar motion at constant depth)

    The camera-space delta is rotated to world space via the view-to-world
    rotation matrix derived from the stored world-view transform.

    Parameters
    ----------
    centroid_before : (col, row) in the before render's pixel coordinates
    centroid_after  : (col, row) in the after image's pixel coordinates

    Returns
    -------
    delta_world  : float32 tensor (3,) — displacement vector in world space
    depth_at_obj : float — depth used for the back-projection
    """
    W, H = view.image_width, view.image_height
    fx = 0.5 * W / math.tan(view.FoVx * 0.5)
    fy = 0.5 * H / math.tan(view.FoVy * 0.5)
    cx, cy = W / 2.0, H / 2.0

    # Depth: mean camera-space z of the selected Gaussians
    with torch.no_grad():
        xyz_sel = gaussians.get_xyz[gaussian_indices]           # (M, 3)
        ones    = torch.ones(xyz_sel.shape[0], 1, device="cuda")
        hom     = torch.cat([xyz_sel, ones], dim=1)             # (M, 4)
        cam_pts = hom @ view.world_view_transform               # (M, 4)
        depth_at_obj = float(cam_pts[:, 2].mean().item())
    depth_at_obj = max(depth_at_obj, 0.1)

    u_b, v_b = centroid_before
    u_a, v_a = centroid_after

    x_b = (u_b - cx) / fx * depth_at_obj
    y_b = (v_b - cy) / fy * depth_at_obj
    x_a = (u_a - cx) / fx * depth_at_obj
    y_a = (v_a - cy) / fy * depth_at_obj

    delta_cam = torch.tensor(
        [x_a - x_b, y_a - y_b, 0.0],
        device="cuda", dtype=torch.float32,
    )

    # Rotate camera-space delta → world space
    # world_view_transform = W2V^T  ⟹  W2V = wvt.T  ⟹  V2W = W2V^{-1}
    W2V   = view.world_view_transform.T
    V2W   = torch.inverse(W2V)
    R_V2W = V2W[:3, :3]

    delta_world = R_V2W @ delta_cam       # (3,)
    return delta_world, depth_at_obj


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic "after" frame generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_synthetic_after(
    before_rgb:    np.ndarray,   # (H, W, 3) uint8
    pred_obj_img:  np.ndarray,   # (H, W) int, argmax class IDs
    obj_id:        int,
    shift_x:       int,
    shift_y:       int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Produce a synthetic "after" frame by rigidly shifting ``obj_id``'s pixels.

    Vacated pixels are filled with the median colour of the background.
    The object mask for the chosen object is also returned.

    Returns
    -------
    after_rgb : uint8 (H, W, 3)
    obj_mask  : uint8 (H, W) — 1 where the object was in the before frame
    """
    H, W = before_rgb.shape[:2]
    obj_mask = (pred_obj_img == obj_id).astype(np.uint8)

    bg_pixels  = before_rgb[obj_mask == 0]
    fill_color = np.median(bg_pixels, axis=0).astype(np.uint8) if len(bg_pixels) else np.zeros(3, dtype=np.uint8)

    after = before_rgb.copy()
    after[obj_mask > 0] = fill_color                            # erase original

    src_ys, src_xs = np.where(obj_mask > 0)
    dst_xs = src_xs + shift_x
    dst_ys = src_ys + shift_y
    valid  = (dst_xs >= 0) & (dst_xs < W) & (dst_ys >= 0) & (dst_ys < H)
    after[dst_ys[valid], dst_xs[valid]] = before_rgb[src_ys[valid], src_xs[valid]]

    return after, obj_mask


# ═══════════════════════════════════════════════════════════════════════════
# Visualisation helpers
# ═══════════════════════════════════════════════════════════════════════════

def tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """(3, H, W) float [0,1] tensor → (H, W, 3) uint8 array."""
    return (t.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)


def overlay_mask(rgb: np.ndarray, mask: np.ndarray,
                 color=(255, 80, 0), alpha=0.50) -> np.ndarray:
    out = rgb.copy()
    tint = np.array(color, dtype=np.float32)
    out[mask > 0] = (
        out[mask > 0].astype(np.float32) * (1 - alpha) + tint * alpha
    ).astype(np.uint8)
    return out


def draw_arrow(img: np.ndarray, pt_from, pt_to,
               color=(0, 220, 0), thickness=3) -> np.ndarray:
    out = img.copy()
    cv2.arrowedLine(
        out,
        (int(pt_from[0]), int(pt_from[1])),
        (int(pt_to[0]),   int(pt_to[1])),
        color, thickness, tipLength=0.25,
    )
    return out


def make_side_by_side(imgs, labels=None, gap=8) -> np.ndarray:
    """Horizontally concatenate images with an optional text label bar."""
    h = max(im.shape[0] for im in imgs)
    padded = [
        cv2.copyMakeBorder(im, 0, h - im.shape[0], 0, 0,
                           cv2.BORDER_CONSTANT, value=(30, 30, 30))
        for im in imgs
    ]
    row = np.concatenate(padded, axis=1)
    if labels:
        x = 0
        for label, im in zip(labels, padded):
            cv2.putText(row, label, (x + 8, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
            x += im.shape[1]
    return row


# ═══════════════════════════════════════════════════════════════════════════
# Rendering helpers
# ═══════════════════════════════════════════════════════════════════════════

def render_frame(view, gaussians, pipeline, background, classifier, num_classes):
    """
    Render one camera view.

    Returns
    -------
    rgb_uint8   : uint8 (H, W, 3)
    pred_obj_id : int32 (H, W) — per-pixel argmax class ID
    """
    with torch.no_grad():
        results   = gg_render(view, gaussians, pipeline, background)
        render_t  = results["render"]                           # (3, H, W) float
        render_obj = results["render_object"]                   # (16, H, W) float

        logits    = classifier(render_obj)                      # (num_classes, H, W)
        pred_obj  = logits.argmax(dim=0).cpu().numpy().astype(np.int32)  # (H, W)

    rgb_uint8 = tensor_to_uint8(render_t)
    return rgb_uint8, pred_obj


# ═══════════════════════════════════════════════════════════════════════════
# Presentation folder builder
# ═══════════════════════════════════════════════════════════════════════════

def _save(path: str, img_rgb: np.ndarray):
    """Save uint8 RGB numpy array as PNG."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))


def build_presentation_folders(
    pdir: str,
    before_rgb:     np.ndarray,
    after_rgb:      np.ndarray,
    updated_rgb:    np.ndarray,
    mask_before:    np.ndarray,
    mask_after:     np.ndarray,
    centroid_b:     Tuple[float, float],
    centroid_a:     Tuple[float, float],
    delta_world:    torch.Tensor,
    source_path:    str,
    diff_gray:      Optional[np.ndarray] = None,
):
    """Write all output/presentation/ sub-folders."""
    os.makedirs(pdir, exist_ok=True)

    # 00 — sample training images
    d00 = os.path.join(pdir, "00_input_static_scene")
    os.makedirs(d00, exist_ok=True)
    img_dir = os.path.join(source_path, "images")
    if os.path.isdir(img_dir):
        imgs = sorted(os.listdir(img_dir))
        for fn in imgs[:6]:
            src = os.path.join(img_dir, fn)
            dst = os.path.join(d00, fn)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

    # 01 — before render
    d01 = os.path.join(pdir, "01_gaussian_map_render")
    _save(os.path.join(d01, "before_render.png"), before_rgb)

    # 02 — synthetic after + diff
    d02 = os.path.join(pdir, "02_synthetic_after_input")
    _save(os.path.join(d02, "after_frame.png"),  after_rgb)
    _save(os.path.join(d02, "before_frame.png"), before_rgb)
    # diff overlay
    diff = np.abs(before_rgb.astype(np.float32) - after_rgb.astype(np.float32)).max(axis=2)
    diff_vis = np.clip(diff * 3, 0, 255).astype(np.uint8)
    diff_vis_rgb = cv2.applyColorMap(diff_vis, cv2.COLORMAP_HOT)
    diff_vis_rgb = cv2.cvtColor(diff_vis_rgb, cv2.COLOR_BGR2RGB)
    _save(os.path.join(d02, "diff_heatmap.png"), diff_vis_rgb)
    _save(os.path.join(d02, "side_by_side.png"),
          make_side_by_side([before_rgb, after_rgb, diff_vis_rgb],
                            ["BEFORE (render)", "AFTER (synthetic)", "DIFF heatmap"]))

    # 03 — change detection mask
    d03 = os.path.join(pdir, "03_change_detection")
    _save(os.path.join(d03, "mask_overlay_before.png"),
          overlay_mask(before_rgb, mask_before, color=(255, 80,   0)))
    _save(os.path.join(d03, "mask_overlay_after.png"),
          overlay_mask(after_rgb,  mask_after,  color=(0,   160, 255)))

    # 04 — displacement estimate
    d04 = os.path.join(pdir, "04_displacement_estimate")
    arrow_img = draw_arrow(
        overlay_mask(before_rgb, mask_before),
        centroid_b, centroid_a,
        color=(0, 230, 0), thickness=4,
    )
    cv2.circle(arrow_img, (int(centroid_b[0]), int(centroid_b[1])), 8, (0, 0, 255), -1)
    cv2.circle(arrow_img, (int(centroid_a[0]), int(centroid_a[1])), 8, (0, 255, 0), -1)
    d_np = delta_world.cpu().numpy()
    info = (f"2D shift: ({centroid_a[0]-centroid_b[0]:.1f}, "
            f"{centroid_a[1]-centroid_b[1]:.1f}) px  |  "
            f"3D delta: ({d_np[0]:.3f}, {d_np[1]:.3f}, {d_np[2]:.3f})")
    cv2.putText(arrow_img, info, (10, arrow_img.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
    _save(os.path.join(d04, "displacement_arrow.png"), arrow_img)

    # 05 — updated model render
    d05 = os.path.join(pdir, "05_updated_gaussian_render")
    _save(os.path.join(d05, "updated_render.png"), updated_rgb)

    # 06 — comparison
    d06 = os.path.join(pdir, "06_comparison")
    comparison = make_side_by_side(
        [before_rgb, after_rgb, updated_rgb],
        ["BEFORE (original render)", "AFTER (target)", "UPDATED (new render)"],
    )
    _save(os.path.join(d06, "comparison.png"), comparison)
    print(f"[info] Presentation folder written to: {pdir}")


# ═══════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════

def detect_and_update(
    model_path:       str,
    source_path:      str,
    iteration:        int,
    cam_idx:          int,
    output_dir:       str,
    after_image_path: Optional[str]  = None,
    synthetic:        bool           = False,
    synthetic_obj_id: Optional[int]  = None,
    synthetic_shift_x: int           = 60,
    synthetic_shift_y: int           = 30,
    diff_threshold:   float          = 25.0,
    num_classes:      int            = 256,
    sh_degree:        int            = 3,
    dev_env:          str            = "dev",
    use_sam2:         bool           = True,
    force_obj_id:     Optional[int]  = None,
) -> dict:
    """
    Full change-detection and Gaussian update pipeline.

    Parameters
    ----------
    model_path        : Path to the trained GG model directory (contains
                        ``point_cloud/iteration_*/point_cloud.ply`` and
                        ``cfg_args``).
    source_path       : Path to the original training data directory (contains
                        ``sparse/`` for camera poses, ``images/`` for training
                        images).
    iteration         : Checkpoint iteration to load (e.g. 30000).
    cam_idx           : Index into the training camera list to use as the
                        reference viewpoint.
    output_dir        : Root of the presentation output tree.
    after_image_path  : Path to the "after" RGB PNG/JPG.  Ignored when
                        ``synthetic=True``.
    synthetic         : If True, generate a synthetic after-frame by shifting a
                        known object's pixels and use it as the after-image.
    synthetic_obj_id  : Class ID of the object to shift (None → auto-select the
                        largest non-background object).
    synthetic_shift_x : Pixel shift in x for the synthetic after-frame.
    synthetic_shift_y : Pixel shift in y for the synthetic after-frame.
    diff_threshold    : Per-pixel max-channel difference threshold for the
                        pixel-diff change detector.
    num_classes       : Number of object classes in the GG classifier.
    sh_degree         : Spherical-harmonics degree of the GG model.
    dev_env           : Name of the conda environment with SAM2 installed.
    use_sam2          : If True, attempt SAM2 subprocess refinement.

    Returns
    -------
    report : dict with keys:
        gaussian_indices_modified, n_gaussians_modified,
        delta_world_xyz, depth_at_obj,
        centroid_before_px, centroid_after_px,
        target_obj_id, cam_idx
    """
    os.chdir(ROOT)                 # ensure relative paths resolve from repo root
    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: Load model and scene ────────────────────────────────────────
    print("[1/8] Loading GG model and scene cameras …")
    parser = ArgumentParser()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    # Do NOT pass --eval: with eval=True the scene splits cameras into
    # train/test (every 8th → test), so cam_idx=0 would skip frame_00000.
    # Without --eval all 163 cameras are in getTrainCameras(), so cam_idx=0
    # correctly picks frame_00000 (the canonical before/after view).
    args = parser.parse_args([
        "--model_path",  model_path,
        "--source_path", source_path,
    ])
    dataset  = lp.extract(args)
    pipeline = pp.extract(args)

    gaussians = GaussianModel(sh_degree)
    scene     = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1).cuda()
    ckpt_path  = os.path.join(model_path, "point_cloud",
                              f"iteration_{iteration}", "classifier.pth")
    classifier.load_state_dict(
        torch.load(ckpt_path, map_location="cuda", weights_only=False)
    )
    classifier.eval()

    bg_color   = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    train_cams = scene.getTrainCameras()
    if cam_idx >= len(train_cams):
        raise IndexError(
            f"cam_idx={cam_idx} out of range; only {len(train_cams)} training cameras."
        )
    view = train_cams[cam_idx]
    print(f"    Using cam #{cam_idx}: '{view.image_name}'  "
          f"({view.image_width}×{view.image_height} px)")

    # ── Step 2: Render the "before" image ──────────────────────────────────
    print("[2/8] Rendering before image …")
    before_rgb, pred_obj_img = render_frame(
        view, gaussians, pipeline, bg_color, classifier, num_classes
    )

    # ── Step 3: Load or generate the "after" image ─────────────────────────
    print("[3/8] Preparing after image …")
    if synthetic:
        # Auto-select the largest FOREGROUND object (neither background class 0
        # nor any class covering > 15 % of the image — those are typically
        # backgrounds or large table surfaces).
        if synthetic_obj_id is None:
            H_img, W_img = pred_obj_img.shape
            total_px = H_img * W_img
            unique, counts = np.unique(pred_obj_img, return_counts=True)
            fracs = counts / total_px
            # Foreground: non-zero class, between 0.5 % and 15 % of image
            fg = (unique > 0) & (fracs > 0.005) & (fracs < 0.15)
            if not fg.any():
                # Relax upper bound: take any non-background object
                fg = unique > 0
            if not fg.any():
                raise RuntimeError("No non-background objects found in before render.")
            synthetic_obj_id = int(unique[fg][counts[fg].argmax()])
            print(f"    Auto-selected synthetic object class: {synthetic_obj_id}  "
                  f"({counts[unique == synthetic_obj_id][0] / total_px * 100:.1f}% of image)")
        else:
            print(f"    Using synthetic object class: {synthetic_obj_id}")

        after_rgb, obj_mask = generate_synthetic_after(
            before_rgb, pred_obj_img,
            obj_id=synthetic_obj_id,
            shift_x=synthetic_shift_x,
            shift_y=synthetic_shift_y,
        )
        print(f"    Synthetic shift: ({synthetic_shift_x}, {synthetic_shift_y}) px  |  "
              f"Object pixels: {obj_mask.sum():,}")
    else:
        if after_image_path is None or not os.path.isfile(after_image_path):
            raise FileNotFoundError(f"After image not found: {after_image_path}")
        after_pil = Image.open(after_image_path).convert("RGB")
        after_pil = after_pil.resize(
            (view.image_width, view.image_height), Image.BILINEAR
        )
        after_rgb = np.array(after_pil)

    # ── Step 4: Detect the changed region ──────────────────────────────────
    print("[4/8] Detecting changed region …")
    sam2_helper = str(ROOT / "scripts" / "sam2_segment.py")
    mask_before, mask_after = detect_changed_region(
        before_rgb, after_rgb,
        diff_thresh =diff_threshold,
        sam2_helper =sam2_helper if use_sam2 else None,
        output_dir  =output_dir,
        dev_env     =dev_env,
    )
    print(f"    Changed pixels: {mask_before.sum():,}")

    # ── Step 5: Find the Gaussians that belong to the changed object ────────
    print("[5/8] Identifying changed Gaussians …")
    if force_obj_id is not None:
        # When the caller already knows which object moved (e.g. from a shared
        # segmentation step), bypass the heuristic class-selection entirely.
        target_obj_id = force_obj_id
        with torch.no_grad():
            logits3d = classifier(gaussians._objects_dc.permute(2, 0, 1))
            pred_ids = logits3d.argmax(dim=0).squeeze()
        all_idx = torch.where(pred_ids == target_obj_id)[0]

        # Filter out depth outliers: keep only Gaussians within ±3σ of the mean
        # camera-space z.  GG classes often contain scattered "noise" Gaussians
        # far outside the visible object; moving them causes scene-wide artifacts.
        with torch.no_grad():
            xyz_all = gaussians.get_xyz[all_idx]
            ones    = torch.ones(xyz_all.shape[0], 1, device="cuda")
            cam_z   = (torch.cat([xyz_all, ones], dim=1) @ view.world_view_transform)[:, 2]
            z_mean  = cam_z.mean(); z_std = cam_z.std()
            z_lo    = z_mean - 3.0 * z_std
            z_hi    = z_mean + 3.0 * z_std
            valid   = (cam_z >= z_lo) & (cam_z <= z_hi) & (cam_z > 0.0)
        gaussian_indices = all_idx[valid]
        print(f"[info] Using forced object class: {target_obj_id}  |  "
              f"Gaussians selected: {len(gaussian_indices):,} / {len(all_idx):,} "
              f"(depth filter ±3σ: z∈[{float(z_lo):.2f}, {float(z_hi):.2f}])")
    else:
        gaussian_indices, target_obj_id = get_changed_gaussian_indices(
            gaussians, classifier, pred_obj_img, mask_before
        )
    if len(gaussian_indices) == 0:
        raise RuntimeError("No Gaussians matched the changed object — check diff_threshold or force_obj_id.")

    # ── Step 6: Compute 2-D centroid shift ─────────────────────────────────
    # The diff mask alone cannot give us centroids: it is the UNION of the
    # vacated and arrival areas, whose centroids are on opposite sides of the
    # object, not at the before/after object centres.
    #
    # Strategy (in priority order):
    #   A. SAM2 mask on after-image  → centroid of SAM2 mask (most accurate)
    #   B. Template matching          → match before-object crop in after-image
    #   C. Diff midpoint heuristic    → 2×(centroid(diff) − centroid(before_obj))
    print("[6/8] Computing displacement …")
    mask_obj_before = (pred_obj_img == target_obj_id).astype(np.uint8)
    centroid_b = mask_centroid(mask_obj_before)   # centroid of object in before render

    # Strategy A: SAM2 gives a mask on the after-image
    sam2_gave_after_mask = (mask_after is not mask_before)  # True if SAM2 ran
    if sam2_gave_after_mask and mask_after.sum() > 100:
        centroid_a = mask_centroid(mask_after)
        print(f"    Centroid method: SAM2 mask on after-image")
    else:
        # Strategy B: template matching
        print("    Centroid method: template matching (SAM2 not available)")
        shift_x, shift_y = estimate_2d_shift_template(
            before_rgb, after_rgb, mask_obj_before
        )
        centroid_a = (centroid_b[0] + shift_x, centroid_b[1] + shift_y)

    px_shift = (centroid_a[0] - centroid_b[0], centroid_a[1] - centroid_b[1])
    print(f"    2-D centroid shift: ({px_shift[0]:+.1f}, {px_shift[1]:+.1f}) px")

    # ── Step 7: Back-project 2-D shift → 3-D displacement ──────────────────
    delta_world, depth_at_obj = backproject_displacement(
        view, gaussian_indices, gaussians, centroid_b, centroid_a
    )
    d = delta_world.cpu().numpy()
    print(f"    Depth at object: {depth_at_obj:.3f} units")
    print(f"    3-D world displacement: ({d[0]:+.4f}, {d[1]:+.4f}, {d[2]:+.4f})")

    # ── Step 8: Apply displacement and render the updated model ─────────────
    print("[7/8] Applying displacement to Gaussians and re-rendering …")
    with torch.no_grad():
        gaussians._xyz.data[gaussian_indices] += delta_world.float()

    updated_rgb, _ = render_frame(
        view, gaussians, pipeline, bg_color, classifier, num_classes
    )

    # ── Save checkpoint ─────────────────────────────────────────────────────
    print("[8/8] Saving outputs …")
    updated_ply_dir = os.path.join(output_dir, "updated_checkpoint",
                                   f"iteration_{iteration}")
    os.makedirs(updated_ply_dir, exist_ok=True)
    gaussians.save_ply(os.path.join(updated_ply_dir, "point_cloud.ply"))
    shutil.copy2(ckpt_path, os.path.join(updated_ply_dir, "classifier.pth"))
    print(f"    Updated PLY → {updated_ply_dir}/point_cloud.ply")

    # ── JSON report ─────────────────────────────────────────────────────────
    report = {
        "cam_idx":                   cam_idx,
        "camera_name":               view.image_name,
        "target_obj_id":             target_obj_id,
        "n_gaussians_total":         int(gaussians._xyz.shape[0]),
        "n_gaussians_modified":      len(gaussian_indices),
        "gaussian_indices_modified": gaussian_indices.cpu().tolist(),
        "depth_at_obj":              depth_at_obj,
        "centroid_before_px":        list(centroid_b),
        "centroid_after_px":         list(centroid_a),
        "px_shift":                  list(px_shift),
        "delta_world_xyz":           d.tolist(),
        "synthetic":                 synthetic,
        "synthetic_shift_x":         synthetic_shift_x if synthetic else None,
        "synthetic_shift_y":         synthetic_shift_y if synthetic else None,
    }
    report_path = os.path.join(output_dir, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"    Report → {report_path}")

    # ── Presentation folders ─────────────────────────────────────────────────
    build_presentation_folders(
        pdir         =output_dir,
        before_rgb   =before_rgb,
        after_rgb    =after_rgb,
        updated_rgb  =updated_rgb,
        mask_before  =mask_obj_before,
        mask_after   =mask_after if sam2_gave_after_mask else mask_before,
        centroid_b   =centroid_b,
        centroid_a   =centroid_a,
        delta_world  =delta_world,
        source_path  =source_path,
    )

    return report


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="GG Change-Detection and Gaussian Update Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_path",   required=True,
                   help="Path to GG checkpoint dir (contains point_cloud/ and cfg_args)")
    p.add_argument("--source_path",  required=True,
                   help="Path to training data dir (contains sparse/ and images/)")
    p.add_argument("--iteration",    type=int, default=30000,
                   help="Checkpoint iteration to load")
    p.add_argument("--cam_idx",      type=int, default=0,
                   help="Index into the training camera list")
    p.add_argument("--after",        default=None,
                   help="Path to after-frame PNG/JPG (ignored if --synthetic)")
    p.add_argument("--output",       default="output/presentation",
                   help="Output directory for all results")

    # Synthetic test options
    p.add_argument("--synthetic",    action="store_true",
                   help="Generate a synthetic after-frame instead of loading one")
    p.add_argument("--synthetic_obj_id", type=int, default=None,
                   help="Object class ID to shift (None = auto-select largest)")
    p.add_argument("--synthetic_shift_x", type=int, default=60)
    p.add_argument("--synthetic_shift_y", type=int, default=30)

    # Detection options
    p.add_argument("--diff_threshold", type=float, default=25.0,
                   help="Pixel-diff threshold for change detection")
    p.add_argument("--no_sam2",     action="store_true",
                   help="Skip SAM2 refinement (use pixel diff mask only)")
    p.add_argument("--dev_env",     default="dev",
                   help="Conda env name that has SAM2 installed")

    # Model options
    p.add_argument("--num_classes", type=int, default=256)
    p.add_argument("--sh_degree",   type=int, default=3)
    p.add_argument("--force_obj_id", type=int, default=None,
                   help="Skip change-detection and use this class ID directly")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    report = detect_and_update(
        model_path       =args.model_path,
        source_path      =args.source_path,
        iteration        =args.iteration,
        cam_idx          =args.cam_idx,
        output_dir       =args.output,
        after_image_path =args.after,
        synthetic        =args.synthetic,
        synthetic_obj_id =args.synthetic_obj_id,
        synthetic_shift_x=args.synthetic_shift_x,
        synthetic_shift_y=args.synthetic_shift_y,
        diff_threshold   =args.diff_threshold,
        num_classes      =args.num_classes,
        sh_degree        =args.sh_degree,
        dev_env          =args.dev_env,
        use_sam2         =not args.no_sam2,
        force_obj_id     =args.force_obj_id,
    )
    print("\n=== DONE ===")
    print(f"Modified {report['n_gaussians_modified']:,} Gaussians "
          f"(class {report['target_obj_id']})")
    print(f"3-D displacement: {report['delta_world_xyz']}")
