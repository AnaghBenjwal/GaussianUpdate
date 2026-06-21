#!/usr/bin/env python3
"""
track_and_update.py — Gaussian Grouping update via SAM2 + CoTracker tracking.

This is the SAM2+CoTracker version of detect_and_update.py.
Instead of pixel-diff + template matching, it uses:
  • SAM2 (image predictor)  — clean initial object mask in frame 0
  • CoTracker3              — tracks a point grid through the motion video
  • Tracked displacement    — back-projected to 3D world space

The two scripts are kept completely separate:
  detect_and_update.py  →  pixel-diff pipeline  →  output/presentation/final_example/
  track_and_update.py   →  SAM2+CoTracker pipe  →  output/presentation/final_example_cotracker/

Run in the gaussian_grouping conda env:
    conda run -n gaussian_grouping python scripts/track_and_update.py --help

SAM2 and CoTracker subprocesses are called in the `dev` conda env automatically.
"""

from __future__ import annotations
import json, math, os, subprocess, sys, tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import cv2
import torch
from PIL import Image

# ── Repo path setup ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)

from argparse import ArgumentParser
from scene import Scene
from scene.gaussian_model import GaussianModel
from arguments import ModelParams, PipelineParams
from gaussian_renderer import render as gs_render

# Re-use rendering helpers from detect_and_update
from detect_and_update import render_frame


# ═══════════════════════════════════════════════════════════════════════════════
# SAM2 via subprocess (dev env)
# ═══════════════════════════════════════════════════════════════════════════════

def _run_sam2(image_path: str, bbox: tuple[int, int, int, int],
              output_path: str, dev_env: str = "dev") -> bool:
    """
    Run sam2_segment.py in the dev conda env.
    ``bbox`` is (x1, y1, x2, y2) in pixel coords.
    Returns True on success.
    """
    cmd = [
        "conda", "run", "-n", dev_env, "--no-capture-output",
        "python3", str(ROOT / "scripts" / "sam2_segment.py"),
        "--image", image_path,
        "--bbox", str(bbox[0]), str(bbox[1]), str(bbox[2]), str(bbox[3]),
        "--output", output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    [SAM2 warning] {result.stderr.strip()[:200]}")
        return False
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# CoTracker via subprocess (dev env)
# ═══════════════════════════════════════════════════════════════════════════════

def _run_cotracker(frames_dir: str, mask_path: str, output_path: str,
                   viz_path: Optional[str] = None,
                   n_grid: int = 6, dev_env: str = "dev") -> Optional[dict]:
    """
    Run cotracker_track.py in the dev conda env.
    Returns the parsed JSON result or None on failure.
    """
    cmd = [
        "conda", "run", "-n", dev_env, "--no-capture-output",
        "python3", str(ROOT / "scripts" / "cotracker_track.py"),
        "--frames_dir", frames_dir,
        "--mask",       mask_path,
        "--output",     output_path,
        "--n_grid",     str(n_grid),
        "--checkpoint", str(ROOT / "weights/cotracker/cotracker3_scaled_online.pth"),
    ]
    if viz_path:
        cmd += ["--viz_out", viz_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    [CoTracker error] {result.stderr.strip()[:300]}")
        return None
    # Last line of stdout is the JSON
    for line in reversed(result.stdout.strip().splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Gaussian position update (same formula as detect_and_update.py)
# ═══════════════════════════════════════════════════════════════════════════════

def backproject_and_apply(view, gaussians, gaussian_indices,
                          shift_x: float, shift_y: float) -> tuple[torch.Tensor, float]:
    """
    Back-project a 2-D pixel shift to a 3-D world displacement and apply it
    to the selected Gaussians.

    Returns (delta_world, depth_at_obj).
    """
    W, H = view.image_width, view.image_height
    fx = 0.5 * W / math.tan(view.FoVx * 0.5)
    fy = 0.5 * H / math.tan(view.FoVy * 0.5)
    cx, cy = W / 2.0, H / 2.0

    with torch.no_grad():
        xyz_sel = gaussians.get_xyz[gaussian_indices]
        ones    = torch.ones(xyz_sel.shape[0], 1, device="cuda")
        cam_pts = torch.cat([xyz_sel, ones], dim=1) @ view.world_view_transform
        depth   = float(cam_pts[:, 2].mean().item())

    depth = max(depth, 0.1)

    # Approximate centroid of selected Gaussians in image space
    u_b = float(cam_pts[:, 0].mean() / cam_pts[:, 2].mean() * fx + cx)
    v_b = float(cam_pts[:, 1].mean() / cam_pts[:, 2].mean() * fy + cy)
    u_a, v_a = u_b + shift_x, v_b + shift_y

    x_b = (u_b - cx) / fx * depth
    y_b = (v_b - cy) / fy * depth
    x_a = (u_a - cx) / fx * depth
    y_a = (v_a - cy) / fy * depth

    delta_cam   = torch.tensor([x_a - x_b, y_a - y_b, 0.0],
                                device="cuda", dtype=torch.float32)
    W2V  = view.world_view_transform.T
    V2W  = torch.inverse(W2V)
    R_V2W = V2W[:3, :3]
    delta_world = R_V2W @ delta_cam

    with torch.no_grad():
        gaussians._xyz.data[gaussian_indices] += delta_world.float()

    return delta_world, depth


# ═══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def track_and_update(
    model_path:       str,
    source_path:      str,
    iteration:        int,
    cam_idx:          int,
    output_dir:       str,
    frames_dir:       str,               # directory of frame_*.png (the motion clip)
    force_obj_id:     Optional[int] = None,
    num_classes:      int = 256,
    sh_degree:        int = 3,
    dev_env:          str = "dev",
    n_grid:           int = 6,
    use_sam2:         bool = True,
    # Optional pre-computed results (avoids GPU memory conflict between GG+CoTracker)
    precomputed_shift: Optional[tuple] = None,   # (shift_x, shift_y) if already tracked
    precomputed_mask:  Optional[np.ndarray] = None,  # initial mask if already computed
    cotracker_meta:    Optional[dict] = None,    # extra CoTracker stats for report
) -> dict:
    """
    SAM2 + CoTracker variant of detect_and_update.

    Parameters
    ----------
    frames_dir  : Path to directory of frame_0000.png … (the motion video).
                  Frame 0 = before position, last frame = after position.
    force_obj_id: GG object class to update.  Use the GG pred_obj_img to find it
                  (same as in detect_and_update).
    """
    os.makedirs(output_dir, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="gg_cotracker_")

    # ── Step 1: Load GG model ─────────────────────────────────────────────────
    print("[1/7] Loading GG model and scene cameras …")
    parser = ArgumentParser()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    args_parsed = parser.parse_args([
        "--model_path",  model_path,
        "--source_path", source_path,
        "--eval",
    ])
    dataset  = lp.extract(args_parsed)
    pipeline = pp.extract(args_parsed)

    gaussians = GaussianModel(sh_degree)
    scene     = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1).cuda()
    ckpt_path  = os.path.join(model_path, "point_cloud",
                              f"iteration_{iteration}", "classifier.pth")
    classifier.load_state_dict(
        torch.load(ckpt_path, map_location="cuda", weights_only=False)
    )
    classifier.eval()

    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    view     = scene.getTrainCameras()[cam_idx]
    print(f"    Using cam #{cam_idx}: '{view.image_name}'  "
          f"({view.image_width}×{view.image_height} px)")

    # ── Step 2: Render before frame + get per-pixel object IDs ───────────────
    print("[2/7] Rendering before frame …")
    before_rgb, pred_obj_img = render_frame(
        view, gaussians, pipeline, bg_color, classifier, num_classes
    )

    before_save = os.path.join(output_dir, "before_render.png")
    cv2.imwrite(before_save, cv2.cvtColor(before_rgb, cv2.COLOR_RGB2BGR))

    # ── Step 3: Identify object Gaussians ────────────────────────────────────
    print("[3/7] Identifying object Gaussians …")
    if force_obj_id is None:
        raise ValueError("force_obj_id is required for track_and_update. "
                         "Use GG pred_obj_img to identify the target class, then pass it.")

    target_obj_id = force_obj_id
    with torch.no_grad():
        logits3d = classifier(gaussians._objects_dc.permute(2, 0, 1))
        pred_ids = logits3d.argmax(dim=0).squeeze()
        all_idx  = torch.where(pred_ids == target_obj_id)[0]
        # Depth outlier filter (same as detect_and_update)
        xyz_all = gaussians.get_xyz[all_idx]
        ones    = torch.ones(xyz_all.shape[0], 1, device="cuda")
        cam_z   = (torch.cat([xyz_all, ones], dim=1) @ view.world_view_transform)[:, 2]
        z_mean, z_std = cam_z.mean(), cam_z.std()
        valid   = (cam_z >= z_mean - 3 * z_std) & (cam_z <= z_mean + 3 * z_std) & (cam_z > 0)
    gaussian_indices = all_idx[valid]

    # Build object mask in image space
    obj_mask_img = (pred_obj_img == target_obj_id).astype(np.uint8)
    ys, xs = np.where(obj_mask_img)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

    print(f"    Class {target_obj_id}: {len(gaussian_indices):,} Gaussians "
          f"(depth-filtered)  bbox {bbox[:2]}→{bbox[2:]}")

    # ── Steps 4-5: SAM2 mask + CoTracker tracking ────────────────────────────
    # Accept pre-computed results to avoid GPU conflict (GG model + CoTracker
    # both on same GPU exhausts VRAM).  When precomputed values are passed from
    # the demo script, these steps are skipped here.
    if precomputed_shift is not None and precomputed_mask is not None:
        shift_x, shift_y = precomputed_shift
        initial_mask     = precomputed_mask
        ct_result        = cotracker_meta or {}
        method           = ct_result.get("mask_source", "precomputed")
        print(f"[4/7] Using pre-computed mask + shift ({shift_x:+.1f}, {shift_y:+.1f}) px")
    else:
        print("[4/7] Running SAM2 for initial object mask …")
        frame0_path    = sorted(Path(frames_dir).glob("frame_*.png"))[0]
        sam2_mask_path = os.path.join(tmp, "sam2_mask.npy")
        sam2_ok        = False

        if use_sam2:
            sam2_ok = _run_sam2(str(frame0_path), bbox, sam2_mask_path, dev_env)

        if sam2_ok:
            initial_mask = np.load(sam2_mask_path).astype(np.uint8)
            method       = "SAM2"
        else:
            initial_mask = obj_mask_img
            method       = "GG pred_obj_img (SAM2 unavailable)"

        print(f"    Mask source: {method}  |  {initial_mask.sum():,} pixels")
        np.save(os.path.join(tmp, "initial_mask.npy"), initial_mask)

        print("[5/7] Running CoTracker …")
        disp_path  = os.path.join(tmp, "displacement.json")
        tracks_mp4 = os.path.join(output_dir, "cotracker_tracks.mp4")
        ct_result  = _run_cotracker(frames_dir, os.path.join(tmp, "initial_mask.npy"),
                                     disp_path, viz_path=tracks_mp4,
                                     n_grid=n_grid, dev_env=dev_env)
        if ct_result is None:
            raise RuntimeError(
                "CoTracker failed — check dev env has CoTracker installed and "
                "weights/cotracker/cotracker3_scaled_online.pth exists."
            )

        shift_x = ct_result["shift_x"]
        shift_y = ct_result["shift_y"]
        print(f"    CoTracker 2-D shift: ({shift_x:+.1f}, {shift_y:+.1f}) px  "
              f"[{ct_result.get('n_visible_last','?')}/{ct_result.get('n_tracks','?')} "
              f"tracks visible]")

    # Save SAM2/initial mask visualisation regardless of source
    sam2_vis = before_rgb.copy()
    sam2_vis[initial_mask > 0] = (
        0.4 * sam2_vis[initial_mask > 0] + 0.6 * np.array([0, 200, 100])
    ).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, "sam2_mask.png"),
                cv2.cvtColor(sam2_vis, cv2.COLOR_RGB2BGR))

    # ── Step 6: Back-project 2D → 3D and update Gaussians ────────────────────
    print("[6/7] Back-projecting and applying displacement …")
    delta_world, depth = backproject_and_apply(
        view, gaussians, gaussian_indices, shift_x, shift_y
    )
    print(f"    Depth at object:   {depth:.3f} world units")
    print(f"    3-D displacement:  ({delta_world[0]:+.4f}, "
          f"{delta_world[1]:+.4f}, {delta_world[2]:+.4f})")

    # ── Step 7: Re-render and save ────────────────────────────────────────────
    print("[7/7] Re-rendering updated model …")
    updated_rgb, _ = render_frame(
        view, gaussians, pipeline, bg_color, classifier, num_classes
    )

    cv2.imwrite(os.path.join(output_dir, "updated_render.png"),
                cv2.cvtColor(updated_rgb, cv2.COLOR_RGB2BGR))

    # Save updated PLY
    ply_dir = os.path.join(output_dir, "updated_checkpoint",
                           "point_cloud", f"iteration_{iteration}")
    os.makedirs(ply_dir, exist_ok=True)
    gaussians.save_ply(os.path.join(ply_dir, "point_cloud.ply"))

    report = {
        "pipeline":           "SAM2+CoTracker",
        "target_obj_id":      target_obj_id,
        "n_gaussians_total":  int(pred_ids.shape[0]),
        "n_gaussians_modified": int(len(gaussian_indices)),
        "mask_source":        method,
        "px_shift":           [round(shift_x, 2), round(shift_y, 2)],
        "depth_at_obj":       round(depth, 3),
        "delta_world_xyz":    [round(float(v), 4) for v in delta_world.tolist()],
        "n_tracks":           ct_result.get("n_tracks", 0),
        "n_visible_last":     ct_result.get("n_visible_last", 0),
        "n_frames":           ct_result.get("n_frames", 0),
        "mask_source":        ct_result.get("mask_source", method),
    }
    with open(os.path.join(output_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    return report
