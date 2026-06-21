#!/usr/bin/env python3
"""
run_batch_experiment.py — Batch experiment: 5 objects × 5 displacements × 2 methods.

Methods compared per experiment:
  cv2.inpaint  — OpenCV TELEA texture propagation (baseline)
  LaMa         — Large Mask inpainting (photo-realistic)

The Gaussian update step is IDENTICAL for both methods (same known shift → same
3-D displacement). The comparison shows inpainting quality, not tracking accuracy.

Outputs go to output/experiments/<obj_id>_<shift_name>/
  comparison.png          — 2-row × 3-col grid:
                              cv2 row:  Before | cv2 after  | GG updated render
                              LaMa row: Before | LaMa after | GG updated render
  cv2_after.png
  lama_background.png
  lama_after.png
  updated_render.png      — shared (same GG update for both methods)
  report.json
overview.png              — 5×5 thumbnail grid (one cell per experiment)

Run in gaussian_grouping conda env:
    conda run -n gaussian_grouping python scripts/run_batch_experiment.py
"""

from __future__ import annotations
import json, math, os, subprocess, sys, tempfile, time
from pathlib import Path

import numpy as np
import cv2
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)

from argparse import ArgumentParser
from scene import Scene
from scene.gaussian_model import GaussianModel
from arguments import ModelParams, PipelineParams
from detect_and_update import render_frame

# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_PATH  = "output/lerf_mask/figurines"
SOURCE_PATH = "data/lerf_mask/figurines"
ITERATION   = 30000
CAM_IDX     = 0
NUM_CLASSES = 256

# 5 specific objects (surveyed: class 32 is wide background, skip it)
TARGET_OBJECTS = [50, 70, 42, 5, 25]

# 5 displacement patterns: (name, dx, dy) in pixels
SHIFTS = [
    ("left_sm",   -120,    0),   # small left
    ("left_lg",   -300,    0),   # large left
    ("up",           0, -180),   # straight up
    ("right",     +160,    0),   # right
    ("diagonal",  -180, -130),   # upper-left diagonal
]

FEATHER_SIGMA  = 6.0
FEATHER_KSIZE  = 31
CV2_DILATE     = 9
LAMA_DILATE    = 15
MARGIN         = 20   # px margin from image edge

OUTPUT_DIR = "output/experiments"


# ═══════════════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_model():
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
    view = scene.getTrainCameras()[CAM_IDX]
    return gaussians, scene, pipeline, bg, clf, view


# ═══════════════════════════════════════════════════════════════════════════════
# Per-object helpers
# ═══════════════════════════════════════════════════════════════════════════════

def get_gaussian_indices(gaussians, clf, view, obj_id: int):
    """Depth-filtered Gaussian indices for obj_id (same filter as detect_and_update)."""
    with torch.no_grad():
        logits3d = clf(gaussians._objects_dc.permute(2, 0, 1))
        pred_ids = logits3d.argmax(dim=0).squeeze()
        all_idx  = torch.where(pred_ids == obj_id)[0]
        if len(all_idx) == 0:
            return all_idx
        xyz_all = gaussians.get_xyz[all_idx]
        ones    = torch.ones(xyz_all.shape[0], 1, device="cuda")
        cam_z   = (torch.cat([xyz_all, ones], dim=1) @ view.world_view_transform)[:, 2]
        z_mean, z_std = cam_z.mean(), cam_z.std()
        valid   = (cam_z >= z_mean - 3 * z_std) & (cam_z <= z_mean + 3 * z_std) & (cam_z > 0)
    return all_idx[valid]


def clamp_shift(bbox, dx: int, dy: int, W: int, H: int) -> tuple[int, int]:
    """Clamp shift so the object bbox stays MARGIN px inside image bounds."""
    x1, y1, x2, y2 = bbox
    dx = int(np.clip(dx, MARGIN - x1, (W - MARGIN) - x2))
    dy = int(np.clip(dy, MARGIN - y1, (H - MARGIN) - y2))
    return dx, dy


def feathered_paste(bg_bgr: np.ndarray, src_bgr: np.ndarray,
                    mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Paste src pixels (mask region) onto bg at (x+dx, y+dy) with feathered alpha."""
    H, W = bg_bgr.shape[:2]
    kernel_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    inner    = cv2.erode(mask, kernel_e).astype(np.float32)
    alpha    = cv2.GaussianBlur(inner, (FEATHER_KSIZE, FEATHER_KSIZE), FEATHER_SIGMA)
    alpha    = np.clip(alpha, 0.0, 1.0)

    result   = bg_bgr.copy().astype(np.float32)
    src_ys, src_xs = np.where(mask > 0)
    dst_xs   = src_xs + dx
    dst_ys   = src_ys + dy
    valid    = (dst_xs >= 0) & (dst_xs < W) & (dst_ys >= 0) & (dst_ys < H)
    sy, sx   = src_ys[valid], src_xs[valid]
    dy2, dx2 = dst_ys[valid], dst_xs[valid]
    a        = alpha[sy, sx, None]
    result[dy2, dx2] = (
        a * src_bgr[sy, sx].astype(np.float32)
        + (1 - a) * result[dy2, dx2]
    )
    return result.astype(np.uint8)


def make_cv2_after(before_bgr: np.ndarray, mask: np.ndarray,
                   dx: int, dy: int) -> np.ndarray:
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CV2_DILATE, CV2_DILATE))
    inpaint = cv2.dilate(mask, kernel).astype(np.uint8) * 255
    bg      = cv2.inpaint(before_bgr, inpaint, 7, cv2.INPAINT_TELEA)
    return feathered_paste(bg, before_bgr, mask, dx, dy)


def run_lama_subprocess(before_bgr_path: str, mask_npy_path: str,
                        out_path: str) -> bool:
    cmd = [
        "conda", "run", "-n", "dev",
        "python3", str(ROOT / "scripts" / "lama_inpaint.py"),
        "--image",  before_bgr_path,
        "--mask",   mask_npy_path,
        "--output", out_path,
        "--dilate", str(LAMA_DILATE),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"    [LaMa STDOUT] {r.stdout.strip()[:300]}")
        print(f"    [LaMa STDERR] {r.stderr.strip()[:800]}")
        return False
    return True


def make_lama_after(before_bgr: np.ndarray, mask: np.ndarray,
                    lama_bg_path: str, dx: int, dy: int,
                    H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
    lama_bg = cv2.imread(lama_bg_path)
    if lama_bg.shape[:2] != (H, W):
        lama_bg = cv2.resize(lama_bg, (W, H), interpolation=cv2.INTER_LINEAR)
    after = feathered_paste(lama_bg, before_bgr, mask, dx, dy)
    return after, lama_bg


# ═══════════════════════════════════════════════════════════════════════════════
# Gaussian update
# ═══════════════════════════════════════════════════════════════════════════════

def backproject_delta(view, gaussians, indices, dx: float, dy: float):
    W, H  = view.image_width, view.image_height
    fx    = 0.5 * W / math.tan(view.FoVx * 0.5)
    fy    = 0.5 * H / math.tan(view.FoVy * 0.5)
    cx, cy = W / 2.0, H / 2.0

    with torch.no_grad():
        xyz_sel = gaussians.get_xyz[indices]
        ones    = torch.ones(xyz_sel.shape[0], 1, device="cuda")
        cam_pts = torch.cat([xyz_sel, ones], dim=1) @ view.world_view_transform
        depth   = float(cam_pts[:, 2].mean().item())

    depth = max(depth, 0.1)
    ub = float(cam_pts[:, 0].mean() / cam_pts[:, 2].mean() * fx + cx)
    vb = float(cam_pts[:, 1].mean() / cam_pts[:, 2].mean() * fy + cy)
    ua, va = ub + dx, vb + dy

    x_b  = (ub - cx) / fx * depth;  x_a = (ua - cx) / fx * depth
    y_b  = (vb - cy) / fy * depth;  y_a = (va - cy) / fy * depth
    delta_cam   = torch.tensor([x_a - x_b, y_a - y_b, 0.0],
                               device="cuda", dtype=torch.float32)
    W2V         = view.world_view_transform.T
    delta_world = torch.inverse(W2V)[:3, :3] @ delta_cam
    return delta_world, depth


# ═══════════════════════════════════════════════════════════════════════════════
# Visualisation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def make_comparison_cell(
    before_rgb: np.ndarray,
    cv2_after:  np.ndarray,   # BGR
    lama_after: np.ndarray,   # BGR
    updated_rgb: np.ndarray,  # RGB (shared for both methods)
    title: str,
    out_path: str,
) -> None:
    """
    2-row × 3-col grid:
      cv2 row:  Before | cv2 inpaint after | GG updated render
      LaMa row: Before | LaMa after        | GG updated render
    """
    H, W   = before_rgb.shape[:2]
    pad    = 8
    lh     = 40   # label bar height
    th     = 44   # title bar height
    cols   = 3
    rows   = 2
    scale  = 0.75

    fscale_sm = 0.70
    fscale_ti = 0.80
    thick     = 1

    gw = cols * W + (cols + 1) * pad
    gh = th + rows * (H + lh) + (rows + 1) * pad
    canvas = np.full((gh, gw, 3), 18, dtype=np.uint8)

    # Title bar
    canvas[0:th, :] = (40, 40, 40)
    (tw, tth), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, fscale_ti, thick + 1)
    cv2.putText(canvas, title,
                ((gw - tw) // 2, (th + tth) // 2 - 2),
                cv2.FONT_HERSHEY_DUPLEX, fscale_ti, (230, 230, 230), thick + 1, cv2.LINE_AA)

    before_bgr  = cv2.cvtColor(before_rgb,  cv2.COLOR_RGB2BGR)
    updated_bgr = cv2.cvtColor(updated_rgb, cv2.COLOR_RGB2BGR)

    row_panels = [
        [before_bgr,  cv2_after,  updated_bgr],
        [before_bgr,  lama_after, updated_bgr],
    ]
    row_labels = [
        ["Before (GG render)", "After: cv2.inpaint",   "GG updated render"],
        ["Before (GG render)", "After: LaMa",          "GG updated render"],
    ]
    row_colors = [(60, 35, 35), (35, 55, 35)]  # row tint: reddish cv2, greenish lama

    for r in range(rows):
        y0_row = th + pad + r * (H + lh + pad)
        for c in range(cols):
            img   = row_panels[r][c]
            label = row_labels[r][c]
            x0    = pad + c * (W + pad)
            y0    = y0_row

            canvas[y0 : y0 + H, x0 : x0 + W] = img

            bar_y = y0 + H
            canvas[bar_y : bar_y + lh, x0 : x0 + W] = row_colors[r]
            (lw, lth), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, fscale_sm, thick)
            cv2.putText(canvas, label,
                        (x0 + (W - lw) // 2, bar_y + (lh + lth) // 2 - 2),
                        cv2.FONT_HERSHEY_DUPLEX, fscale_sm, (220, 220, 220), thick, cv2.LINE_AA)

    # Row divider
    ydiv = th + pad + H + lh + pad // 2
    cv2.line(canvas, (0, ydiv), (gw, ydiv), (80, 80, 80), 1)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cv2.imwrite(out_path, canvas)


def make_overview_grid(
    results: dict,               # (obj_id, shift_name) → result dict
    objects: list[int],
    shifts:  list[tuple],
    out_dir: str,
) -> None:
    """
    5×5 thumbnail grid — rows = objects, cols = shifts.
    Each cell: left half = cv2_after, right half = LaMa_after.
    """
    thumb_w = 240
    thumb_h = 178
    pad     = 8
    lh      = 32
    th      = 52
    col_lh  = 38

    n_rows = len(objects)
    n_cols = len(shifts)
    gw = n_cols * thumb_w + (n_cols + 1) * pad
    gh = th + col_lh + n_rows * (thumb_h + lh) + (n_rows + 1) * pad
    canvas = np.full((gh, gw, 3), 18, dtype=np.uint8)

    # Title
    canvas[0:th, :] = (35, 35, 35)
    title = "Experiment Overview: cv2.inpaint (left) | LaMa (right)  --  5 objects x 5 displacements"
    (tw, tth), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.putText(canvas, title, ((gw - tw) // 2, (th + tth) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 1, cv2.LINE_AA)

    # Column headers
    for ci, (sname, dx, dy) in enumerate(shifts):
        x0 = pad + ci * (thumb_w + pad)
        canvas[th : th + col_lh, x0 : x0 + thumb_w] = (45, 45, 45)
        hdr = f"{sname} ({dx:+d},{dy:+d})"
        (lw, lth), _ = cv2.getTextSize(hdr, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.putText(canvas, hdr,
                    (x0 + (thumb_w - lw) // 2, th + (col_lh + lth) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

    fscale = 0.42

    for ri, obj_id in enumerate(objects):
        y0_row = th + col_lh + pad + ri * (thumb_h + lh + pad)
        for ci, (sname, _, _) in enumerate(shifts):
            key = (obj_id, sname)
            x0  = pad + ci * (thumb_w + pad)
            y0  = y0_row

            if key not in results:
                canvas[y0 : y0 + thumb_h, x0 : x0 + thumb_w] = (40, 40, 40)
                cv2.putText(canvas, "N/A", (x0 + 5, y0 + thumb_h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
                continue

            r = results[key]
            cv2_img  = cv2.imread(r["cv2_after_path"])
            lama_img = cv2.imread(r["lama_after_path"])

            if cv2_img is None or lama_img is None:
                canvas[y0 : y0 + thumb_h, x0 : x0 + thumb_w] = (50, 30, 30)
                continue

            # Resize
            half_w = thumb_w // 2
            cv2_t  = cv2.resize(cv2_img,  (half_w, thumb_h))
            lama_t = cv2.resize(lama_img, (thumb_w - half_w, thumb_h))
            cell   = np.concatenate([cv2_t, lama_t], axis=1)
            canvas[y0 : y0 + thumb_h, x0 : x0 + thumb_w] = cell

            # Divider between halves
            cv2.line(canvas, (x0 + half_w, y0), (x0 + half_w, y0 + thumb_h),
                     (255, 255, 255), 1)

            # Label bar
            bar_y = y0 + thumb_h
            canvas[bar_y : bar_y + lh, x0 : x0 + thumb_w] = (30, 30, 30)
            actual_dx, actual_dy = r["actual_dx"], r["actual_dy"]
            lbl = f"cls{obj_id}  ({actual_dx:+d},{actual_dy:+d})"
            (lw2, lth2), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, fscale, 1)
            cv2.putText(canvas, lbl,
                        (x0 + (thumb_w - lw2) // 2, bar_y + (lh + lth2) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, fscale, (180, 180, 180), 1, cv2.LINE_AA)

        # Row label (left margin — skip, use caption in bar instead)

    cv2.imwrite(os.path.join(out_dir, "overview.png"), canvas)
    print(f"Overview → {out_dir}/overview.png  ({gw}×{gh} px)")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tmp_root = tempfile.mkdtemp(prefix="gg_batch_")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1: Load GG model → render → collect all masks → build cv2 afters
    #          → THEN free GPU → run all LaMa subprocesses
    # (GG model + render buffers fill ~8 GB; LaMa needs ~3 GB — can't coexist)
    # ══════════════════════════════════════════════════════════════════════════

    print("=" * 68)
    print("PHASE 1 — Render + mask extraction")
    t0 = time.time()
    gaussians, scene, pipeline, bg, clf, view = load_model()
    W, H = view.image_width, view.image_height
    print(f"  Model loaded in {time.time()-t0:.1f}s  |  {view.image_name}  {W}×{H} px")

    print("Rendering before frame …")
    before_rgb, pred_obj_img = render_frame(view, gaussians, pipeline, bg, clf, NUM_CLASSES)
    before_bgr = cv2.cvtColor(before_rgb, cv2.COLOR_RGB2BGR)
    before_tmp = os.path.join(tmp_root, "before.png")
    cv2.imwrite(before_tmp, before_bgr)

    print("Computing Gaussian indices …")
    obj_data: dict = {}
    for obj_id in TARGET_OBJECTS:
        mask  = (pred_obj_img == obj_id).astype(np.uint8)
        ys, xs = np.where(mask)
        if len(ys) == 0:
            print(f"  class {obj_id}: no pixels — skipping")
            continue
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        idx  = get_gaussian_indices(gaussians, clf, view, obj_id)
        print(f"  class {obj_id:3d}: {int(mask.sum()):6,} px  {len(idx):6,} Gaussians")
        obj_data[obj_id] = {"mask": mask, "bbox": bbox, "indices": idx}

    # Build per-experiment plan (clamp shifts, save masks, generate cv2 afters)
    plan: list[dict] = []
    for obj_id, od in obj_data.items():
        mask, bbox, indices = od["mask"], od["bbox"], od["indices"]
        for shift_name, dx_raw, dy_raw in SHIFTS:
            dx, dy   = clamp_shift(bbox, dx_raw, dy_raw, W, H)
            exp_name = f"obj_{obj_id:03d}_{shift_name}"
            exp_dir  = os.path.join(OUTPUT_DIR, exp_name)
            tmp_dir  = os.path.join(tmp_root, exp_name)
            os.makedirs(exp_dir, exist_ok=True)
            os.makedirs(tmp_dir, exist_ok=True)

            # cv2 after image (CPU-only, fast)
            cv2_after = make_cv2_after(before_bgr, mask, dx, dy)
            cv2_path  = os.path.join(exp_dir, "cv2_after.png")
            cv2.imwrite(cv2_path, cv2_after)
            cv2.imwrite(os.path.join(exp_dir, "before.png"), before_bgr)

            # Save mask for LaMa subprocess
            mask_npy = os.path.join(tmp_dir, "mask.npy")
            np.save(mask_npy, mask)

            plan.append({
                "obj_id": obj_id, "shift_name": shift_name,
                "dx_raw": dx_raw, "dy_raw": dy_raw, "dx": dx, "dy": dy,
                "exp_name": exp_name, "exp_dir": exp_dir, "tmp_dir": tmp_dir,
                "mask": mask, "indices": indices,
                "cv2_after": cv2_after, "cv2_path": cv2_path,
                "mask_npy": mask_npy,
                "lama_bg_path": os.path.join(exp_dir, "lama_background.png"),
                "lama_path": os.path.join(exp_dir, "lama_after.png"),
            })

    # FREE GPU so LaMa subprocess can load its model
    print("\nFreeing GG model from GPU …")
    del gaussians, clf, scene
    torch.cuda.empty_cache()
    print(f"  GPU free: {torch.cuda.memory_allocated() // 1024 // 1024} MB allocated")

    # ── Run all LaMa subprocesses (GPU now available) ─────────────────────────
    print(f"\nRunning {len(plan)} LaMa inpaintings …")
    for i, p in enumerate(plan):
        lama_ok = run_lama_subprocess(before_tmp, p["mask_npy"], p["lama_bg_path"])
        p["lama_ok"] = lama_ok
        if lama_ok:
            lama_after, lama_bg = make_lama_after(
                before_bgr, p["mask"], p["lama_bg_path"], p["dx"], p["dy"], H, W)
        else:
            print(f"  [{i+1}/{len(plan)}] LaMa FAILED for {p['exp_name']} — using cv2 fallback")
            lama_after = p["cv2_after"].copy()
        cv2.imwrite(p["lama_path"], lama_after)
        p["lama_after"] = lama_after
        if not lama_ok or (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(plan)}] {p['exp_name']}  lama_ok={lama_ok}")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2: Reload GG model → apply Gaussian updates → render → assemble
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 68)
    print("PHASE 2 — Gaussian updates + renders")
    gaussians, scene, pipeline, bg, clf, view = load_model()
    orig_xyz = gaussians._xyz.data.clone()

    results: dict[tuple, dict] = {}
    for i, p in enumerate(plan):
        obj_id, shift_name = p["obj_id"], p["shift_name"]
        dx, dy = p["dx"], p["dy"]
        indices = p["indices"]
        print(f"\n[{i+1:2d}/{len(plan)}] class {obj_id}  {shift_name}  ({dx:+d},{dy:+d})")

        if len(indices) == 0:
            updated_rgb = before_rgb
            depth, delta_world = None, None
        else:
            delta_world, depth = backproject_delta(view, gaussians, indices, dx, dy)
            with torch.no_grad():
                gaussians._xyz.data[indices] += delta_world
            updated_rgb, _ = render_frame(view, gaussians, pipeline, bg, clf, NUM_CLASSES)
            with torch.no_grad():
                gaussians._xyz.data.copy_(orig_xyz)
            print(f"    depth={depth:.2f}  delta=({float(delta_world[0]):+.3f},"
                  f"{float(delta_world[1]):+.3f},{float(delta_world[2]):+.3f})")

        cv2.imwrite(os.path.join(p["exp_dir"], "updated_render.png"),
                    cv2.cvtColor(updated_rgb, cv2.COLOR_RGB2BGR))

        # Comparison grid
        make_comparison_cell(
            before_rgb, p["cv2_after"], p["lama_after"], updated_rgb,
            title=f"Class {obj_id}  |  shift: {shift_name} ({dx:+d},{dy:+d} px)",
            out_path=os.path.join(p["exp_dir"], "comparison.png"),
        )

        # Report
        report = {
            "obj_id": obj_id, "shift_name": shift_name,
            "dx_requested": p["dx_raw"], "dy_requested": p["dy_raw"],
            "dx_actual": dx, "dy_actual": dy,
            "n_pixels": int(p["mask"].sum()),
            "n_gaussians": int(len(indices)),
            "depth": round(depth, 3) if depth is not None else None,
            "delta_world": ([round(float(v), 4) for v in delta_world.tolist()]
                            if delta_world is not None else None),
            "lama_ok": p["lama_ok"],
        }
        with open(os.path.join(p["exp_dir"], "report.json"), "w") as f:
            json.dump(report, f, indent=2)

        results[(obj_id, shift_name)] = {
            "cv2_after_path":  p["cv2_path"],
            "lama_after_path": p["lama_path"],
            "actual_dx": dx, "actual_dy": dy,
        }

    # ── Overview grid ─────────────────────────────────────────────────────────
    print("\nGenerating overview grid …")
    make_overview_grid(results, TARGET_OBJECTS, SHIFTS, OUTPUT_DIR)

    n_lama_ok = sum(1 for p in plan if p["lama_ok"])
    print("\n" + "=" * 68)
    print(f"DONE  —  {OUTPUT_DIR}/")
    print(f"  {len(plan)} experiments  "
          f"({len(obj_data)} objects x {len(SHIFTS)} shifts x 2 methods)")
    print(f"  LaMa success: {n_lama_ok}/{len(plan)}")
    print(f"\nSubfolders:")
    for name in sorted(os.listdir(OUTPUT_DIR)):
        p2 = os.path.join(OUTPUT_DIR, name)
        if os.path.isdir(p2):
            print(f"  {name}/  ({len(os.listdir(p2))} files)")
    print(f"\n  overview.png  — thumbnail grid for spreadsheet")
    print("=" * 68)
    print("=" * 68)


if __name__ == "__main__":
    main()
