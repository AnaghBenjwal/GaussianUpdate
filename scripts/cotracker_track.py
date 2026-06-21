#!/usr/bin/env python3
"""
cotracker_track.py — Track object points through a motion video using CoTracker3.

Runs in the `dev` conda env (Python 3.10, CoTracker + SAM2 installed).
Called as a subprocess from track_and_update.py (gaussian_grouping env).

Usage
-----
    python scripts/cotracker_track.py \
        --frames_dir /path/to/frame_pngs/  \
        --mask       /path/to/mask.npy     \
        --output     /path/to/disp.json    \
        --viz_out    /path/to/tracks.mp4   \
        --checkpoint weights/cotracker/cotracker3_scaled_online.pth \
        --n_grid     6

Input
-----
    frames_dir : directory of frame_0000.png … frame_NNNN.png (RGB, all same size)
    mask       : (H, W) uint8/bool — object mask in frame 0 for query points

Output
------
    JSON: {"shift_x": ..., "shift_y": ..., "n_tracks": ...,
           "n_visible_last": ..., "n_frames": ...}
"""

from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

import numpy as np
import torch
import cv2


DEFAULT_CKPT = "weights/cotracker/cotracker3_scaled_online.pth"


def sample_query_points(mask: np.ndarray, n_grid: int) -> list[list[float]]:
    """Return a grid of [frame=0, x, y] query points inside ``mask``."""
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return []
    y_lo, y_hi = int(ys.min()) + 5, int(ys.max()) - 5
    x_lo, x_hi = int(xs.min()) + 5, int(xs.max()) - 5
    if y_hi <= y_lo or x_hi <= x_lo:
        y_lo, y_hi = int(ys.min()), int(ys.max())
        x_lo, x_hi = int(xs.min()), int(xs.max())
    pts: list[list[float]] = []
    for gy in np.linspace(y_lo, y_hi, n_grid):
        for gx in np.linspace(x_lo, x_hi, n_grid):
            ri, ci = int(round(gy)), int(round(gx))
            ri = np.clip(ri, 0, mask.shape[0] - 1)
            ci = np.clip(ci, 0, mask.shape[1] - 1)
            if mask[ri, ci]:
                pts.append([0.0, float(ci), float(ri)])  # frame, x, y
    return pts


def make_track_video(frames_np: list[np.ndarray],
                     tracks: np.ndarray,
                     visibility: np.ndarray,
                     out_path: str,
                     fps: int = 8) -> None:
    """Save a visualisation video with coloured point trails."""
    T, N = tracks.shape[:2]
    H, W = frames_np[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    # Generate N distinct colours using HSV
    colors: list[tuple[int, int, int]] = []
    for i in range(N):
        hue = int(180 * i / max(N - 1, 1))
        c = cv2.cvtColor(np.uint8([[[hue, 220, 200]]]), cv2.COLOR_HSV2BGR)[0, 0]
        colors.append((int(c[0]), int(c[1]), int(c[2])))

    for t in range(min(T, len(frames_np))):
        frame_bgr = cv2.cvtColor(frames_np[min(t, len(frames_np) - 1)].copy(),
                                  cv2.COLOR_RGB2BGR)
        for k in range(N):
            if not visibility[t, k]:
                continue
            x, y = int(tracks[t, k, 0]), int(tracks[t, k, 1])
            col = colors[k]
            cv2.circle(frame_bgr, (x, y), 4, col, -1)
            # Trail: draw line from previous position
            if t > 0 and visibility[t - 1, k]:
                xp, yp = int(tracks[t - 1, k, 0]), int(tracks[t - 1, k, 1])
                cv2.line(frame_bgr, (xp, yp), (x, y), col, 2)
        writer.write(frame_bgr)
    writer.release()


def main() -> None:
    p = argparse.ArgumentParser(description="CoTracker3 displacement estimator")
    p.add_argument("--frames_dir", required=True,
                   help="Directory with frame_0000.png … frame_NNNN.png")
    p.add_argument("--mask",       required=True,
                   help="(H,W) uint8 mask npy for query points in frame 0")
    p.add_argument("--output",     required=True, help="Output JSON path")
    p.add_argument("--viz_out",    default=None,  help="Optional track visualisation mp4")
    p.add_argument("--n_grid",     type=int, default=6,
                   help="Grid density for query sampling")
    p.add_argument("--checkpoint", default=DEFAULT_CKPT)
    args = p.parse_args()

    # ── Load frames ──────────────────────────────────────────────────────────
    frame_paths = sorted(Path(args.frames_dir).glob("frame_*.png"))
    if not frame_paths:
        print("ERROR: no frame_*.png in", args.frames_dir, file=sys.stderr)
        sys.exit(1)

    frames_np = [
        cv2.cvtColor(cv2.imread(str(fp)), cv2.COLOR_BGR2RGB)
        for fp in frame_paths
    ]
    T = len(frames_np)
    H, W = frames_np[0].shape[:2]
    print(f"Frames: {T}  |  {W}×{H}", flush=True)

    # ── Sample query points from initial mask ─────────────────────────────────
    mask = np.load(args.mask).astype(np.uint8)
    pts = sample_query_points(mask, args.n_grid)
    if len(pts) < 4:
        ys, xs = np.where(mask > 0)
        step = max(1, len(ys) // 36)
        pts = [[0.0, float(xs[i]), float(ys[i])] for i in range(0, len(ys), step)]
    N = len(pts)
    print(f"Query points: {N}", flush=True)

    # ── Build video tensor (1, T, 3, H, W) ───────────────────────────────────
    video = (
        torch.from_numpy(np.stack(frames_np))   # (T, H, W, 3)
        .permute(0, 3, 1, 2)                    # (T, 3, H, W)
        .float()
        .unsqueeze(0)                           # (1, T, 3, H, W)
    )
    queries = torch.tensor(pts, dtype=torch.float32).unsqueeze(0)  # (1, N, 3)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    video   = video.to(device)
    queries = queries.to(device)

    # ── Load CoTracker online predictor ───────────────────────────────────────
    from cotracker.predictor import CoTrackerOnlinePredictor
    model = CoTrackerOnlinePredictor(checkpoint=args.checkpoint)
    model = model.to(device)
    model.eval()

    step   = model.step        # typically 8
    window = step * 2          # typically 16

    print(f"CoTracker  window={window}  step={step}", flush=True)

    # ── Online inference ──────────────────────────────────────────────────────
    # Initialise with the first window
    model(
        video[:, : min(window, T)],
        is_first_step=True,
        queries=queries,
        add_support_grid=False,
    )

    # Slide through the rest of the video
    pred_tracks: torch.Tensor | None = None
    pred_visibility: torch.Tensor | None = None

    for ind in range(0, T - step, step):
        chunk = video[:, ind : min(ind + window, T)]
        if chunk.shape[1] < 2:
            break
        pred_tracks, pred_visibility = model(
            chunk, add_support_grid=False, grid_size=0
        )

    # If video fits in a single window (T <= window), the loop may not run —
    # run one more pass to get the actual tracks.
    if pred_tracks is None:
        pred_tracks, pred_visibility = model(
            video[:, : min(window, T)],
            add_support_grid=False,
            grid_size=0,
        )

    # ── Extract shift: last vs first tracked position ─────────────────────────
    # pred_tracks:     (1, T', N, 2)  —  x, y per frame
    # pred_visibility: (1, T', N)
    tracks = pred_tracks[0].cpu().numpy()       # (T', N, 2)
    vis    = pred_visibility[0].cpu().numpy()   # (T', N)

    vis_last = vis[-1].astype(bool)
    if vis_last.sum() < 1:
        vis_last = np.ones(N, dtype=bool)

    delta = tracks[-1, vis_last] - tracks[0, vis_last]   # (K, 2)
    shift = delta.mean(axis=0)                            # [shift_x, shift_y]

    result = {
        "shift_x":       float(shift[0]),
        "shift_y":       float(shift[1]),
        "n_tracks":      N,
        "n_visible_last": int(vis_last.sum()),
        "n_frames":      T,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result), flush=True)

    # ── Optional track visualisation ─────────────────────────────────────────
    if args.viz_out:
        make_track_video(frames_np, tracks, vis, args.viz_out)
        print(f"Viz → {args.viz_out}", flush=True)


if __name__ == "__main__":
    main()
