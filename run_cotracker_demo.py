"""
CoTracker demo: track 15 points on the main object from frame 0 through all frames.
Uses CoTrackerOnlinePredictor with local checkpoint (no torch.hub download).
Saves video with point tracks overlaid.
"""
import os
import torch
import numpy as np
import cv2
from pathlib import Path

FRAMES_DIR = "/tmp/test_frames"
OUTPUT_DIR = "output/cotracker_demo"
CHECKPOINT = "weights/cotracker/cotracker3_scaled_online.pth"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load all frames as a tensor (T, H, W, C) -> (1, T, 3, H, W)
frame_paths = sorted(Path(FRAMES_DIR).glob("*.jpg"))
frames_np = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in frame_paths]
T, H, W = len(frames_np), frames_np[0].shape[0], frames_np[0].shape[1]
print(f"Loaded {T} frames, {W}x{H}")

video = torch.from_numpy(np.stack(frames_np)).permute(0, 3, 1, 2).float()  # (T, 3, H, W)
video = video.unsqueeze(0)  # (1, T, 3, H, W)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
video = video.to(device)
print(f"Using device: {device}")

# Load model from local checkpoint
from cotracker.predictor import CoTrackerOnlinePredictor
model = CoTrackerOnlinePredictor(checkpoint=CHECKPOINT)
model = model.to(device)
model.eval()
print(f"Loaded CoTracker (window_len={model.model.window_len}, step={model.step})")

# Define 15 query points spread across center region (frame 0)
cx, cy = W // 2, H // 2
offsets = [
    (-80, -80), (0, -80), (80, -80),
    (-80,   0), (0,   0), (80,   0),
    (-80,  80), (0,  80), (80,  80),
    (-40, -40), (40, -40), (-40, 40), (40, 40),
    (-120, 0), (120, 0),
]
queries = []
for dx, dy in offsets:
    x = float(max(0, min(W - 1, cx + dx)))
    y = float(max(0, min(H - 1, cy + dy)))
    queries.append([0.0, x, y])  # [frame_idx, x, y]

queries_tensor = torch.tensor(queries, dtype=torch.float32, device=device).unsqueeze(0)  # (1, N, 3)
N = len(queries)
print(f"Tracking {N} points")

# Online processing: init step, then sliding window
step = model.step          # 8 frames per step
window = model.step * 2    # 16-frame window

# Init step (is_first_step=True)
model(
    video_chunk=video[:, :window],
    is_first_step=True,
    queries=queries_tensor,
    add_support_grid=False,
)

# Process rest of video in windows
pred_tracks, pred_visibility = None, None
for ind in range(0, T - step, step):
    chunk = video[:, ind : ind + window]
    if chunk.shape[1] < 2:
        break
    pred_tracks, pred_visibility = model(
        video_chunk=chunk,
        add_support_grid=False,
        grid_size=0,
    )

# pred_tracks: (1, T_out, N, 2), pred_visibility: (1, T_out, N)
tracks = pred_tracks[0].cpu().numpy()        # (T_out, N, 2)
visibility = pred_visibility[0].cpu().numpy()  # (T_out, N)
T_out = tracks.shape[0]
print(f"Track tensor: {tracks.shape}")

# Color palette for tracked points
colors = [
    (255,   0,   0), (  0, 255,   0), (  0,   0, 255),
    (255, 255,   0), (  0, 255, 255), (255,   0, 255),
    (255, 128,   0), (128,   0, 255), (  0, 128, 255),
    (255,   0, 128), (  0, 255, 128), (128, 255,   0),
    (200, 100,  50), ( 50, 200, 100), (100,  50, 200),
]

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_vid = cv2.VideoWriter(f"{OUTPUT_DIR}/cotracker_tracks.mp4", fourcc, 10, (W, H))

for t, frame_path in enumerate(frame_paths):
    img = cv2.imread(str(frame_path))
    t_idx = min(t, T_out - 1)

    # Draw trail (last 10 frames)
    trail_len = 10
    for n in range(N):
        color_bgr = tuple(reversed(colors[n % len(colors)]))
        t_start = max(0, t_idx - trail_len)
        for tt in range(t_start, t_idx):
            tt2 = min(tt + 1, T_out - 1)
            if visibility[tt, n] and visibility[tt2, n]:
                p1 = (int(tracks[tt, n, 0]), int(tracks[tt, n, 1]))
                p2 = (int(tracks[tt2, n, 0]), int(tracks[tt2, n, 1]))
                cv2.line(img, p1, p2, color_bgr, 2)

    # Draw current point
    for n in range(N):
        if t_idx < T_out and visibility[t_idx, n]:
            x, y = int(tracks[t_idx, n, 0]), int(tracks[t_idx, n, 1])
            color_bgr = tuple(reversed(colors[n % len(colors)]))
            cv2.circle(img, (x, y), 5, color_bgr, -1)
            cv2.circle(img, (x, y), 7, (255, 255, 255), 1)

    cv2.putText(img, f"CoTracker Frame {t:03d} | {N} pts", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    out_vid.write(img)
    cv2.imwrite(f"{OUTPUT_DIR}/frame_{t:05d}.png", img)

out_vid.release()
print(f"Saved video to {OUTPUT_DIR}/cotracker_tracks.mp4")
print(f"Saved {T} frames to {OUTPUT_DIR}/")
