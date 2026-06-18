"""
SAM2 video segmentation demo on GG rendered frames.
Segments the main object across all frames and saves a video with mask overlay.
"""
import os
import torch
import numpy as np
import cv2
from pathlib import Path

FRAMES_DIR = "/tmp/test_frames"
OUTPUT_DIR = "output/sam2_demo"
CHECKPOINT = "weights/sam2/sam2.1_hiera_large.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load frames and get image size
frame_paths = sorted(Path(FRAMES_DIR).glob("*.jpg"))
sample = cv2.imread(str(frame_paths[0]))
H, W = sample.shape[:2]
print(f"Found {len(frame_paths)} frames, size {W}x{H}")

# Build SAM2 video predictor
from sam2.build_sam import build_sam2_video_predictor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

predictor = build_sam2_video_predictor(MODEL_CFG, CHECKPOINT, device=device)

# Use center-ish point as prompt for first frame (targets the main object region)
# Figurines scene — center point tends to hit one of the figurines
cx, cy = W // 2, H // 2

with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
    state = predictor.init_state(video_path=FRAMES_DIR)

    # Add click prompt on frame 0: center point, positive click
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=state,
        frame_idx=0,
        obj_id=1,
        points=np.array([[cx, cy]], dtype=np.float32),
        labels=np.array([1], dtype=np.int32),
    )

    # Propagate across all frames
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
        video_segments[out_frame_idx] = {
            obj_id: (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
            for i, obj_id in enumerate(out_obj_ids)
        }

print(f"Propagated masks for {len(video_segments)} frames")

# Save video with mask overlay
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_vid = cv2.VideoWriter(f"{OUTPUT_DIR}/sam2_segmentation.mp4", fourcc, 10, (W, H))

for frame_idx, frame_path in enumerate(frame_paths):
    img = cv2.imread(str(frame_path))
    overlay = img.copy()

    if frame_idx in video_segments:
        for obj_id, mask in video_segments[frame_idx].items():
            color = np.array([0, 120, 255], dtype=np.uint8)  # orange-ish
            overlay[mask] = (overlay[mask] * 0.5 + color * 0.5).astype(np.uint8)
            # Draw contour
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 200, 255), 2)

    # Label frame
    cv2.putText(overlay, f"SAM2 Frame {frame_idx:03d}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    out_vid.write(overlay)

    # Also save individual frames
    frame_out = Path(OUTPUT_DIR) / f"frame_{frame_idx:05d}.png"
    cv2.imwrite(str(frame_out), overlay)

out_vid.release()
print(f"Saved video to {OUTPUT_DIR}/sam2_segmentation.mp4")
print(f"Saved {len(frame_paths)} individual frames to {OUTPUT_DIR}/")
