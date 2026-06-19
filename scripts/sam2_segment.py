#!/usr/bin/env python3
"""
sam2_segment.py — SAM2 bbox-prompted segmentation helper.

Runs in the ``dev`` conda env (Python ≥ 3.10, SAM2 installed).
Called as a subprocess by detect_and_update.py.

Usage:
    conda run -n dev python3 scripts/sam2_segment.py \\
        --image  /path/to/after.png \\
        --bbox   x1 y1 x2 y2       \\
        --output /path/to/mask.npy

Output: binary uint8 mask saved as .npy (1 = object, 0 = background).
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def run(image_path: str, bbox, output_path: str):
    """Segment the region indicated by ``bbox`` in ``image_path`` using SAM2."""
    # SAM2 image predictor (single-frame, no video propagation needed here)
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    weights = str(ROOT / "weights" / "sam2" / "sam2.1_hiera_large.pt")
    cfg     = "configs/sam2.1/sam2.1_hiera_l.yaml"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = build_sam2(cfg, weights, device=device)
    predictor = SAM2ImagePredictor(model)

    image = np.array(Image.open(image_path).convert("RGB"))
    predictor.set_image(image)

    # bbox format SAM2 expects: [x1, y1, x2, y2]
    box = np.array([[bbox[0], bbox[1], bbox[2], bbox[3]]], dtype=np.float32)
    masks, scores, _ = predictor.predict(
        box=box,
        multimask_output=True,
    )
    # Take the highest-scoring mask
    best = masks[scores.argmax()].astype(np.uint8)
    np.save(output_path, best)
    print(f"[sam2_segment] Saved mask {best.shape} → {output_path}  "
          f"(coverage: {best.mean()*100:.1f}%)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image",  required=True)
    p.add_argument("--bbox",   nargs=4, type=int, required=True,
                   metavar=("X1", "Y1", "X2", "Y2"))
    p.add_argument("--output", required=True)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.image, args.bbox, args.output)
