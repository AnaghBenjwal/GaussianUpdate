#!/usr/bin/env python3
"""
lama_inpaint.py — LaMa (Large Mask inpainting) helper.
Runs in the `dev` conda env.

Usage
-----
    python scripts/lama_inpaint.py \
        --image  /path/to/image.png  \
        --mask   /path/to/mask.npy   \
        --output /path/to/result.png \
        --dilate 15

The mask (H,W) uint8 marks the region to fill (255 = inpaint, 0 = keep).
Outputs a single RGB PNG with the masked region filled by LaMa.
"""

import argparse, os, sys
import numpy as np
import cv2
from PIL import Image


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--image",  required=True, help="Input RGB image (PNG/JPG)")
    p.add_argument("--mask",   required=True, help="(H,W) uint8 mask .npy OR .png")
    p.add_argument("--output", required=True, help="Output PNG path")
    p.add_argument("--dilate", type=int, default=15,
                   help="Dilation kernel size to expand mask before inpainting")
    args = p.parse_args()

    # Load image
    img = Image.open(args.image).convert("RGB")

    # Load mask — accept .npy or image file
    if args.mask.endswith(".npy"):
        mask_arr = np.load(args.mask).astype(np.uint8)
    else:
        mask_arr = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)

    # Dilate mask so LaMa fills the object border cleanly
    if args.dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.dilate, args.dilate))
        mask_arr = cv2.dilate(mask_arr, k)

    # Binarise: LaMa expects white = inpaint
    mask_arr = (mask_arr > 0).astype(np.uint8) * 255
    mask_pil = Image.fromarray(mask_arr)

    print(f"Image: {img.size}  |  Mask coverage: {mask_arr.sum() // 255:,} px", flush=True)

    # Run LaMa
    from simple_lama_inpainting import SimpleLama
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"LaMa device: {device}", flush=True)

    lama = SimpleLama(device=device)
    result = lama(img, mask_pil)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    result.save(args.output)
    print(f"Saved → {args.output}", flush=True)


if __name__ == "__main__":
    main()
