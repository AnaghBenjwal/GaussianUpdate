"""
dino_raft.py — DINOv2 semantic change detection + RAFT optical flow displacement.

Replaces the pixel-diff + template-matching steps in detect_and_update.py with a
domain-gap-robust pipeline:

  1. DINOv2 (ViT-B/14) computes per-patch cosine similarity between the GG
     render (before) and the real AI2THOR frame (after). Patches below the
     similarity threshold are flagged as semantically changed.

  2. RAFT optical flow estimates a dense per-pixel 2D displacement field
     between the two images. The median flow over the masked object region
     gives the 2D pixel shift used for 3D backprojection.

  3. 3D backprojection is unchanged — handled by backproject_displacement()
     in detect_and_update.py.

Usage from detect_and_update.py
    from scripts.dino_raft import detect_changed_region_dino, estimate_displacement_raft

Environment: gaussian_grouping conda env (PyTorch 2.4+, torchvision 0.19+).
Models download automatically to ~/.cache/torch/hub/ on first call.
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# ─── Module-level model singletons (lazy-loaded) ─────────────────────────────

_dino_model = None
_raft_model = None

_DINO_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
_DINO_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
_PATCH_SIZE = 14   # DINOv2 ViT-B/14 patch size in pixels


def _get_dino(device: str = "cuda") -> torch.nn.Module:
    global _dino_model
    if _dino_model is None:
        print("[dino_raft] Loading DINOv2 ViT-B/14 …")
        _dino_model = (
            torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", verbose=False)
            .to(device)
            .eval()
        )
    return _dino_model


def _get_raft(device: str = "cuda") -> torch.nn.Module:
    global _raft_model
    if _raft_model is None:
        print("[dino_raft] Loading RAFT-large …")
        try:
            from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
            _raft_model = (
                raft_large(weights=Raft_Large_Weights.DEFAULT).to(device).eval()
            )
        except TypeError:
            from torchvision.models.optical_flow import raft_large
            _raft_model = raft_large(pretrained=True).to(device).eval()
    return _raft_model


# ─── DINOv2 helpers ───────────────────────────────────────────────────────────

def _to_dino_tensor(
    rgb_uint8: np.ndarray, device: str
) -> Tuple[torch.Tensor, int, int]:
    """
    Crop image to the nearest multiple of PATCH_SIZE, normalise to ImageNet
    stats, and return (tensor (1,3,H',W'), n_patches_h, n_patches_w).
    """
    H, W = rgb_uint8.shape[:2]
    nh = H // _PATCH_SIZE
    nw = W // _PATCH_SIZE
    H_crop = nh * _PATCH_SIZE
    W_crop = nw * _PATCH_SIZE

    img = cv2.resize(rgb_uint8, (W_crop, H_crop), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(img).float() / 255.0          # (H', W', 3)
    mean = _DINO_MEAN.to(t.device)
    std  = _DINO_STD.to(t.device)
    t = (t - mean) / std                               # ImageNet normalise
    t = t.permute(2, 0, 1).unsqueeze(0).to(device)    # (1, 3, H', W')
    return t, nh, nw


# ─── RAFT helpers ─────────────────────────────────────────────────────────────

def _to_raft_tensor(rgb_uint8: np.ndarray, device: str) -> torch.Tensor:
    """
    Convert uint8 (H, W, 3) RGB → float32 (1, 3, H_pad, W_pad) in [0, 255].

    RAFT requires H and W divisible by 8; pad with border replication if needed.
    """
    H, W = rgb_uint8.shape[:2]
    H_pad = ((H + 7) // 8) * 8
    W_pad = ((W + 7) // 8) * 8
    if H_pad != H or W_pad != W:
        img = cv2.copyMakeBorder(
            rgb_uint8, 0, H_pad - H, 0, W_pad - W, cv2.BORDER_REPLICATE
        )
    else:
        img = rgb_uint8
    return (
        torch.from_numpy(img).float()
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )


def _flow_to_color(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """HSV colour-wheel visualisation of a 2D flow field → uint8 (H, W, 3) RGB."""
    magnitude = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)
    hue = ((angle + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
    sat = np.full_like(hue, 255)
    max_mag = float(magnitude.max()) or 1.0
    val = (magnitude / max_mag * 255).astype(np.uint8)
    hsv = np.stack([hue, sat, val], axis=2)
    return cv2.cvtColor(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2RGB)


# ─── Public API ───────────────────────────────────────────────────────────────

def detect_changed_region_dino(
    before_rgb:          np.ndarray,
    after_rgb:           np.ndarray,
    similarity_threshold: float = 0.8,
    morph_kernel_size:    int   = 5,
    device:              str   = "cuda",
) -> np.ndarray:
    """
    Detect changed pixels using DINOv2 patch-level cosine similarity.

    Patches whose before/after cosine similarity falls below
    ``similarity_threshold`` are flagged as semantically changed. The patch-
    level binary map is upsampled to the original resolution and cleaned with
    morphological closing + dilation.

    DINOv2's semantic features are robust to the domain gap between a
    GG-rendered image and a real AI2THOR frame: irrelevant rendering
    differences (background shading, anti-aliasing) don't fire because they
    occupy the same semantic patch as the surrounding scene.

    Parameters
    ----------
    before_rgb, after_rgb    : uint8 (H, W, 3) arrays in RGB order
    similarity_threshold     : patches below this cosine similarity are changed
                               (0.8 works well for rigid object displacements)
    morph_kernel_size        : ellipse kernel size for morphological cleanup

    Returns
    -------
    change_mask : uint8 (H, W), 1 = changed, 0 = unchanged
    """
    H, W = before_rgb.shape[:2]
    model = _get_dino(device)

    with torch.no_grad():
        t_b, nh, nw = _to_dino_tensor(before_rgb, device)
        t_a, _,  _  = _to_dino_tensor(after_rgb,  device)

        # x_norm_patchtokens: (1, N_patches, D=768)  N = nh * nw
        feats_b = model.forward_features(t_b)["x_norm_patchtokens"]
        feats_a = model.forward_features(t_a)["x_norm_patchtokens"]

        # Per-patch cosine similarity → (nh, nw)
        sim = F.cosine_similarity(feats_b, feats_a, dim=-1)
        sim = sim.squeeze(0).reshape(nh, nw).cpu().numpy()

    changed_patches = (sim < similarity_threshold).astype(np.uint8)

    change_map = cv2.resize(changed_patches, (W, H), interpolation=cv2.INTER_NEAREST)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
    )
    change_map = cv2.morphologyEx(change_map, cv2.MORPH_CLOSE, kernel)
    change_map = cv2.dilate(change_map, kernel, iterations=1)

    n_changed = int(change_map.sum())
    print(f"[dino_raft] DINOv2 changed pixels: {n_changed:,} "
          f"/ {H * W:,}  (threshold={similarity_threshold})")
    return change_map.astype(np.uint8)


def estimate_displacement_raft(
    before_rgb:  np.ndarray,
    after_rgb:   np.ndarray,
    object_mask: np.ndarray,
    device:      str = "cuda",
    min_mask_px: int = 20,
) -> Tuple[float, float, float, np.ndarray]:
    """
    Estimate 2D pixel displacement of an object using RAFT optical flow.

    RAFT computes a dense per-pixel flow field (before → after). The median
    flow vector over all pixels in ``object_mask`` gives a robust estimate of
    the object's displacement, rejecting outliers from occluding/disoccluding
    boundaries.

    Parameters
    ----------
    before_rgb, after_rgb : uint8 (H, W, 3) arrays
    object_mask           : uint8 (H, W) — 1 where the object is
                            (typically DINO change mask ∩ GG object mask)
    min_mask_px           : if fewer masked pixels, return zero displacement

    Returns
    -------
    dx         : float — median horizontal pixel displacement (before → after)
    dy         : float — median vertical pixel displacement
    confidence : float — 1 / (1 + std of flow vectors); 1.0 = perfectly consistent
    flow_viz   : uint8 (H, W, 3) — HSV colour-wheel visualisation of flow field
    """
    H, W = before_rgb.shape[:2]
    ys, xs = np.where(object_mask > 0)

    flow_viz = np.zeros((H, W, 3), dtype=np.uint8)
    if len(ys) < min_mask_px:
        print(f"[dino_raft] Object mask has only {len(ys)} px — returning zero flow.")
        return 0.0, 0.0, 0.0, flow_viz

    model = _get_raft(device)
    t_b = _to_raft_tensor(before_rgb, device)
    t_a = _to_raft_tensor(after_rgb,  device)

    with torch.no_grad():
        flow_list = model(t_b, t_a)
        flow = flow_list[-1]                   # (1, 2, H_pad, W_pad)
        flow = flow[:, :, :H, :W]             # crop to original size

    flow_np = flow.squeeze(0).cpu().numpy()    # (2, H, W)
    dx_field = flow_np[0]                      # (H, W)  horizontal
    dy_field = flow_np[1]                      # (H, W)  vertical

    dx_samples = dx_field[ys, xs]
    dy_samples = dy_field[ys, xs]

    dx_med = float(np.median(dx_samples))
    dy_med = float(np.median(dy_samples))

    flow_std = float(np.std(np.stack([dx_samples, dy_samples], axis=1)))
    confidence = 1.0 / (1.0 + flow_std)

    flow_viz = _flow_to_color(dx_field, dy_field)

    print(
        f"[dino_raft] RAFT  dx={dx_med:+.1f}  dy={dy_med:+.1f} px  "
        f"std={flow_std:.2f}  conf={confidence:.3f}  mask_px={len(ys):,}"
    )
    return dx_med, dy_med, confidence, flow_viz
