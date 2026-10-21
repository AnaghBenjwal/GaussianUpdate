"""
data.py — builds the model input from a sample_io.py sample (see DESIGN.md).

Model input (C_IN = 11 channels at RES x RES):
    0-2  BEFORE RGB (ImageNet-normalised)       3-5  AFTER RGB
    6    object mask in BEFORE                  7-9  BEFORE point map, gravity-aligned camera frame, minus the
                                                     object anchor, / POINT_SCALE metres (norm='fixed') or / the
                                                     metric width of the crop (norm='crop'), clipped
    10   depth valid
Global vector (G_IN = 5): anchor distance, anchor direction (3, unit), metres per model pixel at the anchor.
Target: delta_grav (3, metres).

The crop is the bounding box of (object mask ∪ change mask) — both available at test time — scaled by CROP_SCALE
(jittered in training). mask_after is never used as an input.
"""
from __future__ import annotations
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

cv2.setNumThreads(0)                                          # one thread per DataLoader worker
RES = 256                                                     # model input size; train/predict may override (data.RES = ...)
C_IN, G_IN = 11, 5
POINT_SCALE = 0.5
CROP_SCALE = 1.3
MIN_SIDE = 48
CHANGE_THR = 20
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def grav_points(depth, K, horizon_deg, off=(0, 0)):
    """depth (H, W) metres (0 = invalid) -> (H, W, 3) points in the gravity-aligned camera frame.
    off = (x, y) pixel offset of `depth` inside the image K refers to."""
    H, W = depth.shape
    fx, fy, cx, cy = K
    u, v = np.meshgrid(np.arange(W, dtype=np.float32) + 0.5 + off[0], np.arange(H, dtype=np.float32) + 0.5 + off[1])
    xc, yc, zc = (u - cx) / fx * depth, (v - cy) / fy * depth, depth
    t = np.deg2rad(horizon_deg)
    c, s = np.cos(t), np.sin(t)
    return np.stack([xc, -c * yc - s * zc, -s * yc + c * zc], -1).astype(np.float32)   # M = M^-1 (symmetric, orthogonal)


def _points_at(depth, K, horizon_deg, xs, ys):
    """grav_points for the pixels (xs, ys) only -> (N, 3)."""
    fx, fy, cx, cy = K
    z = depth[ys, xs]
    xc, yc = (xs + 0.5 - cx) / fx * z, (ys + 0.5 - cy) / fy * z
    t = np.deg2rad(horizon_deg)
    c, s = np.cos(t), np.sin(t)
    return np.stack([xc, -c * yc - s * z, -s * yc + c * z], -1).astype(np.float32)


def load_sample(d: Path):
    m = json.load(open(d / "meta.json"))
    rd = lambda n: np.asarray(Image.open(d / n))
    return dict(meta=m, before=rd("before.png")[..., :3], after=rd("after.png")[..., :3],
                depth=rd("depth.png").astype(np.float32) / 1000.0, mask=rd("mask_before.png") > 127,
                mask_after=rd("mask_after.png") > 127)            # mask_after: only to place the truncation augmentation


def change_mask(a, b):
    return np.abs(a.astype(np.int16) - b.astype(np.int16)).max(-1) > CHANGE_THR


def roi_box(mask, chg):
    m = mask | chg
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        H, W = mask.shape
        return W / 2, H / 2, max(W, H) / 2
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    return (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0)


def augment_depth(depth, valid, rng):
    """Make simulator depth look like GG depth: global scale error, noise, holes and blurred edges."""
    d = depth * rng.normal(1.0, 0.03)                        # GG depth is ~3 % off (median)
    d = d * (1 + rng.normal(0, 0.01, d.shape).astype(np.float32))
    if rng.random() < 0.5:                                   # smooth depth discontinuities (GG blurs edges)
        k = int(rng.choice([3, 5, 7]))
        d = cv2.GaussianBlur(d, (k, k), 0)
    if rng.random() < 0.7:                                   # holes
        H, W = d.shape
        holes = np.zeros((H, W), np.uint8)
        for _ in range(rng.integers(1, 4)):
            cv2.circle(holes, (int(rng.integers(W)), int(rng.integers(H))), int(rng.integers(2, max(3, min(W, H) // 20))), 1, -1)
        valid = valid & (holes == 0)
    return d, valid


def augment_mask(mask, rng):
    m = mask.astype(np.uint8)
    r = rng.random()
    k = int(rng.integers(1, 4)) * 2 + 1
    if r < 0.3:
        e = cv2.erode(m, np.ones((k, k), np.uint8))
        if e.sum() >= 0.6 * m.sum():                          # never erase thin objects
            m = e
    elif r < 0.6:
        m = cv2.dilate(m, np.ones((k, k), np.uint8))
    if rng.random() < 0.3:                                   # stray Gaussians of the same class
        H, W = m.shape
        for _ in range(rng.integers(1, 4)):
            cv2.circle(m, (int(rng.integers(W)), int(rng.integers(H))), int(rng.integers(1, 4)), 1, -1)
    return m.astype(bool) if m.any() else mask


def augment_truncate(before, after, depth, valid, mask, mask_after, rng):
    """Simulate the frame border cutting through the object (0% of training views are truncated, 13-49% of test
    views are): everything beyond a random line through the object's box is blanked in every channel."""
    m = mask if mask_after is None else (mask | mask_after)
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return before, after, depth, valid, mask
    side = int(rng.integers(4))
    if side < 2:                                              # left / right border
        lo, hi = xs.min(), xs.max()
    else:                                                     # top / bottom border
        lo, hi = ys.min(), ys.max()
    cut = int(lo + rng.uniform(0.2, 0.6) * (hi - lo)) if side % 2 == 0 else int(hi - rng.uniform(0.2, 0.6) * (hi - lo))
    keep = np.ones(m.shape, bool)
    if side == 0: keep[:, :cut] = False
    elif side == 1: keep[:, cut:] = False
    elif side == 2: keep[:cut] = False
    else: keep[cut:] = False
    if not (mask & keep).any():
        return before, after, depth, valid, mask
    return (before * keep[..., None], after * keep[..., None], depth * keep, valid & keep, mask & keep)


def build_input(s, rng=None, norm="crop"):
    """Returns x (C_IN, RES, RES) float32, g (G_IN,) float32, y (3,) float32 or None, info dict.
    g[4] * RES = metric width of the crop at the object (the model's output unit when norm='crop')."""
    m = s["meta"]
    before, after, depth, mask = s["before"], s["after"], s["depth"], s["mask"]
    valid = depth > 0
    train = rng is not None
    if train:
        depth, valid = augment_depth(depth, valid, rng)
        mask = augment_mask(mask, rng)
        if rng.random() < 0.25:
            before, after, depth, valid, mask = augment_truncate(before, after, depth, valid, mask, s.get("mask_after"), rng)
    chg = change_mask(before, after)
    cx, cy, side = roi_box(mask, chg)
    scale = CROP_SCALE
    if train:
        scale = rng.uniform(1.1, 1.6)
        cx += rng.uniform(-0.1, 0.1) * side
        cy += rng.uniform(-0.1, 0.1) * side
    side = max(side * scale, MIN_SIDE)
    # work only on the window the crop covers (+1 px), in context-crop pixels
    H, W = mask.shape
    wx0, wy0 = max(int(np.floor(cx - side / 2)) - 1, 0), max(int(np.floor(cy - side / 2)) - 1, 0)
    wx1, wy1 = min(int(np.ceil(cx + side / 2)) + 1, W), min(int(np.ceil(cy + side / 2)) + 1, H)
    if wx1 <= wx0 or wy1 <= wy0:
        wx0, wy0, wx1, wy1 = 0, 0, W, H
    win = (slice(wy0, wy1), slice(wx0, wx1))
    a = RES / side                                            # crop px -> model px
    A = np.array([[a, 0, RES / 2 - a * (cx - wx0)], [0, a, RES / 2 - a * (cy - wy0)]], np.float32)
    warp = lambda x, interp: cv2.warpAffine(x, A, (RES, RES), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    K = m["K_crop"]
    sel = mask & valid
    if sel.sum() < 3:
        sel = valid & (cv2.dilate(mask.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0)
    if sel.sum():                                             # anchor from the whole (context) object
        ys, xs = np.nonzero(sel)
        anchor = np.median(_points_at(depth, K, m["cam_horizon"], xs, ys), 0)
    else:
        anchor = np.array([0, 0, 1.0], np.float32)
    dw, vw = depth[win], valid[win]
    P = grav_points(np.where(vw, dw, 0), K, m["cam_horizon"], off=(wx0, wy0))
    crop_m = anchor[2] / (K[0] * a) * RES
    Pr = np.clip((P - anchor) / (max(crop_m, 0.02) if norm == "crop" else POINT_SCALE), -4, 4) * vw[..., None]

    b = before[win].astype(np.float32) / 255
    f = after[win].astype(np.float32) / 255
    if train:                                                 # shared photometric jitter + small independent noise
        gain, bias = rng.uniform(0.8, 1.2, 3).astype(np.float32), rng.uniform(-0.05, 0.05, 3).astype(np.float32)
        b = np.clip(b * gain + bias, 0, 1)
        f = np.clip(f * gain + bias + rng.normal(0, 0.01, 3).astype(np.float32), 0, 1)
    b, f = (b - MEAN) / STD, (f - MEAN) / STD
    x = np.concatenate([warp(b, cv2.INTER_LINEAR), warp(f, cv2.INTER_LINEAR),
                        warp(mask[win].astype(np.float32), cv2.INTER_NEAREST)[..., None],
                        warp(Pr.astype(np.float32), cv2.INTER_NEAREST),
                        warp(vw.astype(np.float32), cv2.INTER_NEAREST)[..., None]], -1)
    dist = float(np.linalg.norm(anchor))
    g = np.array([dist, *(anchor / max(dist, 1e-6)), anchor[2] / (K[0] * a)], np.float32)
    y = np.array(m["delta_grav"], np.float32) if "delta_grav" in m else None
    if train and rng.random() < 0.5:                         # horizontal flip: x of points, anchor and target
        x = x[:, ::-1].copy()
        x[..., 7] *= -1
        g[1] *= -1
        if y is not None:
            y = y * np.array([-1, 1, 1], np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1)), g, y, dict(crop=[float(cx), float(cy), float(side)])


class DispDataset(torch.utils.data.Dataset):
    def __init__(self, roots, train=False, seed=0, cache=False, norm="crop"):
        self.norm = norm
        self.dirs = sorted(p.parent for r in roots for p in Path(r).glob("*/*/meta.json"))
        self.train, self.seed, self.epoch = train, seed, 0
        self.cache = {} if cache else None

    def preload(self):
        """Decode every sample once in this process; forked DataLoader workers share the arrays (copy-on-write)."""
        self.cache = {i: load_sample(d) for i, d in enumerate(self.dirs)}

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, i):
        d = self.dirs[i]
        if self.cache is not None and i in self.cache:
            s = self.cache[i]
        else:
            s = load_sample(d)
            if self.cache is not None:
                self.cache[i] = s
        rng = np.random.default_rng([self.seed, self.epoch, i]) if self.train else None
        x, g, y, info = build_input(s, rng, self.norm)
        return dict(x=torch.from_numpy(x), g=torch.from_numpy(g),
                    y=torch.from_numpy(y) if y is not None else torch.zeros(3), idx=i)
