"""
sample_io.py — one on-disk format for every displacement sample (training, validation and the benchmark test sets).

A sample is one camera view of one object move: BEFORE photo, AFTER photo, BEFORE depth, the object's BEFORE mask,
plus camera and ground truth. Everything is stored as a "context crop" around the change at the 1600x900 evaluation
resolution (same resize as the GG renders), so the model's own crop (with jitter) can be cut from it at train time.

    <root>/<split>/<scene>/<sample_id>/
        before.png  after.png         RGB uint8 context crop
        depth.png                     BEFORE depth, uint16 millimetres (0 = invalid), camera z
        mask_before.png               object mask in BEFORE (training: simulator instance mask; GG test variant: A0)
        mask_after.png                object mask in AFTER (ground truth only — never a model input)
        meta.json                     see make_meta()

Frames (all metres):
    unity    AI2THOR world (left-handed, y up)
    gg       fixed-dataset / GG world = diag(1, 1, -1) @ unity
    grav     gravity-aligned camera frame: x = camera right, y = up, z = camera forward projected on the floor.
             Independent of camera yaw, so it is the frame the model predicts in. delta_grav = R_grav @ delta_unity.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from PIL import Image

FULL_W, FULL_H = 1600, 900                 # evaluation resolution (GG auto-downscale of 1920x1080)
SRC_W, SRC_H = 1920, 1080                  # simulator render resolution
CTX_MIN, CTX_MAX = 320, 640                # context crop side (px at 1600x900); larger crops are downscaled to CTX_MAX
CTX_FACTOR = 2.0                           # context side = CTX_FACTOR * max(ROI w, h)
CHANGE_THR = 20                            # |after - before| max-channel threshold (uint8) for the change mask
S_GG = np.diag([1.0, 1.0, -1.0])


def resize_rgb(x: np.ndarray) -> np.ndarray:
    return np.asarray(Image.fromarray(x).resize((FULL_W, FULL_H)))          # PIL default filter, as GG's PILtoTorch


def resize_nearest(x: np.ndarray) -> np.ndarray:
    if x.dtype == bool:
        return np.asarray(Image.fromarray(x.astype(np.uint8) * 255).resize((FULL_W, FULL_H), Image.NEAREST)) > 127
    return np.asarray(Image.fromarray(x).resize((FULL_W, FULL_H), Image.NEAREST))


def intrinsics(fov_v_deg: float, w=FULL_W, h=FULL_H) -> list:
    f = (h / 2) / np.tan(np.deg2rad(fov_v_deg) / 2)                           # AI2THOR fieldOfView is vertical
    return [f, f, w / 2, h / 2]


def grav_rotation(yaw_deg: float) -> np.ndarray:
    """unity world -> gravity-aligned camera frame (x right, y up, z forward on the floor)."""
    y = np.deg2rad(yaw_deg)
    fwd = np.array([np.sin(y), 0.0, np.cos(y)])
    right = np.array([np.cos(y), 0.0, -np.sin(y)])
    return np.stack([right, [0.0, 1.0, 0.0], fwd])


def change_mask(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    return np.abs(after.astype(np.int16) - before.astype(np.int16)).max(-1) > CHANGE_THR


def context_box(*masks: np.ndarray):
    """Square box (x0, y0, side) around the union of masks, CTX_FACTOR x its size, clamped inside the frame."""
    m = np.zeros_like(masks[0], bool)
    for k in masks:
        m |= k
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    side = int(np.clip(CTX_FACTOR * max(x1 - x0, y1 - y0), CTX_MIN, FULL_H))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bx = int(np.clip(round(cx - side / 2), 0, FULL_W - side))
    by = int(np.clip(round(cy - side / 2), 0, FULL_H - side))
    return bx, by, side


def touches_border(m: np.ndarray) -> bool:
    return bool(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any())


def write_sample(d: Path, before, after, depth_m, mask_before, mask_after, meta: dict, box=None):
    """All image inputs at FULL_W x FULL_H. Crops to the context box, writes the files, returns the final meta."""
    chg = change_mask(before, after)
    box = box or context_box(mask_before, mask_after, chg)
    bx, by, side = box
    out = min(side, CTX_MAX)
    sc = out / side
    sl = (slice(by, by + side), slice(bx, bx + side))

    def crop(x, nearest):
        x = x[sl]
        if out == side:
            return x
        if x.dtype == bool:
            return np.asarray(Image.fromarray(x.astype(np.uint8) * 255).resize((out, out), Image.NEAREST)) > 127
        return np.asarray(Image.fromarray(x).resize((out, out), Image.NEAREST if nearest else Image.BILINEAR))

    d.mkdir(parents=True, exist_ok=True)
    Image.fromarray(crop(before, False)).save(d / "before.png")
    Image.fromarray(crop(after, False)).save(d / "after.png")
    dmm = np.clip(np.nan_to_num(depth_m, nan=0.0) * 1000.0, 0, 65535).astype(np.uint16)
    Image.fromarray(crop(dmm, True)).save(d / "depth.png")
    Image.fromarray(crop(mask_before, True).astype(np.uint8) * 255).save(d / "mask_before.png")
    Image.fromarray(crop(mask_after, True).astype(np.uint8) * 255).save(d / "mask_after.png")
    fx, fy, cx, cy = meta["K_full"]
    meta = dict(meta)
    meta.update(crop_box_full=[bx, by, side], crop_size=out, crop_scale=sc,
                K_crop=[fx * sc, fy * sc, (cx - bx) * sc, (cy - by) * sc],
                px_before=int(mask_before.sum()), px_after=int(mask_after.sum()), px_change=int(chg.sum()),
                truncated_before=touches_border(mask_before), truncated_after=touches_border(mask_after))
    json.dump(meta, open(d / "meta.json", "w"), indent=1)
    return meta


def make_meta(scene, sample_id, split, obj, cam, delta_unity, extra=None) -> dict:
    """obj: AI2THOR object dict (BEFORE). cam: dict(pos_unity, yaw, horizon, fov). delta_unity: actual object move."""
    du = np.asarray(delta_unity, float)
    R = grav_rotation(cam["yaw"])
    m = dict(scene=scene, sample_id=sample_id, split=split,
             object_id=obj["objectId"], object_type=obj["objectType"],
             cam_pos_unity=list(map(float, cam["pos_unity"])), cam_yaw=float(cam["yaw"]), cam_horizon=float(cam["horizon"]),
             fov_v=float(cam["fov"]), K_full=intrinsics(cam["fov"]),
             delta_unity=du.tolist(), delta_gg=(S_GG @ du).tolist(), delta_grav=(R @ du).tolist(),
             delta_norm=float(np.linalg.norm(du)),
             obj_pos_unity=[float(obj["position"][a]) for a in "xyz"])
    m.update(extra or {})
    return m
