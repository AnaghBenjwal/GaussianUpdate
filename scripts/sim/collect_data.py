#!/usr/bin/env python3
"""
collect_data.py — Capture multi-view training data from AI2THOR.

Run in dev conda env:
    conda run -n dev python scripts/sim/collect_data.py [--scene FloorPlan1] [--out data/sim]

Outputs to --out/<scene>/:
    images/             RGB frames (PNG)
    object_mask/        grayscale PNG, pixel value = object class ID (0=bg)
    sparse/0/           COLMAP binary (cameras.bin, images.bin, points3D.bin)
    scene_meta.json     camera FOV, object list, cam0 parameters

Camera 0 is ALWAYS the "canonical" view: agent at (0, 0.9, 1.5) facing the scene
centre. Use cam_idx=0 in detect_and_update.py.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "sim"))
from colmap_writer import look_at_c2w, ai2thor_lookat_params, write_colmap_sparse

# ── Config ───────────────────────────────────────────────────────────────────
AGENT_Y      = 0.9                          # default agent height (m)
FOV_DEG      = 90.0
MAX_OBJECTS  = 12
# Steeper downward angles for close-up tabletop object capture
HORIZONS     = [0.0, 20.0, 40.0, 60.0]


def fov_to_focal(fov_deg: float, dim: int) -> float:
    return dim / (2.0 * np.tan(np.deg2rad(fov_deg) / 2.0))


def compute_look_target(moveable_objects: list[dict]) -> np.ndarray:
    """Compute scene look-target as mean XZ of moveable object positions, Y fixed at 0.7."""
    if not moveable_objects:
        return np.array([0.0, 0.7, 0.0])
    positions = [o["position"] for o in moveable_objects if "position" in o]
    if not positions:
        return np.array([0.0, 0.7, 0.0])
    mean_x = float(np.mean([p["x"] for p in positions]))
    mean_z = float(np.mean([p["z"] for p in positions]))
    return np.array([mean_x, 0.7, mean_z])


def build_cameras_from_reachable(
    reachable_positions: list[dict],
    look_target: np.ndarray,
    max_dist: float | None = None,
) -> list[dict]:
    """
    Build camera specs from AI2THOR reachable positions.
    Camera 0 is the reachable position that gives the best frontal view of
    look_target (furthest in +Z from target, within max_dist).
    Each position gets one camera per HORIZONS angle.

    max_dist: if set, only use positions within this many metres of look_target.
    """
    pos_arr = np.array([[p['x'], p['y'], p['z']] for p in reachable_positions])
    dists_to_target = np.linalg.norm(pos_arr[:, [0, 2]] - look_target[[0, 2]], axis=1)

    if max_dist is not None:
        mask = dists_to_target <= max_dist
        filtered = [p for p, m in zip(reachable_positions, mask) if m]
        filtered_arr = pos_arr[mask]
        if len(filtered) == 0:
            print(f"  WARNING: max_dist={max_dist} excluded all positions; using all.")
            filtered = reachable_positions
            filtered_arr = pos_arr
    else:
        filtered = reachable_positions
        filtered_arr = pos_arr

    # Camera 0: position in front of look_target (largest +Z offset from it)
    rel_z = filtered_arr[:, 2] - look_target[2]
    cam0_idx = int(np.argmax(rel_z))
    cam0_pos = filtered[cam0_idx]

    cameras = []
    eye0 = np.array([cam0_pos['x'], cam0_pos['y'], cam0_pos['z']])
    c2w0 = look_at_c2w(eye0, look_target)
    yaw0, _ = ai2thor_lookat_params(eye0, look_target)
    cameras.append({"eye": eye0, "c2w": c2w0, "yaw": yaw0, "horizon": HORIZONS[0]})

    for i, p in enumerate(filtered):
        if i == cam0_idx:
            continue
        eye = np.array([p['x'], p['y'], p['z']])
        yaw, _ = ai2thor_lookat_params(eye, look_target)
        for hor in HORIZONS:
            c2w = look_at_c2w(eye, look_target)
            cameras.append({"eye": eye, "c2w": c2w, "yaw": yaw, "horizon": hor})

    return cameras, eye0  # also return cam0 eye for metadata


def collect(scene: str, out_root: Path, max_dist: float | None = 1.2) -> dict:
    """
    max_dist: only use reachable positions within this many metres of the object
              cluster centre. 1.2m keeps cameras close to objects so their
              features are large and clearly resolved in the image.
    """
    import ai2thor.controller

    out_dir = out_root / scene / "env"
    img_dir = out_dir / "images"
    msk_dir = out_dir / "object_mask"
    img_dir.mkdir(parents=True, exist_ok=True)
    msk_dir.mkdir(parents=True, exist_ok=True)

    W, H = 1920, 1080
    fx = fy = fov_to_focal(FOV_DEG, W)
    cx, cy = W / 2.0, H / 2.0

    print(f"Starting AI2THOR — scene: {scene}")
    controller = ai2thor.controller.Controller(
        agentMode="default",
        scene=scene,
        headless=False,
        platform="CloudRendering",
        renderDepthImage=True,
        renderInstanceSegmentation=True,
        width=W,
        height=H,
        fieldOfView=FOV_DEG,
        snapToGrid=False,
    )

    # ── Enumerate objects ────────────────────────────────────────────────────
    event = controller.step(dict(action="GetReachablePositions", renderImage=False))
    reachable = event.metadata["reachablePositions"]
    objects   = event.metadata["objects"]
    moveable  = sorted(
        [o for o in objects if o.get("moveable") or o.get("pickupable")],
        key=lambda o: o["objectId"],
    )[:MAX_OBJECTS]
    obj_to_cls = {o["objectId"]: i + 1 for i, o in enumerate(moveable)}

    look_target = compute_look_target(moveable)
    print(f"  Reachable positions: {len(reachable)}")
    print(f"  Look target (auto): {look_target.tolist()}")
    print(f"  Max camera distance: {max_dist}")
    print(f"  Moveable objects tracked: {len(moveable)}")
    for o in moveable:
        print(f"    cls={obj_to_cls[o['objectId']]:2d}  {o['objectId']}")

    # ── Build camera list ────────────────────────────────────────────────────
    camera_specs, cam0_eye = build_cameras_from_reachable(reachable, look_target, max_dist)
    print(f"  Planned cameras: {len(camera_specs)}")

    image_names, c2w_list, depth_imgs, rgb_imgs = [], [], [], []

    for spec in camera_specs:
        eye = spec["eye"]
        event = controller.step(dict(
            action="TeleportFull",
            x=float(eye[0]), y=float(eye[1]), z=float(eye[2]),
            rotation={"x": 0.0, "y": spec["yaw"], "z": 0.0},
            horizon=spec["horizon"],
            standing=True,
            renderImage=True,
        ))
        if not event.metadata.get("lastActionSuccess", False):
            continue   # wall/object collision — skip

        frame_id = len(image_names)
        fname = f"frame_{frame_id:05d}.png"

        rgb = event.frame                       # (H, W, 3) uint8
        depth = event.depth_frame               # (H, W) float32, metres
        inst = event.instance_segmentation_frame
        color_map = event.color_to_object_id

        Image.fromarray(rgb).save(img_dir / fname)
        mask = _inst_to_mask(inst, color_map, obj_to_cls, H, W)
        Image.fromarray(mask).save(msk_dir / fname)

        image_names.append(fname)
        c2w_list.append(spec["c2w"])
        depth_imgs.append(depth)
        rgb_imgs.append(rgb)

        if (frame_id + 1) % 10 == 0:
            print(f"    captured {frame_id + 1}/{len(camera_specs)}")

    controller.stop()
    print(f"  Total frames: {len(image_names)}")

    # ── Write COLMAP sparse ──────────────────────────────────────────────────
    write_colmap_sparse(
        out_dir=out_dir,
        image_names=image_names,
        c2w_matrices=c2w_list,
        width=W, height=H,
        fx=fx, fy=fy, cx=cx, cy=cy,
        depth_images=depth_imgs,
        rgb_images=rgb_imgs,
        max_depth=8.0,
    )

    # ── Metadata ─────────────────────────────────────────────────────────────
    cam0_spec = camera_specs[0]
    meta = {
        "scene": scene,
        "width": W, "height": H,
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "fov_deg": FOV_DEG,
        "num_frames": len(image_names),
        "object_ids": [o["objectId"] for o in moveable],
        "objects": [
            {"object_id": o["objectId"], "class_idx": obj_to_cls[o["objectId"]],
             "orig_pos": o["position"]}
            for o in moveable
        ],
        "cam0": {
            "eye": cam0_spec["eye"].tolist(),
            "look_target": look_target.tolist(),
            "yaw": cam0_spec["yaw"],
            "horizon": cam0_spec["horizon"],
        },
    }
    with open(out_dir / "scene_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Done → {out_dir}")
    return meta


def _inst_to_mask(inst: np.ndarray, color_map: dict,
                  obj_to_cls: dict, H: int, W: int) -> np.ndarray:
    mask = np.zeros(H * W, dtype=np.uint8)
    inst_flat = inst.reshape(-1, 3)
    for (r, g, b), obj_id in color_map.items():
        if obj_id not in obj_to_cls:
            continue
        cls = obj_to_cls[obj_id]
        m = (inst_flat[:, 0] == r) & (inst_flat[:, 1] == g) & (inst_flat[:, 2] == b)
        mask[m] = cls
    return mask.reshape(H, W)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene",    default="FloorPlan1")
    parser.add_argument("--out",      default=str(Path.home() / "Documents" / "dataset_1080p"))
    parser.add_argument("--max_dist", type=float, default=1.2,
                        help="Max camera distance from object cluster centre (m)")
    parser.add_argument("--display",  default=":99")
    args = parser.parse_args()

    if "DISPLAY" not in os.environ:
        os.environ["DISPLAY"] = args.display

    collect(scene=args.scene, out_root=Path(args.out), max_dist=args.max_dist)
