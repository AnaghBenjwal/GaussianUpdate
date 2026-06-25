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
LOOK_TARGET  = np.array([0.0, 0.7, 0.0])   # scene centre, mid-table height
AGENT_Y      = 0.9                          # default agent height (m)
FOV_DEG      = 90.0
MAX_OBJECTS  = 12
# Three pitch angles give ~300 views when combined with all reachable positions
HORIZONS     = [5.0, 20.0, 35.0]

# Camera 0: canonical front-centre position (closest reachable to this target)
CAM0_TARGET  = np.array([0.25, AGENT_Y, 1.5])   # preferred cam0 location


def fov_to_focal(fov_deg: float, dim: int) -> float:
    return dim / (2.0 * np.tan(np.deg2rad(fov_deg) / 2.0))


def build_cameras_from_reachable(reachable_positions: list[dict]) -> list[dict]:
    """
    Build camera specs from AI2THOR reachable positions.
    Camera 0 is the reachable position closest to CAM0_TARGET.
    Each position gets one camera per HORIZONS angle.
    """
    pos_arr = np.array([[p['x'], p['y'], p['z']] for p in reachable_positions])

    # Find camera 0: closest reachable position to the preferred location
    dists = np.linalg.norm(pos_arr - CAM0_TARGET[np.newaxis], axis=1)
    cam0_idx = int(np.argmin(dists))
    cam0_pos = reachable_positions[cam0_idx]

    cameras = []

    # Camera 0 first (canonical view, horizon=HORIZONS[0])
    eye0 = np.array([cam0_pos['x'], cam0_pos['y'], cam0_pos['z']])
    c2w0 = look_at_c2w(eye0, LOOK_TARGET)
    yaw0, _ = ai2thor_lookat_params(eye0, LOOK_TARGET)
    cameras.append({"eye": eye0, "c2w": c2w0, "yaw": yaw0, "horizon": HORIZONS[0]})

    # All other reachable positions (skip the cam0 position)
    other_indices = [i for i in range(len(reachable_positions)) if i != cam0_idx]

    for i in other_indices:
        p = reachable_positions[i]
        eye = np.array([p['x'], p['y'], p['z']])
        yaw, _ = ai2thor_lookat_params(eye, LOOK_TARGET)
        for hor in HORIZONS:
            c2w = look_at_c2w(eye, LOOK_TARGET)
            cameras.append({"eye": eye, "c2w": c2w, "yaw": yaw, "horizon": hor})

    return cameras


def collect(scene: str, out_root: Path) -> dict:
    import ai2thor.controller

    out_dir = out_root / scene / "env"
    img_dir = out_dir / "images"
    msk_dir = out_dir / "object_mask"
    img_dir.mkdir(parents=True, exist_ok=True)
    msk_dir.mkdir(parents=True, exist_ok=True)

    W, H = 640, 480
    fx = fy = fov_to_focal(FOV_DEG, W)
    cx, cy = W / 2.0, H / 2.0

    print(f"Starting AI2THOR — scene: {scene}")
    # CloudRendering uses Vulkan GPU rendering, works headlessly without X display
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
    print(f"  Reachable positions: {len(reachable)}")
    print(f"  Moveable objects tracked: {len(moveable)}")
    for o in moveable:
        print(f"    cls={obj_to_cls[o['objectId']]:2d}  {o['objectId']}")

    # ── Build camera list ────────────────────────────────────────────────────
    camera_specs = build_cameras_from_reachable(reachable)
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
    # Camera 0 is the canonical view (front-centre)
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
            "look_target": LOOK_TARGET.tolist(),
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
    parser.add_argument("--scene", default="FloorPlan1")
    parser.add_argument("--out", default=str(Path.home() / "Documents" / "dataset"))
    parser.add_argument("--display", default=":99")
    args = parser.parse_args()

    if "DISPLAY" not in os.environ:
        os.environ["DISPLAY"] = args.display

    collect(scene=args.scene, out_root=Path(args.out))
