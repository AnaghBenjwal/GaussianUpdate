#!/usr/bin/env python3
"""
before_after.py — Render a before/after image pair from AI2THOR with a known object displacement.

The camera is placed at the same position as cam0 from collect_data.py (the canonical
view), so detect_and_update.py with --cam_idx 0 will use the matching GG camera.

Run in dev conda env:
    conda run -n dev python scripts/sim/before_after.py \\
        --scene FloorPlan1 \\
        --meta data/sim/FloorPlan1/scene_meta.json \\
        --object_idx 0 \\
        --dx 0.3 --dz 0.0 \\
        --out data/sim/FloorPlan1/before_after/obj0_right

Outputs:
    before.png     — simulator render before move (for visual reference)
    after.png      — simulator render after move (fed to detect_and_update)
    meta.json      — object, displacement, ground-truth pixel shift
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "sim"))
from colmap_writer import look_at_c2w


def render_before_after(
    scene: str,
    scene_meta: dict,
    object_idx: int,
    dx: float,
    dz: float,
    out_dir: Path,
) -> dict:
    import ai2thor.controller

    out_dir.mkdir(parents=True, exist_ok=True)

    W, H   = scene_meta["width"], scene_meta["height"]
    fov    = scene_meta["fov_deg"]

    obj_id  = scene_meta["object_ids"][object_idx]
    cls_idx = scene_meta["objects"][object_idx]["class_idx"]

    cam0      = scene_meta["cam0"]
    eye       = np.array(cam0["eye"])
    look_tgt  = np.array(cam0["look_target"])
    yaw       = cam0["yaw"]
    horizon   = cam0["horizon"]

    controller = ai2thor.controller.Controller(
        agentMode="default",
        scene=scene,
        headless=False,
        platform="CloudRendering",
        renderDepthImage=False,
        renderInstanceSegmentation=True,
        width=W, height=H,
        fieldOfView=fov,
        snapToGrid=False,
    )

    def _teleport_cam():
        ev = controller.step(dict(
            action="TeleportFull",
            x=float(eye[0]), y=float(eye[1]), z=float(eye[2]),
            rotation={"x": 0.0, "y": yaw, "z": 0.0},
            horizon=horizon,
            standing=True,
            renderImage=True,
        ))
        if not ev.metadata.get("lastActionSuccess", False):
            raise RuntimeError(
                f"Camera teleport failed: {ev.metadata.get('errorMessage')}\n"
                f"  eye={eye.tolist()} yaw={yaw:.1f} horizon={horizon:.1f}"
            )
        return ev

    # ── Before render ─────────────────────────────────────────────────────────
    event = _teleport_cam()
    before_rgb = event.frame.copy()
    Image.fromarray(before_rgb).save(out_dir / "before.png")

    inst_b  = event.instance_segmentation_frame
    cmap_b  = event.color_to_object_id
    px_before = _object_pixels(inst_b, cmap_b, obj_id)
    if len(px_before) == 0:
        controller.stop()
        raise ValueError(
            f"Object '{obj_id}' not visible from canonical camera.\n"
            "  Try a different object_idx or increase orbit radius in collect_data.py."
        )
    centre_before = px_before.mean(axis=0)
    print(f"Before: object '{obj_id}' — {len(px_before)} px, centre "
          f"({centre_before[0]:.0f}, {centre_before[1]:.0f})")

    # ── Move object ───────────────────────────────────────────────────────────
    objects = {o["objectId"]: o for o in event.metadata["objects"]}
    orig_pos = objects[obj_id]["position"]
    new_pos  = {"x": orig_pos["x"] + dx, "y": orig_pos["y"], "z": orig_pos["z"] + dz}

    move_ev = controller.step(
        action="PlaceObjectAtPoint",
        objectId=obj_id,
        position=new_pos,
    )
    if not move_ev.metadata.get("lastActionSuccess", False):
        # Fallback: TeleportObject (available in some AI2THOR versions)
        orig_rot = objects[obj_id].get("rotation", {"x": 0, "y": 0, "z": 0})
        move_ev = controller.step(
            action="TeleportObject",
            objectId=obj_id,
            x=new_pos["x"], y=new_pos["y"], z=new_pos["z"],
            rotation=orig_rot,
        )
    if not move_ev.metadata.get("lastActionSuccess", False):
        controller.stop()
        raise RuntimeError(f"Move failed: {move_ev.metadata.get('errorMessage')}")
    print(f"Moved: {orig_pos} → {new_pos}")

    # ── After render ─────────────────────────────────────────────────────────
    event_a = _teleport_cam()
    after_rgb = event_a.frame.copy()
    Image.fromarray(after_rgb).save(out_dir / "after.png")

    inst_a = event_a.instance_segmentation_frame
    cmap_a = event_a.color_to_object_id
    px_after = _object_pixels(inst_a, cmap_a, obj_id)

    if len(px_after) > 0:
        centre_after = px_after.mean(axis=0)
        gt_dx = float(centre_after[0] - centre_before[0])
        gt_dy = float(centre_after[1] - centre_before[1])
        print(f"After:  centre ({centre_after[0]:.0f}, {centre_after[1]:.0f})")
        print(f"GT pixel shift: dx={gt_dx:.1f}  dy={gt_dy:.1f}")
    else:
        gt_dx = gt_dy = None
        print("Warning: object not visible after move (off-screen or occluded)")

    controller.stop()

    meta = {
        "scene": scene,
        "object_id": obj_id,
        "class_idx": cls_idx,
        "world_dx": dx,
        "world_dz": dz,
        "gt_pixel_dx": gt_dx,
        "gt_pixel_dy": gt_dy,
        "orig_pos": orig_pos,
        "new_pos": new_pos,
        "cam0": cam0,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved → {out_dir}")
    return meta


def _object_pixels(inst: np.ndarray, color_map: dict, obj_id: str) -> np.ndarray:
    """Return (N, 2) array of (col, row) pixel coords for obj_id."""
    H, W = inst.shape[:2]
    flat = inst.reshape(-1, 3)
    mask = np.zeros(H * W, dtype=bool)
    for (r, g, b), oid in color_map.items():
        if oid == obj_id:
            mask |= (flat[:, 0] == r) & (flat[:, 1] == g) & (flat[:, 2] == b)
    idx  = np.where(mask)[0]
    cols = (idx % W).astype(np.float32)
    rows = (idx // W).astype(np.float32)
    return np.stack([cols, rows], axis=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="FloorPlan1")
    parser.add_argument("--meta", default="data/sim/FloorPlan1/scene_meta.json")
    parser.add_argument("--object_idx", type=int, default=0)
    parser.add_argument("--dx", type=float, default=0.3,
                        help="World-space X displacement (metres)")
    parser.add_argument("--dz", type=float, default=0.0,
                        help="World-space Z displacement (metres)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--display", default=":99")
    args = parser.parse_args()

    if "DISPLAY" not in os.environ:
        os.environ["DISPLAY"] = args.display

    with open(args.meta) as f:
        meta = json.load(f)

    render_before_after(
        scene=args.scene,
        scene_meta=meta,
        object_idx=args.object_idx,
        dx=args.dx, dz=args.dz,
        out_dir=Path(args.out),
    )
