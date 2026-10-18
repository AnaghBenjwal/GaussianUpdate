#!/usr/bin/env python3
"""
gen_trials.py — Ground-truth object-displacement trials for the fixed AI2THOR datasets.

For every tracked object of a scene, tries a set of planar moves on its support surface and keeps
the ones that are physically clean (object stays at the same height/rotation, nothing else moves,
object visible in the input view before and after). For each kept trial it renders the AFTER scene
from the input view (frame_00000) and every held-out test view (every 8th frame, the GG --eval split).
BEFORE renders are identical for all trials of a scene (scene reset), so they are stored once.

World coordinates: the fixed dataset uses world' = diag(1, 1, -1) @ unity_world, so every 3D quantity
written here is already in GG/COLMAP world coordinates ("_gg" keys).

Run in the dev env (AI2THOR):
    conda run -n dev python scripts/bench/gen_trials.py --scene FloorPlan1
Outputs  ~/Documents/displacement_bench/<scene>/
    views.json                       eval views (name, teleport params)
    before/<view>.png                1600x900 before renders (same resize as GG training)
    trials/<obj>_<k>/after/<view>.png
    trials/<obj>_<k>/mask_before/<view>.png, mask_after/<view>.png   (moved-object instance masks)
    trials/<obj>_<k>/meta.json
"""
from __future__ import annotations
import argparse, json, os, re
from pathlib import Path

import numpy as np
from PIL import Image

W, H = 1920, 1080
EVAL_RES = (1600, 900)                 # GG/3DGS auto-downscale target (width 1600)
S = np.diag([1.0, 1.0, -1.0])          # unity -> fixed-dataset world
MOVES = [(0.15, 0.0), (-0.15, 0.0), (0.0, 0.15), (0.0, -0.15),
         (0.30, 0.0), (-0.30, 0.0), (0.0, 0.30), (0.0, -0.30)]
MAX_TRIALS_PER_OBJ = 4
MIN_VIS_PX = 300                       # at 1920x1080, input view


def to_gg(p) -> list:
    v = np.array([p["x"], p["y"], p["z"]], float) if isinstance(p, dict) else np.asarray(p, float)
    return (S @ v.T).T.tolist()


def obj_mask(ev, oid) -> np.ndarray:
    m = ev.instance_masks.get(oid)
    return np.zeros((H, W), bool) if m is None else m


def save_rgb(path: Path, rgb: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).resize(EVAL_RES).save(path)          # PIL default filter, as in GG's PILtoTorch


def save_mask(path: Path, m: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((m * 255).astype(np.uint8)).resize(EVAL_RES, Image.NEAREST).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="FloorPlan1")
    ap.add_argument("--data", default=str(Path.home() / "Documents/dataset_1080p_fixed"))
    ap.add_argument("--out", default=str(Path.home() / "Documents/displacement_bench"))
    args = ap.parse_args()
    os.environ.setdefault("DISPLAY", ":99")
    import ai2thor.controller

    meta = json.load(open(f"{args.data}/{args.scene}/env/scene_meta.json"))
    poses = {p["name"]: p for p in meta["true_poses"]}
    names = sorted(poses)
    views = [n for i, n in enumerate(names) if i % 8 == 0]       # GG --eval test split; views[0] = frame_00000 = input view
    out = Path(args.out) / args.scene
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"input_view": views[0], "views": [poses[v] for v in views]}, open(out / "views.json", "w"), indent=1)

    c = ai2thor.controller.Controller(agentMode="default", scene=args.scene, platform="CloudRendering",
                                      renderInstanceSegmentation=True, width=W, height=H,
                                      fieldOfView=meta["fov_deg"], snapToGrid=False)

    def look(v):
        p = poses[v]
        ev = c.step(dict(action="TeleportFull", x=p["spec_eye"][0], y=p["spec_eye"][1], z=p["spec_eye"][2],
                         rotation={"x": 0.0, "y": p["yaw"], "z": 0.0}, horizon=p["spec_horizon"],
                         standing=True, renderImage=True))
        assert ev.metadata["lastActionSuccess"], ev.metadata["errorMessage"]
        return ev

    def objects(ev):
        return {o["objectId"]: o for o in ev.metadata["objects"]}

    # ── before renders (once per scene) + reference check against the training images ────────
    c.reset(scene=args.scene)
    before_masks, before_objs = {}, None
    for v in views:
        ev = look(v)
        save_rgb(out / "before" / v, ev.frame)
        ref = np.asarray(Image.open(f"{args.data}/{args.scene}/env/images/{v}").convert("RGB"), np.float32)
        d = float(np.abs(ref - ev.frame.astype(np.float32)).mean())
        assert d < 1.0, f"before render of {v} differs from training image (mean |diff| {d:.2f})"
        before_masks[v] = {o["object_id"]: obj_mask(ev, o["object_id"]) for o in meta["objects"]}
        before_objs = objects(ev)

    # ── no-move control: some objects drift on their own after a reset (FP1 DishSponge slides ~5 cm).
    # "pushed another object" is measured against where each object ends up WITHOUT our move, with a
    # per-object tolerance from the spread of 3 control runs.
    def settle_run(move=None):
        c.reset(scene=args.scene)
        ev0 = look(views[0]); objs0 = objects(ev0)
        ev = c.step(action="Pass") if move is None else move(objs0)
        ok = ev.metadata["lastActionSuccess"]
        for _ in range(5):                                           # let physics settle
            ev = c.step(action="Pass")
        return ok, objs0, objects(ev)

    def pos(o):
        return np.array([o["position"][a] for a in "xyz"])

    ctrl = [settle_run()[2] for _ in range(3)]
    ctrl_mean = {k: np.mean([pos(r[k]) for r in ctrl], 0) for k in ctrl[0]}
    ctrl_tol = {k: 0.01 + max(np.linalg.norm(pos(r[k]) - ctrl_mean[k]) for r in ctrl) for k in ctrl[0]}
    drift = {k.split("|")[0]: round(float(np.linalg.norm(ctrl_mean[k] - pos(before_objs[k]))), 3)
             for k in ctrl[0] if k in before_objs and np.linalg.norm(ctrl_mean[k] - pos(before_objs[k])) > 0.005}
    print(f"self-drifting objects in control runs (m): {drift}")

    summary = []
    for o in meta["objects"]:
        oid, cls = o["object_id"], o["class_idx"]
        name = re.sub(r"(?<!^)(?=[A-Z])", "_", oid.split("|")[0]).lower()
        if before_masks[views[0]][oid].sum() < MIN_VIS_PX:
            print(f"skip {name}: not visible in input view"); continue
        kept = 0
        for dx, dz in MOVES:
            if kept >= MAX_TRIALS_PER_OBJ:
                break
            ok, objs0, objs1 = settle_run(lambda objs0: c.step(
                action="TeleportObject", objectId=oid, x=objs0[oid]["position"]["x"] + dx, y=objs0[oid]["position"]["y"],
                z=objs0[oid]["position"]["z"] + dz, rotation=objs0[oid]["rotation"], forceAction=True))
            if not ok:
                continue
            ob = objs0[oid]; p0, r0 = ob["position"], ob["rotation"]
            p1, r1 = objs1[oid]["position"], objs1[oid]["rotation"]
            moved = np.array([p1["x"] - p0["x"], p1["y"] - p0["y"], p1["z"] - p0["z"]])
            rot_change = max(abs(((r1[k] - r0[k]) + 180) % 360 - 180) for k in "xyz")
            others = max((np.linalg.norm(pos(objs1[k]) - ctrl_mean[k]) - ctrl_tol[k]
                          for k in objs0 if k != oid and k in ctrl_mean
                          and (objs0[k].get("moveable") or objs0[k].get("pickupable"))), default=0.0)
            reason = None
            if abs(moved[1]) > 0.01: reason = f"height changed {moved[1]:+.3f}"
            elif np.hypot(moved[0] - dx, moved[2] - dz) > 0.03: reason = "did not reach target"
            elif rot_change > 3: reason = f"rotated {rot_change:.1f} deg"
            elif others > 0.0: reason = f"pushed another object {others:.3f} m beyond its control drift"
            evs = {v: look(v) for v in views} if reason is None else {}
            if reason is None and obj_mask(evs[views[0]], oid).sum() < MIN_VIS_PX:
                reason = "not visible after move"
            if reason:
                print(f"  reject {name} ({dx:+.2f},{dz:+.2f}): {reason}"); continue
            tname = f"{name}_{kept}"; td = out / "trials" / tname
            vis = {}
            for v in views:
                save_rgb(td / "after" / v, evs[v].frame)
                mb, ma = before_masks[v][oid], obj_mask(evs[v], oid)
                save_mask(td / "mask_before" / v, mb); save_mask(td / "mask_after" / v, ma)
                vis[v] = [int(mb.sum()), int(ma.sum())]
            aabb0 = ob["axisAlignedBoundingBox"]["cornerPoints"]
            aabb1 = objs1[oid]["axisAlignedBoundingBox"]["cornerPoints"]
            m = {"scene": args.scene, "trial": tname, "object_id": oid, "class_idx": cls, "name": name.replace("_", " "),
                 "requested_unity": [dx, 0.0, dz], "delta_gg": to_gg(moved), "delta_norm_m": float(np.linalg.norm(moved)),
                 "pos_before_gg": to_gg(p0), "pos_after_gg": to_gg(p1),
                 "aabb_before_gg": to_gg(np.array(aabb0)), "aabb_after_gg": to_gg(np.array(aabb1)),
                 "rotation_before": r0, "rotation_after": r1, "input_view": views[0],
                 "views": views, "visible_px_before_after_1080p": vis, "self_drifting_objects_m": drift}
            json.dump(m, open(td / "meta.json", "w"), indent=1)
            summary.append(m); kept += 1
            print(f"  KEEP {tname}: delta_gg={np.round(m['delta_gg'], 3).tolist()}  input-view px {vis[views[0]]}")
    c.stop()
    json.dump([{k: s[k] for k in ("trial", "name", "class_idx", "delta_gg", "delta_norm_m")} for s in summary],
              open(out / "trials.json", "w"), indent=1)
    print(f"{args.scene}: {len(summary)} trials over {len({s['object_id'] for s in summary})} objects, {len(views)} views each")


if __name__ == "__main__":
    main()
