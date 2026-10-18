#!/usr/bin/env python3
"""
export_bench.py — convert benchmark trials (T1 = displacement_bench, T2 = displacement_bench_t2) into the
sample_io.py format, one sample per (trial, input view), so the learned model sees test data in the same form as
training data.

  --variant sim   BEFORE depth and object mask from the simulator (no GG involved; upper bound / gap study)
  --variant gg    BEFORE depth and object mask from the GG model (what the real pipeline has): needs the
                  gaussian_grouping env + GG checkpoints; rendering is done by scripts/learn/gg_inputs.py,
                  which writes gg_depth/<trial>/<view>.png and gg_mask/<trial>/<view>.png next to the trials.

    python scripts/learn/export_bench.py --bench ~/Documents/displacement_bench --split test_t1 --variant sim
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sample_io as io


def load_rgb(p):
    return np.asarray(Image.open(p).convert("RGB"))


def load_mask(p):
    return np.asarray(Image.open(p).convert("L")) > 127


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--variant", default="sim", choices=["sim", "gg"])
    ap.add_argument("--scenes", default="FloorPlan1,FloorPlan3")
    ap.add_argument("--out", default=str(Path.home() / "Documents/displacement_learn"))
    args = ap.parse_args()
    n = 0
    for scene in args.scenes.split(","):
        B = Path(args.bench) / scene
        poses = {p["name"]: p for p in json.load(open(B / "views.json"))["views"]}
        for td in sorted((B / "trials").iterdir()):
            m = json.load(open(td / "meta.json"))
            S = io.S_GG
            obj = dict(objectId=m["object_id"], objectType=m["object_id"].split("|")[0],
                       position=dict(zip("xyz", (S @ np.array(m["pos_before_gg"])).tolist())))
            for v in m.get("input_views", [m["input_view"]]):
                p = poses[v]
                before = load_rgb(B / "before" / v)
                after = load_rgb(td / "after" / v)
                mb, ma = load_mask(td / "mask_before" / v), load_mask(td / "mask_after" / v)
                if args.variant == "sim":
                    depth = np.asarray(Image.open(B / "before_depth" / v)).astype(np.float32) / 1000.0
                    mask_in = mb
                else:
                    depth = np.asarray(Image.open(B / "gg_depth" / td.name / v)).astype(np.float32) / 1000.0
                    mask_in = load_mask(B / "gg_mask" / td.name / v)
                cam = dict(pos_unity=p["cam_pos"], yaw=p["yaw"], horizon=p["horizon"], fov=90.0)
                sid = f"{scene}_{td.name}_{v[:-4]}"
                meta = io.make_meta(scene, sid, args.split, obj, cam, S @ np.array(m["delta_gg"]), dict(
                    trial=td.name, view=v, variant=args.variant, object_name=m["name"], class_idx=m["class_idx"],
                    mask_gt_px_before=int(mb.sum())))
                # the context box always comes from the GT masks + change so both variants crop identically
                box = io.context_box(mb, ma, io.change_mask(before, after))
                io.write_sample(Path(args.out) / f"{args.split}_{args.variant}" / scene / sid, before, after, depth,
                                mask_in, ma, meta, box=box)
                n += 1
    print(f"{args.split}/{args.variant}: {n} samples")


if __name__ == "__main__":
    main()
