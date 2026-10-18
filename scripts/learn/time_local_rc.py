#!/usr/bin/env python3
"""
time_local_rc.py — runtime of the local render-and-compare (±8 cm around the DispNet prediction) against the global
rc, on the same inputs and GPU. Reads the DispNet estimates from <bench>/<scene>/eval_learned_<run>.jsonl.
    python scripts/learn/time_local_rc.py --bench ~/Documents/displacement_bench_t2 --scene FloorPlan1 --tag FP1 --n 40
Prints per-input times and the mean; appends one summary row to <bench>/<scene>/timing_local_rc.jsonl.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import bench_lib as B

ap = argparse.ArgumentParser()
ap.add_argument("--bench", required=True)
ap.add_argument("--scene", default="FloorPlan1"); ap.add_argument("--tag", default="FP1")
ap.add_argument("--run", default="r18crop")
ap.add_argument("--n", type=int, default=40)
args = ap.parse_args()

HOME = Path.home()
BENCH = Path(args.bench) / args.scene
ENV = HOME / f"Documents/dataset_1080p_fixed/{args.scene}/env"
SR = HOME / "Documents/vast_results/server_root/output"
names = [v["name"] for v in json.load(open(BENCH / "views.json"))["views"]]
views = B.load_views(str(ENV), names)
gg, clf, gg_cls = B.load_gg(str(SR / f"gg_{args.tag}_fixed"))


def timed(fn):
    torch.cuda.synchronize(); t = time.time(); out = fn(); torch.cuda.synchronize()
    return out, time.time() - t


rows = [json.loads(l) for l in open(BENCH / f"eval_learned_{args.run}.jsonl")][:args.n]
loc, glob = [], []
for i, r in enumerate(rows):
    V = views[r["view"]]
    before_t = B.to_t(B.load_img(BENCH / "before" / r["view"]))
    TD = BENCH / "trials" / r["trial"]
    meta = json.load(open(TD / "meta.json"))
    aft_t = B.to_t(B.load_img(TD / "after" / r["view"]))
    s = B.sel_clean3d(B.sel_class(gg_cls, meta["class_idx"]), gg)
    d = np.array(r["est"][args.run])
    _, tl = timed(lambda: B.render_and_compare(V, gg, s, aft_t, ref_t=before_t, init=[d], center=d, radius=0.08, step=0.02))
    _, tg = timed(lambda: B.render_and_compare(V, gg, s, aft_t, ref_t=before_t, init=[np.zeros(3)]))
    if i:                                                     # first call includes CUDA warm-up
        loc.append(tl); glob.append(tg)
    print(f"[{r['trial']} {r['view']}] local {tl:.2f}s global {tg:.2f}s", flush=True)

res = dict(scene=args.scene, run=args.run, n=len(loc), local_mean_s=float(np.mean(loc)), local_median_s=float(np.median(loc)),
           global_mean_s=float(np.mean(glob)), global_median_s=float(np.median(glob)))
print(res)
with open(BENCH / "timing_local_rc.jsonl", "a") as f:
    f.write(json.dumps(res) + "\n")
