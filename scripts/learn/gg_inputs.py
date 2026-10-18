#!/usr/bin/env python3
"""
gg_inputs.py — what the real pipeline can give the learned model at test time, rendered from the GG model
(gaussian_grouping env): for every (trial, input view) of a benchmark folder

    gg_depth/<trial>/<view>.png   alpha-weighted GG depth of the BEFORE scene, uint16 mm, 0 where alpha < 0.9
    gg_mask/<trial>/<view>.png    visible alpha of the selected object Gaussians (> 0.3), selection = class_clean3d

Identification: the moved object's class comes from --ids (results_v2.jsonl of the benchmark: V2_2img_diffrc id_pred,
100% correct on T1) when given, else the ground-truth class (2-image identification was 100% on T1 too).

    python scripts/learn/gg_inputs.py --bench ~/Documents/displacement_bench --scene FloorPlan1 --tag FP1
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
import bench_lib as B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--scene", default="FloorPlan1")
    ap.add_argument("--tag", default="FP1")
    ap.add_argument("--env", default=str(Path.home() / "Documents/dataset_1080p_fixed"))
    ap.add_argument("--gg", default=str(Path.home() / "Documents/vast_results/server_root/output"))
    ap.add_argument("--ids", default=None, help="results jsonl with id_pred per trial (config V2_2img_diffrc)")
    args = ap.parse_args()
    bench = Path(args.bench) / args.scene
    names = [v["name"] for v in json.load(open(bench / "views.json"))["views"]]
    views = B.load_views(f"{args.env}/{args.scene}/env", names)
    gg, clf, cls = B.load_gg(f"{args.gg}/gg_{args.tag}_fixed")
    ids = {}
    if args.ids:
        for l in open(args.ids):
            r = json.loads(l)
            if r.get("config") == "V2_2img_diffrc":
                ids[r["trial"]] = r["id_pred"]
    with torch.no_grad():
        full = {v: B.render_depth_alpha(views[v], gg) for v in names}
    n = 0
    for td in sorted((bench / "trials").iterdir()):
        m = json.load(open(td / "meta.json"))
        c = ids.get(td.name, m["class_idx"])
        sel = B.sel_clean3d(B.sel_class(cls, c), gg)
        for v in m.get("input_views", [m["input_view"]]):
            d, a = full[v]
            dm = torch.where(a > 0.9, d, torch.zeros_like(d)).cpu().numpy()
            A0 = (B.visible_alpha(views[v], gg, sel) > 0.3).cpu().numpy()
            for sub, img in (("gg_depth", np.clip(dm * 1000, 0, 65535).astype(np.uint16)),
                             ("gg_mask", A0.astype(np.uint8) * 255)):
                (bench / sub / td.name).mkdir(parents=True, exist_ok=True)
                Image.fromarray(img).save(bench / sub / td.name / v)
            n += 1
    print(f"{args.scene}: {n} (trial, view) inputs")


if __name__ == "__main__":
    main()
