#!/usr/bin/env python3
"""
eval_render.py — rendering metrics for every method's displacement estimate. The selected object Gaussians
(class_clean3d, ground-truth class) are translated by the estimate, every view of the scene is rendered and compared
with the real AFTER photo of that view:

    psnr, ssim              whole image
    psnr_roi, ssim_roi      inside the region of the object (dilated union of the true BEFORE and AFTER masks)
    lpips_roi               LPIPS (AlexNet) on the bounding box of that region
    iou                     visible silhouette of the moved object vs. the true AFTER mask

Reported for the input view and averaged over the other (novel) views that see the object. Reference rows: gt
(the true displacement — the best this GG model + selection can do) and no_update.

Reads eval_cues.jsonl, eval_baselines.jsonl and eval_learned_<run>.jsonl (--runs) of <bench>/<scene>.
    python scripts/learn/eval_render.py --bench ~/Documents/displacement_bench_t2 --scene FloorPlan1 --tag FP1 --runs r18crop
Writes <bench>/<scene>/eval_render.jsonl, one row per (trial, input view, method).
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import bench_lib as B

ap = argparse.ArgumentParser()
ap.add_argument("--bench", required=True)
ap.add_argument("--scene", default="FloorPlan1"); ap.add_argument("--tag", default="FP1")
ap.add_argument("--runs", nargs="*", default=["r18crop"])
ap.add_argument("--skip", default="P2,raft_orig", help="methods not rendered (hopeless, would only cost time)")
args = ap.parse_args()

HOME = Path.home()
BENCH = Path(args.bench) / args.scene
ENV = HOME / f"Documents/dataset_1080p_fixed/{args.scene}/env"
SR = HOME / "Documents/vast_results/server_root/output"
names = [v["name"] for v in json.load(open(BENCH / "views.json"))["views"]]
views = B.load_views(str(ENV), names)
gg, clf, gg_cls = B.load_gg(str(SR / f"gg_{args.tag}_fixed"))
K9 = np.ones((9, 9), np.uint8)
SKIP = set(args.skip.split(",")) | {"gate_" + r for r in args.runs}

import lpips
LP = lpips.LPIPS(net="alex", verbose=False).to("cuda").eval()


def load_estimates():
    E = {}
    files = [BENCH / "eval_cues.jsonl", BENCH / "eval_baselines.jsonl"] + [BENCH / f"eval_learned_{r}.jsonl" for r in args.runs]
    for p in files:
        if not p.exists():
            continue
        for l in open(p):
            r = json.loads(l)
            d = E.setdefault((r["trial"], r["view"]), {})
            d.update({k: v for k, v in r["est"].items() if k not in SKIP})
    return E


def lpips_box(pred, gt, roi):
    ys, xs = np.where(roi)
    y0, y1 = max(ys.min() - 16, 0), min(ys.max() + 17, roi.shape[0])
    x0, x1 = max(xs.min() - 16, 0), min(xs.max() + 17, roi.shape[1])
    if (y1 - y0) < 32 or (x1 - x0) < 32:
        return None
    with torch.no_grad():
        return float(LP(pred[None, :, y0:y1, x0:x1] * 2 - 1, gt[None, :, y0:y1, x0:x1] * 2 - 1).item())


def view_metrics(trial_dir, s, delta, after, roi, mask_after):
    out = {}
    sp = B.apply_translation(gg, s, delta)
    with torch.no_grad():
        for v in names:
            if roi[v].sum() <= 50:                                   # the object is not in this view
                continue
            img = B.render(views[v], sp)[0].clamp(0, 1)
            m = B.image_metrics(img, after[v], roi[v])
            m["lpips_roi"] = lpips_box(img, after[v], roi[v])
            if mask_after[v].sum() >= 50:
                a = (B.visible_alpha(views[v], gg, s, delta) > 0.3).cpu().numpy()
                m["iou"] = float((a & mask_after[v]).sum() / max((a | mask_after[v]).sum(), 1))
            out[v] = m
    return out


def main():
    E = load_estimates()
    out_path = BENCH / "eval_render.jsonl"
    done = set()
    if out_path.exists():
        for l in open(out_path):
            r = json.loads(l); done.add((r["trial"], r["view"], r["method"]))
    by_trial = {}
    for (t, v) in E:
        by_trial.setdefault(t, []).append(v)
    with open(out_path, "a") as f:
        for tname, in_views in sorted(by_trial.items()):
            todo = [(v, m) for v in in_views for m in list(E[(tname, v)]) + ["gt"] if (tname, v, m) not in done]
            if not todo:
                continue
            t0 = time.time()
            TD = BENCH / "trials" / tname
            meta = json.load(open(TD / "meta.json"))
            s = B.sel_clean3d(B.sel_class(gg_cls, meta["class_idx"]), gg)
            after = {v: B.to_t(B.load_img(TD / "after" / v)) for v in names}
            mb = {v: B.load_img(TD / "mask_before" / v)[..., 0] > 127 for v in names}
            ma = {v: B.load_img(TD / "mask_after" / v)[..., 0] > 127 for v in names}
            roi = {v: cv2.dilate((mb[v] | ma[v]).astype(np.uint8), K9) > 0 for v in names}
            cache = {}
            for v, m in todo:
                d = np.array(meta["delta_gg"]) if m == "gt" else np.zeros(3) if E[(tname, v)][m] is None else np.asarray(E[(tname, v)][m])
                key = tuple(np.round(d, 4))
                if key not in cache:
                    cache[key] = view_metrics(TD, s, d, after, roi, ma)
                per = cache[key]
                nov = [per[u] for u in per if u != v]
                agg = lambda L, k: (float(np.mean([x[k] for x in L if x.get(k) is not None]))
                                    if any(x.get(k) is not None for x in L) else None)
                keys = ("psnr", "ssim", "psnr_roi", "ssim_roi", "lpips_roi", "iou")
                f.write(json.dumps(dict(scene=args.scene, trial=tname, view=v, method=m,
                                        err_cm=float(np.linalg.norm(d - np.array(meta["delta_gg"])) * 100),
                                        input={k: per.get(v, {}).get(k) for k in keys},
                                        novel={k: agg(nov, k) for k in keys}, n_novel=len(nov))) + "\n")
            f.flush()
            print(f"[{tname}] {len(todo)} (view, method) rows, {len(cache)} distinct moves rendered, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
