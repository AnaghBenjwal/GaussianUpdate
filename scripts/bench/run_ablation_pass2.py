#!/usr/bin/env python3
"""
run_ablation_pass2.py — second pass over run_ablation.py results: new ways of combining the stored estimator outputs,
each refined with the same local render-and-compare. No external model is re-run.

    python scripts/bench/run_ablation_pass2.py --scene FloorPlan1 --tag FP1 --sets core=roma,depth,sam all=...

Configs written to results_ablation2.jsonl (same record format as run_ablation.py):
  rc|med_<set>        coordinate-wise median of the set → local RC
  rc|cons_<set>       consensus: median of the largest group of estimates agreeing within 5 cm (fallback: median)
  rc|hybrid_<set>     global RC (stored rc|global) unless it reports no motion (< 3 cm) — the change mask says the
                      object moved, so that answer is inconsistent — then rc|cons_<set>
  est|cons_<set>, est|hybrid_<set>   the same without the local RC refinement
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_lib as B

ap = argparse.ArgumentParser()
ap.add_argument("--scene", default="FloorPlan1"); ap.add_argument("--tag", default="FP1")
ap.add_argument("--sets", nargs="+", required=True, help="name=est1,est2,...")
ap.add_argument("--out", default="results_ablation2.jsonl")
args = ap.parse_args()
SETS = {s.split("=")[0]: s.split("=")[1].split(",") for s in args.sets}

HOME = Path.home()
BENCH = HOME / f"Documents/displacement_bench/{args.scene}"
ENV = HOME / f"Documents/dataset_1080p_fixed/{args.scene}/env"
SR = HOME / "Documents/vast_results/server_root/output"
vm = json.load(open(BENCH / "views.json")); names = [v["name"] for v in vm["views"]]; INP = vm["input_view"]
views = B.load_views(str(ENV), names); V = views[INP]
gg, clf, gg_cls = B.load_gg(str(SR / f"gg_{args.tag}_fixed"))
real_before_t = B.to_t(B.load_img(BENCH / "before" / INP))
K = np.ones((9, 9), np.uint8)
prev = {}
for l in open(BENCH / "results_ablation.jsonl"):
    r = json.loads(l)
    prev.setdefault(r["trial"], {})[r["config"]] = r
out_path = BENCH / args.out
done = {(json.loads(l)["trial"], json.loads(l)["config"]) for l in open(out_path)} if out_path.exists() else set()


def evaluate(sp_new, after, roi):
    per = {}
    with torch.no_grad():
        for v in names:
            per[v] = B.image_metrics(B.render(views[v], sp_new)[0], B.to_t(after[v]), roi[v])
    nov = [per[v] for v in names if v != INP]
    agg = lambda L, k: float(np.mean([m[k] for m in L if k in m])) if any(k in m for m in L) else None
    return {"input": per[INP], "novel": {k: agg(nov, k) for k in ("psnr", "ssim", "psnr_roi", "ssim_roi")}}


def consensus(v, thr=0.05):
    v = np.stack(v)
    if len(v) < 3:
        return np.median(v, 0)
    D = np.linalg.norm(v[:, None, [0, 2]] - v[None, :, [0, 2]], axis=2)
    grp = max(range(len(v)), key=lambda i: ((D[i] < thr).sum(), -D[i][D[i] < thr].sum()))
    m = D[grp] < thr
    return np.median(v[m], 0) if m.sum() >= 2 else np.median(v, 0)


for tname, cfgs in sorted(prev.items()):
    t0 = time.time()
    base = cfgs["rc|global"]
    est = {k: (None if v is None else np.array(v)) for k, v in base["diag"]["est"].items()}
    TD = BENCH / "trials" / tname
    meta = json.load(open(TD / "meta.json"))
    gt_d = np.array(meta["delta_gg"])
    after = {v: B.load_img(TD / "after" / v) for v in names}
    roi = {}
    for v in names:
        m = (B.load_img(TD / "mask_before" / v)[..., 0] > 127) | (B.load_img(TD / "mask_after" / v)[..., 0] > 127)
        roi[v] = cv2.dilate(m.astype(np.uint8), K) > 0
    aft_t = B.to_t(after[INP])
    s = B.sel_clean3d(B.sel_class(gg_cls, base["id_pred"]), gg)
    g0 = np.array(base["delta_est"])
    local = lambda c: B.render_and_compare(V, gg, s, aft_t, ref_t=real_before_t, init=[c], center=c, radius=0.08, step=0.02)[0]
    recs, line = [], []

    def record(cfg, d, extra=None):
        r = {k: base[k] for k in ("scene", "trial", "object", "class_idx", "id_pred", "gt_delta", "gt_norm_m")}
        r.update(config=cfg, delta_est=np.asarray(d).tolist(), err_cm=float(np.linalg.norm(d - gt_d) * 100),
                 err_planar_cm=float(np.hypot(*(d - gt_d)[[0, 2]]) * 100))
        r.update(B.sel_prf(s, B.sel_aabb(gg, meta["aabb_before_gg"])))
        r.update(evaluate(B.apply_translation(gg, s, d), after, roi))
        r.update(extra or {})
        recs.append(r); line.append(f"{cfg.split('|')[1]} {r['err_cm']:.1f}")

    for name, keys in SETS.items():
        v = [est[k] for k in keys if est.get(k) is not None]
        if (tname, f"rc|med_{name}") not in done:
            record(f"rc|med_{name}", local(np.median(np.stack(v), 0)) if v else g0, {"fallback": not v})
        moved = np.hypot(g0[0], g0[2]) >= 0.03
        if {(tname, f"rc|cons_{name}"), (tname, f"rc|hybrid_{name}")} - done:
            cons = local(consensus(v)) if v else g0
            if (tname, f"rc|cons_{name}") not in done:
                record(f"rc|cons_{name}", cons, {"fallback": not v})
            if (tname, f"rc|hybrid_{name}") not in done:
                record(f"rc|hybrid_{name}", g0 if moved else cons, {"used": "global" if moved else "consensus"})
        # no refinement: render-and-compare pulls small / transparent objects back towards zero motion
        ce = consensus(v) if v else g0
        if (tname, f"est|cons_{name}") not in done:
            record(f"est|cons_{name}", ce, {"fallback": not v})
        if (tname, f"est|hybrid_{name}") not in done:
            record(f"est|hybrid_{name}", g0 if moved else ce, {"used": "global" if moved else "consensus"})
    with open(out_path, "a") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"[{tname}] {time.time() - t0:.0f}s gt {meta['delta_norm_m'] * 100:.0f}cm | global {base['err_cm']:.1f} | " + "  ".join(line), flush=True)
