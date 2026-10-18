#!/usr/bin/env python3
"""
run_bench_v2.py — second round of displacement methods (two-image setting), same trials/metrics as run_bench.py.

    python scripts/bench/run_bench_v2.py --scene FloorPlan1 --tag FP1

Setting: the robot has a real BEFORE photo and a real AFTER photo from the input view (frame_00000), plus the GG model.
run_bench.py used the model render as "before", so every change mask and render-and-compare loss also contained the
model-vs-reality appearance gap. Here:
  - change mask from the real photo pair (exact up to shadows),
  - differential render-and-compare: fit R(delta) - R(0) to after - before (cancels the appearance gap),
  - identification by differential render-and-compare over the classes touching the change mask.
Writes ~/Documents/displacement_bench/<scene>/results_v2.jsonl.
"""
from __future__ import annotations
import argparse, json, time, sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_lib as B

ap = argparse.ArgumentParser()
ap.add_argument("--scene", default="FloorPlan1"); ap.add_argument("--tag", default="FP1")
ap.add_argument("--trials", nargs="*", default=None); ap.add_argument("--out", default="results_v2.jsonl")
args = ap.parse_args()

HOME = Path.home()
BENCH = HOME / f"Documents/displacement_bench/{args.scene}"
ENV = HOME / f"Documents/dataset_1080p_fixed/{args.scene}/env"
SR = HOME / "Documents/vast_results/server_root/output"
vm = json.load(open(BENCH / "views.json")); names = [v["name"] for v in vm["views"]]; INP = vm["input_view"]
views = B.load_views(str(ENV), names); V = views[INP]
gg, clf, gg_cls = B.load_gg(str(SR / f"gg_{args.tag}_fixed"))
cmap = B.classmap(V, gg, clf)
with torch.no_grad():
    model_before = B.to_u8(B.render(V, gg)[0])
real_before = B.load_img(BENCH / "before" / INP); real_before_t = B.to_t(real_before)
results_path = BENCH / args.out
done = {json.loads(l)["trial"] for l in open(results_path)} if results_path.exists() else set()
K = np.ones((9, 9), np.uint8)


def vis_mask(sp, sel):
    with torch.no_grad():
        d_obj, a_obj = B.render_depth_alpha(V, sp.subset(sel))
        d_all, _ = B.render_depth_alpha(V, sp)
        return ((a_obj > 0.5) & (d_obj <= d_all * 1.02 + 0.02)).cpu().numpy()


def evaluate(sp_new, after, roi):
    per = {}
    with torch.no_grad():
        for v in names:
            per[v] = B.image_metrics(B.render(views[v], sp_new)[0], B.to_t(after[v]), roi[v])
    nov = [per[v] for v in names if v != INP]
    agg = lambda L, k: float(np.mean([m[k] for m in L if k in m])) if any(k in m for m in L) else None
    return {"input": per[INP], "novel": {k: agg(nov, k) for k in ("psnr", "ssim", "psnr_roi", "ssim_roi")}, "per_view": per}


def change_mask_pair(b_u8, a_u8, thresh=12):
    d = np.abs(a_u8.astype(np.int16) - b_u8.astype(np.int16)).max(2)
    m = (d > thresh).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, K) > 0


trials = sorted(p.name for p in (BENCH / "trials").iterdir())
if args.trials:
    trials = [t for t in trials if t in args.trials]
for tname in trials:
    if tname in done:
        continue
    t0 = time.time()
    TD = BENCH / "trials" / tname
    meta = json.load(open(TD / "meta.json"))
    gt_d = np.array(meta["delta_gg"]); gt_c = meta["class_idx"]
    after = {v: B.load_img(TD / "after" / v) for v in names}
    roi = {}
    for v in names:
        m = (B.load_img(TD / "mask_before" / v)[..., 0] > 127) | (B.load_img(TD / "mask_after" / v)[..., 0] > 127)
        roi[v] = cv2.dilate(m.astype(np.uint8), K) > 0
    aft_u8 = after[INP]; aft_t = B.to_t(aft_u8)
    recs = []

    def record(cfg, sp_new, id_pred=None, sel=None, gt_sel=None, delta=None, extra=None):
        r = {"scene": args.scene, "trial": tname, "object": meta["name"], "class_idx": gt_c, "gt_delta": gt_d.tolist(),
             "gt_norm_m": meta["delta_norm_m"], "config": cfg, "model": "gg", "id_pred": id_pred,
             "id_correct": None if id_pred is None else bool(id_pred == gt_c)}
        if delta is not None:
            r["delta_est"] = np.asarray(delta).tolist()
            r["err_cm"] = float(np.linalg.norm(np.asarray(delta) - gt_d) * 100)
            r["err_planar_cm"] = float(np.hypot(*(np.asarray(delta) - gt_d)[[0, 2]]) * 100)
            r["err_vertical_cm"] = float(abs((np.asarray(delta) - gt_d)[1]) * 100)
        if sel is not None and gt_sel is not None:
            r.update(B.sel_prf(sel, gt_sel))
        r.update(evaluate(sp_new, after, roi))
        r.update(extra or {})
        recs.append(r)

    gt_sel = B.sel_aabb(gg, meta["aabb_before_gg"])
    chg = change_mask_pair(real_before, aft_u8)

    # ── A. per-candidate fits on the real photo pair ───────────────────────────
    # diffrc: differential render-and-compare (whole image); sil: silhouette fit to the change mask (appearance-free);
    # drcm: differential render-and-compare with the object's old footprint excluded from the loss, initialised from
    # zero / RAFT-plane / sil. Each candidate class touching the change mask gets all three; scores feed identification.
    frac = B.class_scores_from_change(chg.astype(np.uint8), cmap)
    id_frac = max(frac, key=frac.get) if frac else -1
    ov = {int(c): int(n) for c, n in zip(*np.unique(cmap[chg], return_counts=True)) if c > 0 and n >= 30}
    cands = sorted(ov, key=ov.get, reverse=True)[:6]

    def fit_all(s, mask_fallback):
        vm_c = vis_mask(gg, s)
        sh = B.shift_raft(real_before, aft_u8, vm_c if vm_c.sum() > 20 else mask_fallback)
        plane = B.backproject_plane(V, gg, s, sh)
        d, lm, l0 = B.render_and_compare(V, gg, s, aft_t, init=[np.zeros(3), plane], ref_t=real_before_t)
        d_sil, ib, i0 = B.fit_silhouette(V, gg, s, chg, init=[np.zeros(3), plane])
        ex = cv2.dilate((B.visible_alpha(V, gg, s) > 0.3).cpu().numpy().astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        dm, lmm, l0m = B.render_and_compare(V, gg, s, aft_t, init=[np.zeros(3), plane, d_sil], ref_t=real_before_t, exclude=ex)
        return {"sel": s, "raft2img_plane": plane, "diffrc": d, "drc_gain": l0 - lm, "sil": d_sil, "sil_iou": ib,
                "sil_iou0": i0, "drcm": dm, "drcm_gain": l0m - lmm}

    fits = {}
    for c in cands:
        s = B.sel_clean3d(B.sel_class(gg_cls, c), gg)
        if s.sum() >= 10:
            fits[c] = fit_all(s, cmap == c)
    if gt_c not in fits:
        s = B.sel_clean3d(B.sel_class(gg_cls, gt_c), gg)
        if s.sum() > 0:
            fits[gt_c] = fit_all(s, cmap == gt_c)
    pool = [c for c in cands if c in fits]
    pick = lambda key: max(pool, key=lambda c: fits[c][key]) if pool else -1
    ids = {"2img_frac": id_frac, "2img_diffrc": pick("drc_gain"), "2img_drcm": pick("drcm_gain"), "2img_sil": pick("sil_iou")}

    # ── stage isolation with the oracle id ─────────────────────────────────────
    oracle = {}
    if gt_c in fits:
        oracle["class_clean3d"] = fits[gt_c]
    if gt_sel.sum() > 0:
        oracle["aabb_oracle"] = fit_all(gt_sel, cmap == gt_c)
    for sname, f in oracle.items():
        s = f["sel"]
        for dname in ("raft2img_plane", "diffrc", "sil", "drcm"):
            d = f[dname]
            record(f"oracleID|{sname}|{dname}|translate", B.apply_translation(gg, s, d), id_pred=gt_c, sel=s, gt_sel=gt_sel, delta=d)
            if dname in ("diffrc", "drcm"):
                record(f"oracleID|{sname}|{dname}|translate+fill", B.hole_fill(B.apply_translation(gg, s, d), gg, s, V, aft_u8),
                       id_pred=gt_c, sel=s, gt_sel=gt_sel, delta=d)

    # ── full pipelines ──────────────────────────────────────────────────────────
    for cfg, cid, dname in [("V1_2img_frac_diffrc", id_frac, "diffrc"), ("V2_2img_diffrc", ids["2img_diffrc"], "diffrc"),
                            ("V3_2img_drcm", ids["2img_drcm"], "drcm"), ("V4_2img_sil_drcm", ids["2img_sil"], "drcm")]:
        if cid <= 0:
            record(cfg, gg, id_pred=cid, delta=np.zeros(3)); record(cfg + "+fill", gg, id_pred=cid, delta=np.zeros(3)); continue
        f = fits.get(cid) or fit_all(B.sel_clean3d(B.sel_class(gg_cls, cid), gg), cmap == cid)
        fits[cid] = f
        s, d = f["sel"], f[dname]; mv = B.apply_translation(gg, s, d)
        record(cfg, mv, id_pred=cid, sel=s, gt_sel=gt_sel, delta=d)
        record(cfg + "+fill", B.hole_fill(mv, gg, s, V, aft_u8), id_pred=cid, sel=s, gt_sel=gt_sel, delta=d)

    diag = {"change_px": int(chg.sum()), "cands": {str(c): {"overlap": ov.get(c, 0), "drc_gain": fits[c]["drc_gain"],
            "drcm_gain": fits[c]["drcm_gain"], "sil_iou": fits[c]["sil_iou"], "sil_iou0": fits[c]["sil_iou0"]} for c in fits}}
    for r in recs:
        r["ids"] = ids; r["trial_seconds"] = time.time() - t0; r["diag"] = diag
    with open(results_path, "a") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    p = {r["config"]: r for r in recs}
    print(f"[{tname}] {time.time() - t0:.0f}s  gt {meta['delta_norm_m'] * 100:.0f}cm  chg {int(chg.sum())}px  "
          f"ids frac/drc/drcm/sil={'/'.join(str(v) for v in ids.values())} (gt {gt_c})  err cm: "
          + "  ".join(f"{k} {p[k].get('err_cm', -1):.1f}" for k in ("V1_2img_frac_diffrc", "V2_2img_diffrc", "V3_2img_drcm", "V4_2img_sil_drcm"))
          + "  oracle diffrc/sil/drcm " + "/".join(f"{p.get(f'oracleID|class_clean3d|{m}|translate', {}).get('err_cm', -1):.1f}" for m in ("diffrc", "sil", "drcm")),
          flush=True)
