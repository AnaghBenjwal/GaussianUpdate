#!/usr/bin/env python3
"""
run_bench.py — object-displacement benchmark on the fixed AI2THOR scenes (gaussian_grouping env).

    conda run -n gaussian_grouping python scripts/bench/run_bench.py --scene FloorPlan1 --tag FP1

Input is ONE after-image (the input view, frame_00000). Every configuration is scored on the input view and on the
held-out novel views (the GG/LangSplat --eval test cameras), so a displacement that only looks right in 2D is exposed.
Writes ~/Documents/displacement_bench/<scene>/results.jsonl (one record per trial x configuration) and qual/*.jpg.
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
ap.add_argument("--trials", nargs="*", default=None); ap.add_argument("--no_langsplat", action="store_true")
args = ap.parse_args()

HOME = Path.home()
BENCH = HOME / f"Documents/displacement_bench/{args.scene}"
ENV = HOME / f"Documents/dataset_1080p_fixed/{args.scene}/env"
SR = HOME / "Documents/vast_results/server_root/output"
vm = json.load(open(BENCH / "views.json")); names = [v["name"] for v in vm["views"]]; INP = vm["input_view"]
views = B.load_views(str(ENV), names); V = views[INP]
gg, clf, gg_cls = B.load_gg(str(SR / f"gg_{args.tag}_fixed"))
print(f"GG: {gg.n:,} Gaussians")
use_ls = not args.no_langsplat and (BENCH / "langsplat_relevancy.npz").exists()
if use_ls:
    ls = B.load_langsplat(str(SR / f"langsplat_package/{args.tag}/level_3/chkpnt30000.pth"))
    rel = np.load(BENCH / "langsplat_relevancy.npz")
    print(f"LangSplat: {ls.n:,} Gaussians")

with torch.no_grad():
    static = {"gg": {v: B.render(views[v], gg)[0] for v in names}}
    if use_ls:
        static["ls"] = {v: B.render(views[v], ls)[0] for v in names}
before = {k: B.to_u8(s[INP]) for k, s in static.items()}           # the pipeline's "before" = model render at input view
cmap = B.classmap(V, gg, clf)
results_path = BENCH / "results.jsonl"
done = set()
if results_path.exists():
    done = {json.loads(l)["trial"] for l in open(results_path)}
(BENCH / "qual").mkdir(exist_ok=True)
K = np.ones((9, 9), np.uint8)


def vis_mask(sp, sel):
    """pixels of the input view where the selected Gaussians are visible (not occluded by the rest of the scene)."""
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

    def record(cfg, sp_new, model="gg", id_pred=None, sel=None, gt_sel=None, delta=None, extra=None):
        r = {"scene": args.scene, "trial": tname, "object": meta["name"], "class_idx": gt_c, "gt_delta": gt_d.tolist(),
             "gt_norm_m": meta["delta_norm_m"], "config": cfg, "model": model, "id_pred": id_pred,
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

    # ── no update (lower bound) ─────────────────────────────────────────────────
    record("no_update", gg)

    # ── A. identification ─────────────────────────────────────────────────────
    id_pd, chg_pd = B.identify_pixeldiff(before["gg"], aft_u8, cmap)
    id_dn, chg_dn = B.identify_dino(before["gg"], aft_u8, cmap)
    cands = [int(c) for c, n in zip(*np.unique(cmap, return_counts=True)) if c > 0 and n >= 200]
    rc_all = {}
    for c in cands:
        s = B.sel_clean3d(B.sel_class(gg_cls, c), gg)
        if s.sum() < 10:
            continue
        vm_c = vis_mask(gg, s)
        sh = B.shift_raft(before["gg"], aft_u8, vm_c if vm_c.sum() > 20 else cmap == c)
        init = [np.zeros(3), B.backproject_plane(V, gg, s, sh)]
        d, lm, ls0 = B.render_and_compare(V, gg, s, aft_t, init=init)
        rc_all[c] = (d, lm, ls0, s)
    id_rc = max(rc_all, key=lambda c: (rc_all[c][2] - rc_all[c][1]) / rc_all[c][2]) if rc_all else -1
    ids = {"pixeldiff": id_pd, "dino": id_dn, "render_compare": id_rc}

    # ── B x C with the oracle object id: stage isolation ───────────────────────
    gt_sel = B.sel_aabb(gg, meta["aabb_before_gg"])
    base = B.sel_class(gg_cls, gt_c)
    sels = {"class": base, "class_depthfilter": B.sel_depth_filter(base, gg, V),
            "class_clean3d": rc_all[gt_c][3] if gt_c in rc_all else B.sel_clean3d(base, gg), "aabb_oracle": gt_sel}
    cm_gt = cmap == gt_c
    sh_tm = B.shift_template(before["gg"], aft_u8, cm_gt)
    both = cm_gt & (chg_dn > 0)
    sh_raft_cls = B.shift_raft(before["gg"], aft_u8, both if both.sum() > 50 else cm_gt)
    cb = B.obj_centroid_2d(cm_gt)
    for sname, s in sels.items():
        if s.sum() == 0:
            continue
        vm_s = vis_mask(gg, s)
        sh_vis = B.shift_raft(before["gg"], aft_u8, vm_s if vm_s.sum() > 20 else cm_gt)
        disps = {"gt": gt_d,
                 "template_constdepth": B.backproject_const_depth(V, gg, s, cb, (cb[0] + sh_tm[0], cb[1] + sh_tm[1])),
                 "raft_constdepth": B.backproject_const_depth(V, gg, s, (0, 0), sh_raft_cls),
                 "raft_plane": B.backproject_plane(V, gg, s, sh_vis)}
        if sname == "class_clean3d" and gt_c in rc_all:
            disps["render_compare"] = rc_all[gt_c][0]
        elif sname in ("class", "aabb_oracle"):
            disps["render_compare"] = B.render_and_compare(V, gg, s, aft_t, init=[np.zeros(3), disps["raft_plane"]])[0]
        for dname, d in disps.items():
            record(f"oracleID|{sname}|{dname}|translate", B.apply_translation(gg, s, d), id_pred=gt_c, sel=s, gt_sel=gt_sel, delta=d)
    for sname in ("class_clean3d", "aabb_oracle"):
        s = sels[sname]; mv = B.apply_translation(gg, s, gt_d)
        record(f"oracleID|{sname}|gt|translate+fill", B.hole_fill(mv, gg, s, V, aft_u8), id_pred=gt_c, sel=s, gt_sel=gt_sel, delta=gt_d)

    # ── full pipelines (no oracle) ─────────────────────────────────────────────
    def pipe(cfg, cid, sel_fn, disp_fn, fill=False):
        if cid <= 0:
            record(cfg, gg, id_pred=cid, delta=np.zeros(3)); return
        s = sel_fn(cid)
        d = disp_fn(cid, s)
        mv = B.apply_translation(gg, s, d)
        record(cfg, B.hole_fill(mv, gg, s, V, aft_u8) if fill else mv, id_pred=cid, sel=s, gt_sel=gt_sel, delta=d)

    def tm_const(c, s):
        m = cmap == c; cb_ = B.obj_centroid_2d(m); sx, sy = B.shift_template(before["gg"], aft_u8, m)
        return B.backproject_const_depth(V, gg, s, cb_, (cb_[0] + sx, cb_[1] + sy))

    def raft_const(c, s):
        m = cmap == c; mm = m & (chg_dn > 0)
        return B.backproject_const_depth(V, gg, s, (0, 0), B.shift_raft(before["gg"], aft_u8, mm if mm.sum() > 50 else m))

    def raft_plane(c, s):
        vm_ = vis_mask(gg, s)
        return B.backproject_plane(V, gg, s, B.shift_raft(before["gg"], aft_u8, vm_ if vm_.sum() > 20 else cmap == c))

    clean = lambda c: rc_all[c][3] if c in rc_all else B.sel_clean3d(B.sel_class(gg_cls, c), gg)
    rc_d = lambda c, s: rc_all[c][0] if c in rc_all else B.render_and_compare(V, gg, s, aft_t)[0]
    pipe("P1_current_pixeldiff_template", id_pd, lambda c: B.sel_class(gg_cls, c), tm_const)
    pipe("P2_current_dino_raft", id_dn, lambda c: B.sel_class(gg_cls, c), raft_const)
    pipe("P3_dino_clean3d_raft_plane", id_dn, clean, raft_plane)
    pipe("P4_render_compare", id_rc, clean, rc_d)
    pipe("P5_render_compare+fill", id_rc, clean, rc_d, fill=True)

    # ── LangSplat baseline (oracle text query = object name; its own geometry) ─────────
    if use_ls:
        record("LS|no_update", ls, model="ls")
        ls_gt = B.sel_aabb(ls, meta["aabb_before_gg"])
        r3 = torch.from_numpy(rel[f"L3_{gt_c}"].astype(np.float32)).to(B.DEV)
        ls_sels = {"rel>0.5": r3 > 0.5}
        ls_sels["rel>0.5_clean3d"] = B.sel_clean3d(ls_sels["rel>0.5"], ls)
        ls_sels["aabb_oracle"] = ls_gt
        ls_before = before["ls"]
        for sname, s in ls_sels.items():
            if s.sum() == 0:
                record(f"LS|{sname}|empty", ls, model="ls", id_pred=gt_c, sel=s, gt_sel=ls_gt, delta=np.zeros(3)); continue
            disps = {"gt": gt_d}
            if sname != "aabb_oracle":
                vm_ = vis_mask(ls, s)
                if vm_.sum() > 20:
                    disps["raft_plane"] = B.backproject_plane(V, ls, s, B.shift_raft(ls_before, aft_u8, vm_))
                if sname == "rel>0.5_clean3d":
                    disps["render_compare"] = B.render_and_compare(V, ls, s, aft_t, init=[np.zeros(3), disps.get("raft_plane", np.zeros(3))])[0]
            for dname, d in disps.items():
                record(f"LS|{sname}|{dname}|translate", B.apply_translation(ls, s, d), model="ls", id_pred=gt_c, sel=s, gt_sel=ls_gt, delta=d)

    for r in recs:
        r["ids"] = ids; r["trial_seconds"] = time.time() - t0
    with open(results_path, "a") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    # ── qualitative panel: input view + the novel view where the object is largest ─────
    def tile(img, label):
        im = cv2.resize(img, (400, 225), interpolation=cv2.INTER_AREA).copy()
        cv2.rectangle(im, (0, 0), (400, 22), (0, 0, 0), -1)
        cv2.putText(im, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return im
    nov = max((v for v in names if v != INP), key=lambda v: roi[v].sum())
    show = ["no_update", "P1_current_pixeldiff_template", "P2_current_dino_raft", "P3_dino_clean3d_raft_plane",
            "P5_render_compare+fill", "oracleID|aabb_oracle|gt|translate+fill"]
    rows = []
    for v in (INP, nov):
        cells = [tile(after[v], f"GT after ({'input' if v == INP else 'novel'} view)")]
        for cfg in show:
            r = next(x for x in recs if x["config"] == cfg)
            sp_new = gg  # re-render (cheap) to build the panel
            if "delta_est" in r and r.get("sel_n", 0) > 0:
                cid = r["id_pred"]
                s = {"P1_current_pixeldiff_template": lambda: B.sel_class(gg_cls, cid), "P2_current_dino_raft": lambda: B.sel_class(gg_cls, cid),
                     "P3_dino_clean3d_raft_plane": lambda: clean(cid), "P5_render_compare+fill": lambda: clean(cid),
                     "oracleID|aabb_oracle|gt|translate+fill": lambda: gt_sel}[cfg]()
                sp_new = B.apply_translation(gg, s, np.array(r["delta_est"]))
                if "fill" in cfg:
                    sp_new = B.hole_fill(sp_new, gg, s, V, aft_u8)
            with torch.no_grad():
                img = B.to_u8(B.render(views[v], sp_new)[0])
            m = r["per_view"][v]
            cells.append(tile(img, f"{cfg.split('|')[0][:22]} {m.get('psnr_roi', float('nan')):.1f}dB roi"))
        rows.append(np.concatenate(cells, 1))
    cv2.imwrite(str(BENCH / "qual" / f"{tname}.jpg"), cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    gtn = meta["delta_norm_m"] * 100
    p = {r["config"]: r for r in recs}
    print(f"[{tname}] {time.time() - t0:.0f}s  gt {gtn:.0f}cm  ids pd/dino/rc={id_pd}/{id_dn}/{id_rc} (gt {gt_c})  "
          f"err cm: P1 {p['P1_current_pixeldiff_template'].get('err_cm', -1):.1f}  P2 {p['P2_current_dino_raft'].get('err_cm', -1):.1f}  "
          f"P3 {p['P3_dino_clean3d_raft_plane'].get('err_cm', -1):.1f}  P4 {p['P4_render_compare'].get('err_cm', -1):.1f}", flush=True)
