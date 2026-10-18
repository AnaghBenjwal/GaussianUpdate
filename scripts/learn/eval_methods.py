#!/usr/bin/env python3
"""
eval_methods.py — every displacement method on a benchmark folder (T1 = displacement_bench, T2 = displacement_bench_t2),
one record per (trial, input view), so the hand-built pipelines and the learned model are scored on identical inputs.

Setting (= run_ablation.py): real BEFORE + real AFTER photo of the input view, the GG model of the BEFORE scene,
identification fixed to the ground-truth class (2-image identification was 100 % on T1), selection class_clean3d.

    stage 1 (GPU heavy, once):  --stage cues     -> <bench>/<scene>/eval_cues.jsonl
        P1        current pipeline: template-matching 2D shift + constant-depth back-projection (class selection)
        P2        current pipeline: RAFT 2D shift + constant-depth back-projection (class selection)
        raft_orig RAFT median shift + support-plane cast;  raft_crop / roma (crop matches + plane lifting + RANSAC)
        depth     Depth Anything V2 aligned to GG depth;   sam (SAM outline contact point on the base plane)
        rc        global differential render-and-compare (V2 baseline)
        agree     ablation winner sel|agree_new_est: consensus of {raft_crop, roma, depth, sam} if >= 2 agree within
                  5 cm and the consensus is > 5 cm from rc, else rc
    stage 2 (per learned model): --stage learned --preds NAME=preds.jsonl -> eval_learned_NAME.jsonl
        NAME      DispNet prediction as is
        NAME+rc   local render-and-compare (±8 cm) started at the DispNet prediction
        agree+NAME  the agree rule with DispNet as a fifth cue
        gate_NAME   DispNet unless its predicted uncertainty is high, then agree

    python scripts/learn/eval_methods.py --bench ~/Documents/displacement_bench_t2 --scene FloorPlan1 --tag FP1 --stage cues
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
import ext_models as X

ap = argparse.ArgumentParser()
ap.add_argument("--bench", required=True)
ap.add_argument("--scene", default="FloorPlan1"); ap.add_argument("--tag", default="FP1")
ap.add_argument("--stage", choices=["cues", "learned"], required=True)
ap.add_argument("--preds", nargs="*", default=[], help="NAME=predictions.jsonl (predict.py output)")
ap.add_argument("--gate_cm", type=float, default=3.0, help="gate: use DispNet if its mean Laplace scale < this")
args = ap.parse_args()

HOME = Path.home()
BENCH = Path(args.bench) / args.scene
ENV = HOME / f"Documents/dataset_1080p_fixed/{args.scene}/env"
SR = HOME / "Documents/vast_results/server_root/output"
vm = json.load(open(BENCH / "views.json")); names = [v["name"] for v in vm["views"]]
views = B.load_views(str(ENV), names)
gg, clf, gg_cls = B.load_gg(str(SR / f"gg_{args.tag}_fixed"))
K = np.ones((9, 9), np.uint8)
MIN_PX = 3.0
NEW = ["raft_crop", "roma", "depth", "sam"]


# ── per-view state (set by set_view) ────────────────────────────────────────────
V = real_before = real_before_t = D_GG = cmap = None
H = W = 0


def set_view(v):
    global V, real_before, real_before_t, D_GG, cmap, H, W
    V = views[v]
    real_before = B.load_img(BENCH / "before" / v); real_before_t = B.to_t(real_before)
    H, W = real_before.shape[:2]
    cmap = B.classmap(V, gg, clf)
    with torch.no_grad():
        d, a = B.render_depth_alpha(V, gg)
    D_GG = torch.where(a > 0.9, d, torch.full_like(d, float("nan"))).cpu().numpy()


# ── geometry helpers (= run_ablation.py) ────────────────────────────────────────
def change_mask_pair(b_u8, a_u8, thresh=12):
    d = np.abs(a_u8.astype(np.int16) - b_u8.astype(np.int16)).max(2)
    m = cv2.morphologyEx((d > thresh).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, K) > 0


def backproject(uv, z):
    pc = np.stack([(uv[:, 0] - V.W / 2) / V.fx * z, (uv[:, 1] - V.H / 2) / V.fy * z, z], 1)
    return (pc - V.t) @ V.R


def lift_plane(uv_b, uv_a):
    ui, vi = np.clip(uv_b[:, 0].round().astype(int), 0, W - 1), np.clip(uv_b[:, 1].round().astype(int), 0, H - 1)
    Xb = backproject(uv_b, D_GG[vi, ui])
    o = -V.R.T @ V.t
    dc = np.stack([(uv_a[:, 0] - V.W / 2) / V.fx, (uv_a[:, 1] - V.H / 2) / V.fy, np.ones(len(uv_a))], 1) @ V.R
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = (Xb[:, 1] - o[1]) / dc[:, 1]
    d = o + lam[:, None] * dc - Xb
    d[(lam <= 0) | (np.abs(dc[:, 1]) < 1e-3) | ~np.isfinite(d).all(1)] = np.nan
    return d


def ransac_xz(d, thr=0.02, iters=300, min_inl=8):
    rng = np.random.default_rng(0)
    d = d[np.isfinite(d).all(1)][:, [0, 2]]
    if len(d) < min_inl:
        return None
    best = None
    for i in rng.integers(0, len(d), min(iters, len(d))):
        inl = np.linalg.norm(d - d[i], axis=1) < thr
        if best is None or inl.sum() > best.sum():
            best = inl
    if best.sum() < min_inl:
        return None
    m = np.median(d[best], 0)
    return np.array([m[0], 0.0, m[1]])


def interior_point(mask):
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    v, u = np.unravel_index(dt.argmax(), dt.shape)
    return [float(u), float(v)]


def contact_px(mask):
    ys, xs = np.where(mask)
    if len(ys) < 10:
        return None
    yb = np.percentile(ys, 97)
    return np.array([np.median(xs[ys >= yb]), yb])


def plane_hit(uv, y_plane):
    o, d = V.ray(*uv)
    if abs(d[1]) < 1e-3 or (y_plane - o[1]) / d[1] <= 0:
        return None
    return o + (y_plane - o[1]) / d[1] * d


def align_inv_depth(disp, ref_depth, valid):
    m = valid & np.isfinite(ref_depth) & (ref_depth > 0.05)
    x, y = disp[m].astype(np.float64), 1.0 / ref_depth[m].astype(np.float64)
    if len(x) < 500:
        return None
    keep = np.ones(len(x), bool)
    for _ in range(3):
        A = np.stack([x[keep], np.ones(keep.sum())], 1)
        s, t = np.linalg.lstsq(A, y[keep], rcond=None)[0]
        r = np.abs(s * x + t - y); keep = r < 2.5 * np.median(r[keep]) + 1e-9
    inv = s * disp + t
    return np.where(inv > 1e-3, 1.0 / np.maximum(inv, 1e-3), np.nan).astype(np.float32)


def crop_box(mask, aspect=4 / 3, min_w=480):
    ys, xs = np.where(mask)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    w = max(min_w, (xs.max() - xs.min()) * 2.0, (ys.max() - ys.min()) * 2.0 * aspect)
    w = min(w, W); h = min(w / aspect, H); w = h * aspect
    x0 = int(np.clip(cx - w / 2, 0, W - w)); y0 = int(np.clip(cy - h / 2, 0, H - h))
    return x0, y0, int(w), int(h)


def cons_group(v, thr=0.05):
    v = np.stack(v)
    D = np.linalg.norm(v[:, None, [0, 2]] - v[None, :, [0, 2]], axis=2)
    i = max(range(len(v)), key=lambda i: ((D[i] < thr).sum(), -D[i][D[i] < thr].sum()))
    m = D[i] < thr
    return np.median(v[m], 0), int(m.sum())


def agree(est, keys, g0):
    """sel|agree_<set>_est of summarize_ablation.py."""
    v = [np.asarray(est[k]) for k in keys if est.get(k) is not None]
    if len(v) >= 2:
        c, n = cons_group(v)
        if n >= 2 and np.hypot(*(c - g0)[[0, 2]]) > 0.05:
            return c, "cons"
    return g0, "global"


def selection(c):
    return B.sel_clean3d(B.sel_class(gg_cls, c), gg)


def setup_trial(tname, v):
    TD = BENCH / "trials" / tname
    meta = json.load(open(TD / "meta.json"))
    aft = B.load_img(TD / "after" / v)
    s = selection(meta["class_idx"])
    A0 = (B.visible_alpha(V, gg, s) > 0.3).cpu().numpy()
    if A0.sum() < 20:
        A0 = cmap == meta["class_idx"]
    return meta, aft, s, A0


# ── stage 1: the hand-built cues ────────────────────────────────────────────────
def run_cues(tname, v, meta, aft, s, A0, cache):
    est, diag = {}, {}
    c = meta["class_idx"]
    chg = change_mask_pair(real_before, aft)
    newreg = chg & ~(cv2.dilate(A0.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0)
    n_lab, lab, st, _ = cv2.connectedComponentsWithStats(newreg.astype(np.uint8))
    big = [i for i in range(1, n_lab) if st[i, cv2.CC_STAT_AREA] >= 20]
    newcc = (lab == max(big, key=lambda i: st[i, cv2.CC_STAT_AREA])) if big else np.zeros_like(newreg)
    y_base = float(torch.quantile(gg.xyz[s][:, 1], 0.02))
    x0, y0, cw, ch = crop_box(chg | A0)
    cb, ca = real_before[y0:y0 + ch, x0:x0 + cw], aft[y0:y0 + ch, x0:x0 + cw]
    static = ~(cv2.dilate((chg | A0).astype(np.uint8), np.ones((31, 31), np.uint8)) > 0)
    objc = A0[y0:y0 + ch, x0:x0 + cw]
    diag.update(A0_px=int(A0.sum()), chg_px=int(chg.sum()), newcc_px=int(newcc.sum()))
    T = {}

    # P1 / P2: the current pipeline's displacement stage (class selection, constant depth)
    tt = time.time()
    s_cls = B.sel_class(gg_cls, c)
    m_cls = cmap == c
    cb2 = B.obj_centroid_2d(m_cls)
    sx, sy = B.shift_template(real_before, aft, m_cls)
    est["P1"] = B.backproject_const_depth(V, gg, s_cls, cb2, (cb2[0] + sx, cb2[1] + sy))
    both = m_cls | chg
    est["P2"] = B.backproject_const_depth(V, gg, s_cls, (0, 0), B.shift_raft(real_before, aft, both if both.sum() > 50 else m_cls))
    T["P"] = time.time() - tt

    tt = time.time()
    est["raft_orig"] = B.backproject_plane(V, gg, s, B.shift_raft(real_before, aft, A0))
    fl = X.raft_flow(cb, ca)
    vs, us = np.where(objc)
    est["raft_crop"] = None
    if len(us):
        pick = np.random.default_rng(0).choice(len(us), min(3000, len(us)), replace=False); us, vs = us[pick], vs[pick]
        uv_b = np.stack([us + x0, vs + y0], 1).astype(float); uv_a = uv_b + fl[vs, us]
        mv = np.linalg.norm(uv_a - uv_b, axis=1) > MIN_PX
        est["raft_crop"] = ransac_xz(lift_plane(uv_b[mv], uv_a[mv]))
    T["raft"] = time.time() - tt

    tt = time.time()
    ka, kb, kc = X.roma_match(cb, ca, n=8000)
    ki = np.clip(ka.round().astype(int), 0, [cw - 1, ch - 1])
    on = objc[ki[:, 1], ki[:, 0]] & (kc > 0.1) & (np.linalg.norm(kb - ka, axis=1) > MIN_PX)
    est["roma"] = ransac_xz(lift_plane(ka[on] + [x0, y0], kb[on] + [x0, y0])) if on.sum() else None
    T["roma"] = time.time() - tt

    tt = time.time()
    if "da_before" not in cache:
        cache["da_before"] = X.mono_depth(real_before)
    disp_b, disp_a = cache["da_before"], X.mono_depth(aft)
    win = np.zeros((H, W), bool); win[max(y0 - 200, 0):y0 + ch + 200, max(x0 - 200, 0):x0 + cw + 200] = True
    Zb, Za = align_inv_depth(disp_b, D_GG, static & win), align_inv_depth(disp_a, D_GG, static & win)
    est["depth"] = None
    if Zb is not None and Za is not None and newcc.sum() >= 20:
        vn, un = np.where(newcc); vo, uo = np.where(A0)
        Pn = backproject(np.stack([un, vn], 1).astype(float), Za[vn, un])
        Po = backproject(np.stack([uo, vo], 1).astype(float), Zb[vo, uo])
        Pn, Po = Pn[np.isfinite(Pn).all(1)], Po[np.isfinite(Po).all(1)]
        if len(Pn) > 10 and len(Po) > 10:
            d = np.median(Pn, 0) - np.median(Po, 0); est["depth"] = np.array([d[0], 0.0, d[2]])
    T["depth"] = time.time() - tt

    tt = time.time()
    est["sam"] = None
    if newcc.sum() >= 20:
        masks, scores = X.sam_mask(aft, [interior_point(newcc)], [interior_point(A0)])
        ok = [i for i in range(len(masks)) if (masks[i] & newcc).sum() >= 0.5 * newcc.sum()
              and masks[i].sum() < max(6 * A0.sum(), 4 * newcc.sum()) and masks[i].sum() < 0.05 * H * W]
        if ok:
            Mn = masks[max(ok, key=lambda i: scores[i])]
            pn, po = contact_px(Mn), contact_px(A0)
            if pn is not None and po is not None:
                Xn, Xo = plane_hit(pn, y_base), plane_hit(po, y_base)
                if Xn is not None and Xo is not None:
                    d = Xn - Xo; est["sam"] = np.array([d[0], 0.0, d[2]])
    T["sam"] = time.time() - tt

    tt = time.time()
    aft_t = B.to_t(aft)
    est["rc"] = B.render_and_compare(V, gg, s, aft_t, ref_t=real_before_t, init=[np.zeros(3), est["raft_orig"]])[0]
    T["rc"] = time.time() - tt
    est["agree"], diag["agree_used"] = agree(est, NEW, np.asarray(est["rc"]))
    est["no_update"] = np.zeros(3)
    return est, diag, T


def tolist(d):
    return {k: (None if v is None else np.asarray(v, float).round(5).tolist()) for k, v in d.items()}


def main():
    # every (trial, view) of the folder, grouped by view so per-view renders are done once
    jobs = {}
    for td in sorted((BENCH / "trials").iterdir()):
        m = json.load(open(td / "meta.json"))
        for v in m.get("input_views", [m["input_view"]]):
            jobs.setdefault(v, []).append(td.name)
    cues_path = BENCH / "eval_cues.jsonl"
    cues = {}
    if cues_path.exists():
        for l in open(cues_path):
            r = json.loads(l); cues[(r["trial"], r["view"])] = r
    if args.stage == "cues":
        with open(cues_path, "a") as f:
            for v, trials in sorted(jobs.items()):
                todo = [t for t in trials if (t, v) not in cues]
                if not todo:
                    continue
                set_view(v)
                cache = {}
                for tname in todo:
                    t0 = time.time()
                    meta, aft, s, A0 = setup_trial(tname, v)
                    est, diag, T = run_cues(tname, v, meta, aft, s, A0, cache)
                    r = dict(scene=args.scene, trial=tname, view=v, object=meta["name"], gt_delta=meta["delta_gg"],
                             gt_norm_m=meta["delta_norm_m"], est=tolist(est), diag=diag, timing=T)
                    f.write(json.dumps(r) + "\n"); f.flush()
                    gt = np.array(meta["delta_gg"])
                    print(f"[{tname} {v}] {time.time() - t0:.0f}s gt {meta['delta_norm_m'] * 100:.0f}cm | " + " ".join(
                        f"{k} {'–' if e is None else format(np.linalg.norm(np.asarray(e) - gt) * 100, '.1f')}"
                        for k, e in est.items()), flush=True)
        return

    for spec in args.preds:
        name, path = spec.split("=")
        preds = {}
        for l in open(path):
            p = json.loads(l)
            if p["scene"] == args.scene:
                preds[(p["trial"], p["view"])] = p
        out_path = BENCH / f"eval_learned_{name}.jsonl"
        done = {(json.loads(l)["trial"], json.loads(l)["view"]) for l in open(out_path)} if out_path.exists() else set()
        with open(out_path, "a") as f:
            for v, trials in sorted(jobs.items()):
                todo = [t for t in trials if (t, v) not in done and (t, v) in preds and (t, v) in cues]
                if not todo:
                    continue
                set_view(v)
                for tname in todo:
                    meta, aft, s, A0 = setup_trial(tname, v)
                    c, p = cues[(tname, v)], preds[(tname, v)]
                    est = {k: (None if e is None else np.array(e)) for k, e in c["est"].items()}
                    d = np.array(p["pred_gg"])
                    rc_local = B.render_and_compare(V, gg, s, B.to_t(aft), ref_t=real_before_t, init=[d], center=d,
                                                    radius=0.08, step=0.02)[0]
                    est5 = dict(est, dispnet=d)
                    ag5, used5 = agree(est5, NEW + ["dispnet"], np.asarray(est["rc"]))
                    unc_cm = float(np.mean(p["scale_grav"]) * 100)
                    out = {name: d, f"{name}+rc": rc_local, f"agree+{name}": ag5,
                           f"gate_{name}": d if unc_cm < args.gate_cm else np.asarray(est["agree"])}
                    r = dict(scene=args.scene, trial=tname, view=v, gt_delta=meta["delta_gg"], gt_norm_m=meta["delta_norm_m"],
                             est=tolist(out), unc_cm=unc_cm, agree_used=used5)
                    f.write(json.dumps(r) + "\n"); f.flush()
                    gt = np.array(meta["delta_gg"])
                    print(f"[{name} {tname} {v}] gt {meta['delta_norm_m'] * 100:.0f}cm unc {unc_cm:.1f} | " + " ".join(
                        f"{k} {np.linalg.norm(np.asarray(e) - gt) * 100:.1f}" for k, e in out.items()), flush=True)


if __name__ == "__main__":
    main()
