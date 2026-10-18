#!/usr/bin/env python3
"""
run_ablation.py — displacement-stage ablation of four external cues against the original RAFT matching.

    python scripts/bench/run_ablation.py --scene FloorPlan1 --tag FP1 [--trials ...]

Setting = run_bench_v2 (real before + real after photo of the input view, GG model). Identification and selection are
held fixed (identified class from V2_2img_diffrc, which is 100 % correct; class_clean3d selection); only the
displacement stage changes. Writes ~/Documents/displacement_bench/<scene>/results_ablation.jsonl.

Estimators (each: planar delta in GG frame, or None). Matchers keep only object matches that move > MIN_PX:
  raft_orig    original: RAFT on the full (downscaled) image, median shift, object centre cast to support plane
  raft_crop    RAFT dense flow on a full-res crop + per-match plane lifting + RANSAC   (isolates crop+lift from matcher)
  roma         RoMa dense matches on the same crop + the same lifting + RANSAC           (1. matcher swap)
  depth        Depth Anything V2 on before/after, affine-aligned to GG depth on static pixels; median 3D point of the
               'appeared' region minus that of the old visible footprint                (2. monocular depth)
  sam          SAM on the after photo, prompted inside the appeared region; bottom contact pixel of new vs old outline
               cast to the object's base plane                                            (3. SAM outline / contact)
  mast3r       MASt3R pair on the crop: metric depth for both images (scaled to GG on static pixels) + descriptor
               matches on the object; per-match 3D difference + RANSAC                    (4. MASt3R)
Configs:
  est|X                 estimator X alone
  rc|global             V2 baseline: differential render-and-compare, ±50 cm grid, init {0, raft_orig}
  rc|global+depth       same + differential mono-depth loss
  rc|X                  X → local render-and-compare (±8 cm around X; falls back to rc|global if X is None)
  rc|fuse_all           coordinate-wise median of all estimators → local RC; rc|fuse_all-X leave-one-out
  rc|fuse_all+depth     fused → local RC with the depth loss
"""
from __future__ import annotations
import argparse, json, time, sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_lib as B
import ext_models as X

ap = argparse.ArgumentParser()
ap.add_argument("--scene", default="FloorPlan1"); ap.add_argument("--tag", default="FP1")
ap.add_argument("--trials", nargs="*", default=None); ap.add_argument("--out", default="results_ablation.jsonl")
args = ap.parse_args()

HOME = Path.home()
BENCH = HOME / f"Documents/displacement_bench/{args.scene}"
ENV = HOME / f"Documents/dataset_1080p_fixed/{args.scene}/env"
SR = HOME / "Documents/vast_results/server_root/output"
vm = json.load(open(BENCH / "views.json")); names = [v["name"] for v in vm["views"]]; INP = vm["input_view"]
views = B.load_views(str(ENV), names); V = views[INP]
gg, clf, gg_cls = B.load_gg(str(SR / f"gg_{args.tag}_fixed"))
cmap = B.classmap(V, gg, clf)
real_before = B.load_img(BENCH / "before" / INP); real_before_t = B.to_t(real_before)
with torch.no_grad():
    d_gg, a_gg = B.render_depth_alpha(V, gg)
    D_GG = torch.where(a_gg > 0.9, d_gg, torch.full_like(d_gg, float("nan"))).cpu().numpy()   # model depth, before
CACHE = BENCH / "ablation_cache"; CACHE.mkdir(exist_ok=True)
results_path = BENCH / args.out
done = {json.loads(l)["trial"] for l in open(results_path)} if results_path.exists() else set()
v2_ids = {}
for l in open(BENCH / "results_v2.jsonl"):
    r = json.loads(l)
    if r["config"] == "V2_2img_diffrc":
        v2_ids[r["trial"]] = r["id_pred"]
K = np.ones((9, 9), np.uint8)
ESTS = ["raft_orig", "raft_crop", "roma", "depth", "sam", "mast3r", "mast3r_depth"]
FUSE = ["raft_orig", "roma", "depth", "sam", "mast3r", "mast3r_depth"]
GROUPS = {"raft_orig": ["raft_orig"], "roma": ["roma"], "depth": ["depth"], "sam": ["sam"], "mast3r": ["mast3r", "mast3r_depth"]}
H, W = real_before.shape[:2]
MIN_PX = 3.0      # the object moved: matches that stay put are background / occluder matches, not object motion


def change_mask_pair(b_u8, a_u8, thresh=12):
    d = np.abs(a_u8.astype(np.int16) - b_u8.astype(np.int16)).max(2)
    m = cv2.morphologyEx((d > thresh).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, K) > 0


def evaluate(sp_new, after, roi):
    per = {}
    with torch.no_grad():
        for v in names:
            per[v] = B.image_metrics(B.render(views[v], sp_new)[0], B.to_t(after[v]), roi[v])
    nov = [per[v] for v in names if v != INP]
    agg = lambda L, k: float(np.mean([m[k] for m in L if k in m])) if any(k in m for m in L) else None
    return {"input": per[INP], "novel": {k: agg(nov, k) for k in ("psnr", "ssim", "psnr_roi", "ssim_roi")}}


# ── geometry helpers ────────────────────────────────────────────────────────────
def backproject(uv: np.ndarray, z: np.ndarray) -> np.ndarray:
    """pixels (N,2) + camera depth (N,) -> world points (N,3), GG/OpenCV camera of the input view."""
    pc = np.stack([(uv[:, 0] - V.W / 2) / V.fx * z, (uv[:, 1] - V.H / 2) / V.fy * z, z], 1)
    return (pc - V.t) @ V.R


def lift_plane(uv_b: np.ndarray, uv_a: np.ndarray) -> np.ndarray:
    """Per-match planar displacement: before pixel -> 3D with GG depth; the point keeps its height, so the after ray
    is cut with the horizontal plane through it. Returns (N,3) deltas (NaN where undefined)."""
    ui, vi = np.clip(uv_b[:, 0].round().astype(int), 0, W - 1), np.clip(uv_b[:, 1].round().astype(int), 0, H - 1)
    Xb = backproject(uv_b, D_GG[vi, ui])
    o = -V.R.T @ V.t
    dc = np.stack([(uv_a[:, 0] - V.W / 2) / V.fx, (uv_a[:, 1] - V.H / 2) / V.fy, np.ones(len(uv_a))], 1) @ V.R
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = (Xb[:, 1] - o[1]) / dc[:, 1]
    Xa = o + lam[:, None] * dc
    d = Xa - Xb
    d[(lam <= 0) | (np.abs(dc[:, 1]) < 1e-3) | ~np.isfinite(d).all(1)] = np.nan
    return d


def ransac_xz(d: np.ndarray, thr=0.02, iters=300, min_inl=8, rng=np.random.default_rng(0)):
    d = d[np.isfinite(d).all(1)][:, [0, 2]]
    if len(d) < min_inl:
        return None, 0
    best = None
    for i in rng.integers(0, len(d), min(iters, len(d))):
        inl = np.linalg.norm(d - d[i], axis=1) < thr
        if best is None or inl.sum() > best.sum():
            best = inl
    if best.sum() < min_inl:
        return None, int(best.sum())
    m = np.median(d[best], 0)
    return np.array([m[0], 0.0, m[1]]), int(best.sum())


def interior_point(mask: np.ndarray):
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    v, u = np.unravel_index(dt.argmax(), dt.shape)
    return [float(u), float(v)]


def contact_px(mask: np.ndarray):
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


def align_inv_depth(disp: np.ndarray, ref_depth: np.ndarray, valid: np.ndarray):
    """Fit s*disp + t ~= 1/ref_depth on `valid` (robust: 3 rounds of inlier refits). Returns depth map (NaN invalid)."""
    m = valid & np.isfinite(ref_depth) & (ref_depth > 0.05)
    x, y = disp[m].astype(np.float64), 1.0 / ref_depth[m].astype(np.float64)
    if len(x) < 500:
        return None, None
    keep = np.ones(len(x), bool)
    for _ in range(3):
        A = np.stack([x[keep], np.ones(keep.sum())], 1)
        s, t = np.linalg.lstsq(A, y[keep], rcond=None)[0]
        r = np.abs(s * x + t - y); keep = r < 2.5 * np.median(r[keep]) + 1e-9
    inv = s * disp + t
    z = np.where(inv > 1e-3, 1.0 / np.maximum(inv, 1e-3), np.nan)
    rel = float(np.median(np.abs(z[m] - ref_depth[m]) / ref_depth[m]))
    return z.astype(np.float32), rel


def crop_box(mask: np.ndarray, aspect=4 / 3, min_w=480):
    ys, xs = np.where(mask)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    w = max(min_w, (xs.max() - xs.min()) * 2.0, (ys.max() - ys.min()) * 2.0 * aspect)
    w = min(w, W); h = min(w / aspect, H); w = h * aspect
    x0 = int(np.clip(cx - w / 2, 0, W - w)); y0 = int(np.clip(cy - h / 2, 0, H - h))
    return x0, y0, int(w), int(h)


# ── main loop ───────────────────────────────────────────────────────────────────
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
    aft = after[INP]; aft_t = B.to_t(aft)
    cid = v2_ids.get(tname, gt_c)
    s = B.sel_clean3d(B.sel_class(gg_cls, cid), gg)
    gt_sel = B.sel_aabb(gg, meta["aabb_before_gg"])
    chg = change_mask_pair(real_before, aft)
    A0 = (B.visible_alpha(V, gg, s) > 0.3).cpu().numpy()
    if A0.sum() < 20:
        A0 = cmap == cid
    newreg = chg & ~(cv2.dilate(A0.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0)
    n_lab, lab, st, _ = cv2.connectedComponentsWithStats(newreg.astype(np.uint8))
    big = [i for i in range(1, n_lab) if st[i, cv2.CC_STAT_AREA] >= 20]
    newcc = (lab == max(big, key=lambda i: st[i, cv2.CC_STAT_AREA])) if big else np.zeros_like(newreg)
    y_base = float(torch.quantile(gg.xyz[s][:, 1], 0.02))
    x0, y0, cw, ch = crop_box(chg | A0)
    cb, ca = real_before[y0:y0 + ch, x0:x0 + cw], aft[y0:y0 + ch, x0:x0 + cw]
    static = ~(cv2.dilate((chg | A0).astype(np.uint8), np.ones((31, 31), np.uint8)) > 0)
    est, diag = {}, {"crop": [x0, y0, cw, ch], "newcc_px": int(newcc.sum()), "A0_px": int(A0.sum()), "chg_px": int(chg.sum())}
    timing = {}

    # 0. original RAFT
    tt = time.time()
    sh = B.shift_raft(real_before, aft, A0)
    est["raft_orig"] = B.backproject_plane(V, gg, s, sh); timing["raft_orig"] = time.time() - tt

    # objects pixels inside the crop (before), for matchers
    objc = A0[y0:y0 + ch, x0:x0 + cw]

    # 0b. RAFT dense on crop + lifting
    tt = time.time()
    fl = X.raft_flow(cb, ca)
    vs, us = np.where(objc)
    if len(us):
        pick = np.random.default_rng(0).choice(len(us), min(3000, len(us)), replace=False); us, vs = us[pick], vs[pick]
        uv_b = np.stack([us + x0, vs + y0], 1).astype(float); uv_a = uv_b + fl[vs, us]
        mv = np.linalg.norm(uv_a - uv_b, axis=1) > MIN_PX
        est["raft_crop"], diag["raft_crop_inl"] = ransac_xz(lift_plane(uv_b[mv], uv_a[mv]))
    else:
        est["raft_crop"] = None
    timing["raft_crop"] = time.time() - tt

    # 1. RoMa on the crop + lifting
    tt = time.time()
    ka, kb, kc = X.roma_match(cb, ca, n=8000)
    ki = np.clip(ka.round().astype(int), 0, [cw - 1, ch - 1])
    on = objc[ki[:, 1], ki[:, 0]] & (kc > 0.1) & (np.linalg.norm(kb - ka, axis=1) > MIN_PX)
    diag["roma_on_obj"] = int(on.sum())
    est["roma"], diag["roma_inl"] = ransac_xz(lift_plane(ka[on] + [x0, y0], kb[on] + [x0, y0])) if on.sum() else (None, 0)
    timing["roma"] = time.time() - tt

    # 2. monocular depth
    tt = time.time()
    cache_da_b = CACHE / "da_before.npy"
    if not cache_da_b.exists():
        np.save(cache_da_b, X.mono_depth(real_before))
    disp_b, disp_a = np.load(cache_da_b), X.mono_depth(aft)
    win = np.zeros((H, W), bool); wx0, wy0 = max(x0 - 200, 0), max(y0 - 200, 0); win[wy0:y0 + ch + 200, wx0:x0 + cw + 200] = True
    Zb, rel_b = align_inv_depth(disp_b, D_GG, static & win)
    Za, rel_a = align_inv_depth(disp_a, D_GG, static & win)
    diag["depth_fit_rel"] = [rel_b, rel_a]
    depth_delta = None
    est["depth"] = None
    if Zb is not None and Za is not None:
        dd = Za - Zb
        region = cv2.dilate((chg | A0).astype(np.uint8), K) > 0
        depth_delta = torch.tensor(np.where(region, dd, np.nan).astype(np.float32), device="cuda")
        if newcc.sum() >= 20:
            vn, un = np.where(newcc); vo, uo = np.where(A0)
            Pn = backproject(np.stack([un, vn], 1).astype(float), Za[vn, un])
            Po = backproject(np.stack([uo, vo], 1).astype(float), Zb[vo, uo])
            Pn, Po = Pn[np.isfinite(Pn).all(1)], Po[np.isfinite(Po).all(1)]
            if len(Pn) > 10 and len(Po) > 10:
                d = np.median(Pn, 0) - np.median(Po, 0); est["depth"] = np.array([d[0], 0.0, d[2]])
    timing["depth"] = time.time() - tt

    # 3. SAM outline + contact point
    tt = time.time()
    est["sam"] = None
    if newcc.sum() >= 20:
        masks, scores = X.sam_mask(aft, [interior_point(newcc)], [interior_point(A0)])
        ok = [i for i in range(len(masks)) if (masks[i] & newcc).sum() >= 0.5 * newcc.sum()
              and masks[i].sum() < max(6 * A0.sum(), 4 * newcc.sum()) and masks[i].sum() < 0.05 * H * W]
        if ok:
            i = max(ok, key=lambda i: scores[i]); Mn = masks[i]
            diag["sam"] = {"score": float(scores[i]), "px": int(Mn.sum())}
            pn, po = contact_px(Mn), contact_px(A0)
            if pn is not None and po is not None:
                Xn, Xo = plane_hit(pn, y_base), plane_hit(po, y_base)
                if Xn is not None and Xo is not None:
                    d = Xn - Xo; est["sam"] = np.array([d[0], 0.0, d[2]])
    timing["sam"] = time.time() - tt

    # 4. MASt3R on the crop
    tt = time.time()
    est["mast3r"] = None
    mw, mh = 512, 384
    mr = X.mast3r_pair(cb, ca, (mw, mh))
    sx, sy = cw / mw, ch / mh
    gg_small = cv2.resize(np.nan_to_num(D_GG[y0:y0 + ch, x0:x0 + cw], nan=0), (mw, mh), interpolation=cv2.INTER_NEAREST)
    st_small = cv2.resize(static[y0:y0 + ch, x0:x0 + cw].astype(np.uint8), (mw, mh), interpolation=cv2.INTER_NEAREST) > 0
    okz = st_small & (gg_small > 0.05) & (mr["z1"] > 0.01)
    if okz.sum() > 200:
        scale = float(np.median(gg_small[okz] / mr["z1"][okz])); diag["mast3r_scale"] = scale
        m1, m2 = mr["m1"], mr["m2"]
        obj_small = cv2.resize(objc.astype(np.uint8), (mw, mh), interpolation=cv2.INTER_NEAREST) > 0
        mot = np.hypot((m2[:, 0] - m1[:, 0]) * sx, (m2[:, 1] - m1[:, 1]) * sy) > MIN_PX
        conf = np.minimum(mr["c1"][m1[:, 1], m1[:, 0]], mr["c2"][m2[:, 1], m2[:, 0]]) > 1.5
        on = obj_small[m1[:, 1], m1[:, 0]] & mot & conf
        diag["mast3r_on_obj"] = int(on.sum())
        if on.sum() >= 8:
            m1, m2 = m1[on], m2[on]
            z1 = scale * mr["z1"][m1[:, 1], m1[:, 0]]; z2 = scale * mr["z2"][m2[:, 1], m2[:, 0]]
            uv1 = np.stack([m1[:, 0] * sx + x0, m1[:, 1] * sy + y0], 1); uv2 = np.stack([m2[:, 0] * sx + x0, m2[:, 1] * sy + y0], 1)
            est["mast3r"], diag["mast3r_inl"] = ransac_xz(backproject(uv2, z2) - backproject(uv1, z1), thr=0.03)
        # MASt3R as a two-view depth model only: same estimator as `depth`, with MASt3R's (scaled) depth maps
        est["mast3r_depth"] = None
        if newcc.sum() >= 20:
            Z1 = np.full((H, W), np.nan, np.float32); Z2 = Z1.copy()
            Z1[y0:y0 + ch, x0:x0 + cw] = cv2.resize(scale * mr["z1"], (cw, ch), interpolation=cv2.INTER_LINEAR)
            Z2[y0:y0 + ch, x0:x0 + cw] = cv2.resize(scale * mr["z2"], (cw, ch), interpolation=cv2.INTER_LINEAR)
            vn, un = np.where(newcc); vo, uo = np.where(A0)
            Pn = backproject(np.stack([un, vn], 1).astype(float), Z2[vn, un]); Po = backproject(np.stack([uo, vo], 1).astype(float), Z1[vo, uo])
            Pn, Po = Pn[np.isfinite(Pn).all(1)], Po[np.isfinite(Po).all(1)]
            if len(Pn) > 10 and len(Po) > 10:
                d = np.median(Pn, 0) - np.median(Po, 0); est["mast3r_depth"] = np.array([d[0], 0.0, d[2]])
    else:
        est["mast3r_depth"] = None
    timing["mast3r"] = time.time() - tt

    # ── records ─────────────────────────────────────────────────────────────────
    recs = []

    def record(cfg, d, sp_new=None, extra=None):
        d = np.zeros(3) if d is None else np.asarray(d, float)
        r = {"scene": args.scene, "trial": tname, "object": meta["name"], "class_idx": gt_c, "id_pred": cid,
             "gt_delta": gt_d.tolist(), "gt_norm_m": meta["delta_norm_m"], "config": cfg, "delta_est": d.tolist(),
             "err_cm": float(np.linalg.norm(d - gt_d) * 100), "err_planar_cm": float(np.hypot(*(d - gt_d)[[0, 2]]) * 100)}
        r.update(B.sel_prf(s, gt_sel))
        r.update(evaluate(B.apply_translation(gg, s, d) if sp_new is None else sp_new, after, roi))
        r.update(extra or {})
        recs.append(r)

    for k in ESTS:
        record(f"est|{k}", est[k], extra={"est_none": est[k] is None})
    rc = lambda **kw: B.render_and_compare(V, gg, s, aft_t, ref_t=real_before_t, **kw)[0]
    tt = time.time()
    g0 = rc(init=[np.zeros(3), est["raft_orig"]]); timing["rc_global"] = time.time() - tt
    record("rc|global", g0)
    record("rc|global+fill", g0, B.hole_fill(B.apply_translation(gg, s, g0), gg, s, V, aft))
    if depth_delta is not None:
        record("rc|global+depth", rc(init=[np.zeros(3), est["raft_orig"]], depth_delta=depth_delta))
    else:
        record("rc|global+depth", g0, extra={"fallback": True})
    local = lambda c, dd=None: rc(init=[c], center=c, radius=0.08, step=0.02, depth_delta=dd)
    for k in ESTS:
        record(f"rc|{k}", local(est[k]) if est[k] is not None else g0, extra={"fallback": est[k] is None})

    def fuse(keys):
        v = [est[k] for k in keys if est[k] is not None]
        return np.median(np.stack(v), 0) if v else None
    fa = fuse(FUSE)
    record("est|fuse_all", fa)
    fa_rc = local(fa) if fa is not None else g0
    record("rc|fuse_all", fa_rc)
    record("rc|fuse_all+fill", fa_rc, B.hole_fill(B.apply_translation(gg, s, fa_rc), gg, s, V, aft))
    for k, grp in GROUPS.items():
        f = fuse([q for q in FUSE if q not in grp])
        record(f"rc|fuse_all-{k}", local(f) if f is not None else g0)
    if fa is not None and depth_delta is not None:
        fd = local(fa, depth_delta)
    else:
        fd = fa_rc
    record("rc|fuse_all+depth", fd)
    record("rc|fuse_all+depth+fill", fd, B.hole_fill(B.apply_translation(gg, s, fd), gg, s, V, aft))
    record("no_update", np.zeros(3), gg)

    diag["est"] = {k: (None if v is None else np.asarray(v).round(4).tolist()) for k, v in est.items()}
    for r in recs:
        r["diag"] = diag; r["timing"] = timing; r["trial_seconds"] = time.time() - t0
    with open(results_path, "a") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    p = {r["config"]: r["err_cm"] for r in recs}
    print(f"[{tname}] {time.time() - t0:.0f}s gt {meta['delta_norm_m'] * 100:.0f}cm | est " +
          " ".join(f"{k} " + ("–" if est[k] is None else format(p["est|" + k], ".1f")) for k in ESTS) +
          f" | rc global {p['rc|global']:.1f} +depth {p['rc|global+depth']:.1f} fuse {p['rc|fuse_all']:.1f} fuse+depth {p['rc|fuse_all+depth']:.1f}",
          flush=True)
