#!/usr/bin/env python3
"""
eval_baselines.py — existing methods from the literature, adapted to our task (one BEFORE + one AFTER photo of the
same view + the GG model of the BEFORE scene -> 3D displacement of the moved object), on the same (trial, view)
inputs as eval_methods.py. Identification = ground-truth class, selection = class_clean3d, as for every other method.

    inerf        iNeRF (Yen-Chen et al., IROS 2021): gradient descent on the object's translation through the
                 differentiable GG renderer, photometric loss to the AFTER photo (ROI = change ∪ object), init 0
    lg_pnp       DISK + LightGlue matches (Lindenberger et al., ICCV 2023) BEFORE↔AFTER on the object, BEFORE points
                 lifted with GG depth, PnP-RANSAC for the object's rigid motion — the pose step of 3DGS-CD
                 (Lu et al., RA-L 2025) / Gen6D / OnePose-style model-based pose estimation
    roma_pnp     the same with RoMa dense matches (Edstedt et al., CVPR 2024)
    cotracker    CoTracker3 (Karaev et al., 2024) point tracks BEFORE→AFTER on the object, support-plane lifting + RANSAC
    icp          point-to-point ICP (Besl & McKay 1992): object points BEFORE (GG depth) vs. AFTER (SAM mask on the AFTER
                 photo + Depth Anything V2 depth aligned to GG on static pixels)
    mast3r       MASt3R (Leroy et al., ECCV 2024) pair on the crop: metric depth scaled to GG + reciprocal descriptor
                 matches on the object, 3D
                 differences + RANSAC (only with --mast3r; needs /root/mast3r)

    python scripts/learn/eval_baselines.py --bench ~/Documents/displacement_bench_t2 --scene FloorPlan1 --tag FP1
Writes <bench>/<scene>/eval_baselines.jsonl (same record format as eval_cues.jsonl).
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import bench_lib as B
import ext_models as X

ap = argparse.ArgumentParser()
ap.add_argument("--bench", required=True)
ap.add_argument("--scene", default="FloorPlan1"); ap.add_argument("--tag", default="FP1")
ap.add_argument("--methods", default="inerf,lg_pnp,roma_pnp,cotracker,icp")
ap.add_argument("--mast3r", action="store_true")
args = ap.parse_args()
METHODS = args.methods.split(",") + (["mast3r"] if args.mast3r else [])

HOME = Path.home()
BENCH = Path(args.bench) / args.scene
ENV = HOME / f"Documents/dataset_1080p_fixed/{args.scene}/env"
SR = HOME / "Documents/vast_results/server_root/output"
vm = json.load(open(BENCH / "views.json")); names = [v["name"] for v in vm["views"]]
views = B.load_views(str(ENV), names)
gg, clf, gg_cls = B.load_gg(str(SR / f"gg_{args.tag}_fixed"))
DEV = "cuda"
K9 = np.ones((9, 9), np.uint8)
MIN_PX = 3.0
V = real_before = D_GG = None
H = W = 0


def set_view(v):
    global V, real_before, D_GG, H, W
    V = views[v]
    real_before = B.load_img(BENCH / "before" / v)
    H, W = real_before.shape[:2]
    with torch.no_grad():
        d, a = B.render_depth_alpha(V, gg)
    D_GG = torch.where(a > 0.9, d, torch.full_like(d, float("nan"))).cpu().numpy()


# ── shared geometry (as in eval_methods.py) ─────────────────────────────────────
def change_mask_pair(b_u8, a_u8, thresh=12):
    d = np.abs(a_u8.astype(np.int16) - b_u8.astype(np.int16)).max(2)
    m = cv2.morphologyEx((d > thresh).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, K9) > 0


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


def crop_box(mask, aspect=4 / 3, min_w=480):
    ys, xs = np.where(mask)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    w = max(min_w, (xs.max() - xs.min()) * 2.0, (ys.max() - ys.min()) * 2.0 * aspect)
    w = min(w, W); h = min(w / aspect, H); w = h * aspect
    x0 = int(np.clip(cx - w / 2, 0, W - w)); y0 = int(np.clip(cy - h / 2, 0, H - h))
    return x0, y0, int(w), int(h)


def interior_point(mask):
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    v, u = np.unravel_index(dt.argmax(), dt.shape)
    return [float(u), float(v)]


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


# ── the baselines ───────────────────────────────────────────────────────────────
def inerf(s, aft, roi, iters=300, lr=0.005, scale=0.5):
    Vs = V.scaled(scale)
    tgt = F.interpolate(B.to_t(aft)[None], size=(Vs.H, Vs.W), mode="area")[0]
    w = torch.from_numpy(cv2.resize(roi.astype(np.uint8), (Vs.W, Vs.H), interpolation=cv2.INTER_NEAREST) > 0).to(DEV).float()
    sf = s.float()[:, None]
    d = torch.zeros(3, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([d], lr=lr)
    for _ in range(iters):
        img = B.render(Vs, gg, gg.xyz + sf * d)[0]
        loss = (((img - tgt) ** 2).sum(0) * w).sum() / w.sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); opt.step()
    return d.detach().cpu().numpy().astype(float)


def pnp_motion(Xw, uv_a, centre):
    """Object rigid motion from 3D object points (world, BEFORE) and their pixels in the AFTER photo; the camera is
    fixed, so PnP gives camera∘motion. Starts at 'no motion'. Returns the displacement of the object centre."""
    if len(Xw) < 6:
        return None
    Kc = np.array([[V.fx, 0, V.W / 2], [0, V.fy, V.H / 2], [0, 0, 1]], np.float64)
    rv0, _ = cv2.Rodrigues(V.R.astype(np.float64))
    ok, rv, tv, inl = cv2.solvePnPRansac(Xw.astype(np.float64), uv_a.astype(np.float64), Kc, None, rvec=rv0.copy(),
                                         tvec=V.t.astype(np.float64).reshape(3, 1).copy(), useExtrinsicGuess=True,
                                         iterationsCount=2000, reprojectionError=3.0, confidence=0.999,
                                         flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or inl is None or len(inl) < 6:
        return None
    Rp, _ = cv2.Rodrigues(rv)
    Rm = V.R.T @ Rp
    tm = V.R.T @ (tv.ravel() - V.t)
    return Rm @ centre + tm - centre


def lightglue_matches(a, b):
    import kornia.feature as KF

    def make():
        return KF.DISK.from_pretrained("depth").to(DEV).eval(), KF.LightGlueMatcher("disk").to(DEV).eval()
    disk, lg = X._get("disk_lg", make)
    t = lambda x: torch.from_numpy(x).permute(2, 0, 1).float().div(255)[None].to(DEV)
    with torch.inference_mode():
        f0, f1 = disk(torch.cat([t(a), t(b)]), 2048, pad_if_not_divisible=True)
        k0, k1 = f0.keypoints, f1.keypoints
        laf = lambda k: KF.laf_from_center_scale_ori(k[None], torch.ones(1, len(k), 1, 1, device=DEV))
        _, idx = lg(f0.descriptors, f1.descriptors, laf(k0), laf(k1), hw1=a.shape[:2], hw2=b.shape[:2])
    idx = idx.cpu().numpy()
    return k0.cpu().numpy()[idx[:, 0]], k1.cpu().numpy()[idx[:, 1]]


def cotracker_tracks(a, b, pts):
    def make():
        return torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(DEV).eval()
    model = X._get("cotracker", make)
    frames = [a] * 4 + [b] * 4                                 # a still camera: the object jumps between frames 3 and 4
    video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)[None].float().to(DEV)
    q = torch.from_numpy(np.concatenate([np.zeros((len(pts), 1)), pts], 1)).float()[None].to(DEV)
    with torch.no_grad():
        tr, vis = model(video, queries=q)
    return tr[0, -1].cpu().numpy(), vis[0, -1].cpu().numpy() > 0.5


def icp(src, dst, iters=60, thr=0.05):
    """Point-to-point ICP, init = centroid alignment. Returns (R, t) mapping src onto dst."""
    tree = cKDTree(dst)
    R, t = np.eye(3), dst.mean(0) - src.mean(0)
    for _ in range(iters):
        p = src @ R.T + t
        dd, ii = tree.query(p)
        m = dd < thr
        if m.sum() < 10:
            break
        a, b = src[m], dst[ii[m]]
        ca, cb = a.mean(0), b.mean(0)
        U, _, Vt = np.linalg.svd((a - ca).T @ (b - cb))
        S = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
        R = Vt.T @ S @ U.T; t = cb - R @ ca
    return R, t


def run(tname, v, cache):
    TD = BENCH / "trials" / tname
    meta = json.load(open(TD / "meta.json"))
    aft = B.load_img(TD / "after" / v)
    s = B.sel_clean3d(B.sel_class(gg_cls, meta["class_idx"]), gg)
    if "cmap" not in cache:
        cache["cmap"] = B.classmap(V, gg, clf)
    cmap = cache["cmap"]
    A0 = (B.visible_alpha(V, gg, s) > 0.3).cpu().numpy()
    if A0.sum() < 20:
        A0 = cmap == meta["class_idx"]
    chg = change_mask_pair(real_before, aft)
    newreg = chg & ~(cv2.dilate(A0.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0)
    n_lab, lab, st, _ = cv2.connectedComponentsWithStats(newreg.astype(np.uint8))
    big = [i for i in range(1, n_lab) if st[i, cv2.CC_STAT_AREA] >= 20]
    newcc = (lab == max(big, key=lambda i: st[i, cv2.CC_STAT_AREA])) if big else np.zeros_like(newreg)
    x0, y0, cw, ch = crop_box(chg | A0)
    cb, ca = real_before[y0:y0 + ch, x0:x0 + cw], aft[y0:y0 + ch, x0:x0 + cw]
    objc = A0[y0:y0 + ch, x0:x0 + cw]
    roi = cv2.dilate((chg | A0).astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
    static = ~roi
    centre = gg.xyz[s].mean(0).cpu().numpy().astype(float)
    est, T = {}, {}

    def on_obj_valid(uv):                                       # full-image px on the object with GG depth
        ui, vi = np.clip(uv[:, 0].round().astype(int), 0, W - 1), np.clip(uv[:, 1].round().astype(int), 0, H - 1)
        return A0[vi, ui] & np.isfinite(D_GG[vi, ui]), D_GG[vi, ui]

    if "inerf" in METHODS:
        tt = time.time(); est["inerf"] = inerf(s, aft, roi); T["inerf"] = time.time() - tt
    for name, matcher in (("lg_pnp", lambda: lightglue_matches(cb, ca)),
                          ("roma_pnp", lambda: X.roma_match(cb, ca, n=8000)[:2])):
        if name not in METHODS:
            continue
        tt = time.time()
        ka, kb = matcher()
        uv_b, uv_a = ka + [x0, y0], kb + [x0, y0]
        ok, z = on_obj_valid(uv_b)
        est[name] = pnp_motion(backproject(uv_b[ok], z[ok]), uv_a[ok], centre) if ok.sum() >= 6 else None
        T[name] = time.time() - tt
    if "cotracker" in METHODS:
        tt = time.time()
        vs, us = np.where(objc)
        est["cotracker"] = None
        if len(us):
            pick = np.random.default_rng(0).choice(len(us), min(1500, len(us)), replace=False)
            pts = np.stack([us[pick], vs[pick]], 1).astype(np.float32)
            tr, vis = cotracker_tracks(cb, ca, pts)
            uv_b, uv_a = pts + [x0, y0], tr + [x0, y0]
            mv = vis & (np.linalg.norm(uv_a - uv_b, axis=1) > MIN_PX)
            est["cotracker"] = ransac_xz(lift_plane(uv_b[mv], uv_a[mv]))
        T["cotracker"] = time.time() - tt
    if "icp" in METHODS:
        tt = time.time()
        est["icp"] = None
        if newcc.sum() >= 20:
            masks, scores = X.sam_mask(aft, [interior_point(newcc)], [interior_point(A0)])
            ok = [i for i in range(len(masks)) if (masks[i] & newcc).sum() >= 0.5 * newcc.sum() and masks[i].sum() < 0.05 * H * W]
            if ok:
                Mn = masks[max(ok, key=lambda i: scores[i])]
                if "da_before" not in cache:
                    cache["da_before"] = X.mono_depth(real_before)
                win = np.zeros((H, W), bool); win[max(y0 - 200, 0):y0 + ch + 200, max(x0 - 200, 0):x0 + cw + 200] = True
                Za = align_inv_depth(X.mono_depth(aft), D_GG, static & win)
                if Za is not None:
                    vo, uo = np.where(A0 & np.isfinite(D_GG)); vn, un = np.where(Mn & np.isfinite(Za))
                    if len(uo) > 30 and len(un) > 30:
                        rng = np.random.default_rng(0)
                        io_ = rng.choice(len(uo), min(3000, len(uo)), replace=False); in_ = rng.choice(len(un), min(3000, len(un)), replace=False)
                        src = backproject(np.stack([uo[io_], vo[io_]], 1).astype(float), D_GG[vo[io_], uo[io_]])
                        dst = backproject(np.stack([un[in_], vn[in_]], 1).astype(float), Za[vn[in_], un[in_]])
                        R, t = icp(src, dst)
                        c = src.mean(0); est["icp"] = R @ c + t - c
        T["icp"] = time.time() - tt
    if "mast3r" in METHODS:
        tt = time.time()
        est["mast3r"] = None
        mw, mh = 512, 384
        obj_small = cv2.resize(objc.astype(np.uint8), (mw, mh), interpolation=cv2.INTER_NEAREST) > 0
        mr = X.mast3r_pair(cb, ca, (mw, mh))      # reciprocal; query=obj_small (one-way) was tried: same failure
        sx, sy = cw / mw, ch / mh
        gg_small = cv2.resize(np.nan_to_num(D_GG[y0:y0 + ch, x0:x0 + cw], nan=0), (mw, mh), interpolation=cv2.INTER_NEAREST)
        st_small = cv2.resize(static[y0:y0 + ch, x0:x0 + cw].astype(np.uint8), (mw, mh), interpolation=cv2.INTER_NEAREST) > 0
        okz = st_small & (gg_small > 0.05) & (mr["z1"] > 0.01)
        if okz.sum() > 200:
            scale = float(np.median(gg_small[okz] / mr["z1"][okz]))
            m1, m2 = mr["m1"], mr["m2"]
            mot = np.hypot((m2[:, 0] - m1[:, 0]) * sx, (m2[:, 1] - m1[:, 1]) * sy) > MIN_PX
            conf = np.minimum(mr["c1"][m1[:, 1], m1[:, 0]], mr["c2"][m2[:, 1], m2[:, 0]]) > 1.5
            on = obj_small[m1[:, 1], m1[:, 0]] & mot & conf
            if on.sum() >= 8:
                m1, m2 = m1[on], m2[on]
                z1 = scale * mr["z1"][m1[:, 1], m1[:, 0]]; z2 = scale * mr["z2"][m2[:, 1], m2[:, 0]]
                uv1 = np.stack([m1[:, 0] * sx + x0, m1[:, 1] * sy + y0], 1); uv2 = np.stack([m2[:, 0] * sx + x0, m2[:, 1] * sy + y0], 1)
                est["mast3r"] = ransac_xz(backproject(uv2, z2) - backproject(uv1, z1), thr=0.03)
        T["mast3r"] = time.time() - tt
    return meta, est, T


def main():
    jobs = {}
    for td in sorted((BENCH / "trials").iterdir()):
        m = json.load(open(td / "meta.json"))
        for v in m.get("input_views", [m["input_view"]]):
            jobs.setdefault(v, []).append(td.name)
    out = BENCH / "eval_baselines.jsonl"
    done = {}
    if out.exists():
        for l in open(out):
            r = json.loads(l); done[(r["trial"], r["view"])] = set(r["est"])
    with open(out, "a") as f:
        for v, trials in sorted(jobs.items()):
            todo = [t for t in trials if not set(METHODS) <= done.get((t, v), set())]
            if not todo:
                continue
            set_view(v)
            cache = {}
            for tname in todo:
                t0 = time.time()
                meta, est, T = run(tname, v, cache)
                est = {k: (None if e is None else np.asarray(e, float).round(5).tolist()) for k, e in est.items()}
                f.write(json.dumps(dict(scene=args.scene, trial=tname, view=v, object=meta["name"], gt_delta=meta["delta_gg"],
                                        gt_norm_m=meta["delta_norm_m"], est=est, timing=T)) + "\n"); f.flush()
                gt = np.array(meta["delta_gg"])
                print(f"[{tname} {v}] {time.time() - t0:.0f}s gt {meta['delta_norm_m'] * 100:.0f}cm | " + " ".join(
                    f"{k} {'–' if e is None else format(np.linalg.norm(np.asarray(e) - gt) * 100, '.1f')}" for k, e in est.items()),
                    flush=True)


if __name__ == "__main__":
    main()
