"""
bench_lib.py — building blocks for the object-displacement benchmark (gaussian_grouping env).

Pipeline stages, each with interchangeable implementations:
  A. identification  : which object moved            (pixel-diff | DINO | render-and-compare)
  B. selection       : which Gaussians belong to it  (GG class | GG class + 3D cleanup | LangSplat relevancy | GT AABB oracle)
  C. displacement    : 3D translation                (const-depth backprojection [current] | support-plane ray cast | render-and-compare)
  D. application     : apply + render                (translate | translate + support-plane hole fill)

All 3D quantities are in the fixed-dataset world (world' = diag(1,1,-1) @ unity), y is up.
"""
from __future__ import annotations
import math, sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov

DEV = "cuda"
C0 = 0.28209479177387814
RES = (1600, 900)


# ═══════════════════════════════ cameras ═══════════════════════════════
@dataclass
class View:
    name: str
    W: int
    H: int
    fovx: float
    fovy: float
    wvt: torch.Tensor          # world_view_transform (GG convention, transposed)
    fpt: torch.Tensor          # full_proj_transform
    center: torch.Tensor       # camera centre (world)
    R: np.ndarray              # world -> cam (OpenCV) rotation
    t: np.ndarray
    fx: float
    fy: float

    def scaled(self, s: float) -> "View":
        return View(self.name, int(self.W * s), int(self.H * s), self.fovx, self.fovy, self.wvt, self.fpt,
                    self.center, self.R, self.t, self.fx * s, self.fy * s)

    def project(self, X: np.ndarray) -> np.ndarray:                 # (N,3) world -> (N,2) px, (N,) depth
        pc = X @ self.R.T + self.t
        return np.stack([self.fx * pc[:, 0] / pc[:, 2] + self.W / 2, self.fy * pc[:, 1] / pc[:, 2] + self.H / 2], 1), pc[:, 2]

    def ray(self, u: float, v: float):                               # world origin, unit direction
        d_cam = np.array([(u - self.W / 2) / self.fx, (v - self.H / 2) / self.fy, 1.0])
        d = self.R.T @ d_cam
        return -self.R.T @ self.t, d / np.linalg.norm(d)


def load_views(env_dir: str, names: list[str], res=RES) -> dict[str, View]:
    extr = read_extrinsics_binary(f"{env_dir}/sparse/0/images.bin")
    intr = read_intrinsics_binary(f"{env_dir}/sparse/0/cameras.bin")
    out = {}
    for e in extr.values():
        if e.name not in names:
            continue
        cam = intr[e.camera_id]
        s = res[0] / cam.width
        fx, fy = cam.params[0] * s, cam.params[1] * s
        Rw2c = qvec2rotmat(e.qvec); t = np.array(e.tvec)
        fovx, fovy = focal2fov(fx, res[0]), focal2fov(fy, res[1])
        wvt = torch.tensor(getWorld2View2(Rw2c.T, t)).transpose(0, 1).float().to(DEV)
        proj = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy).transpose(0, 1).to(DEV)
        fpt = wvt.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
        out[e.name] = View(e.name, res[0], res[1], fovx, fovy, wvt, fpt, wvt.inverse()[3, :3], Rw2c, t, fx, fy)
    assert len(out) == len(names), f"missing views: {set(names) - set(out)}"
    return out


# ═══════════════════════════════ splats ═══════════════════════════════
@dataclass
class Splats:
    xyz: torch.Tensor          # (N,3)
    feats: torch.Tensor        # (N,16,3) SH
    opacity: torch.Tensor      # (N,1) activated
    scales: torch.Tensor       # (N,3) activated
    rots: torch.Tensor         # (N,4) normalised
    objs: torch.Tensor         # (N,1,16) GG identity encoding (zeros for non-GG models)
    sh_degree: int = 3
    lang: torch.Tensor | None = None

    def subset(self, m: torch.Tensor) -> "Splats":
        return Splats(self.xyz[m], self.feats[m], self.opacity[m], self.scales[m], self.rots[m], self.objs[m], self.sh_degree,
                      None if self.lang is None else self.lang[m])

    def cat(self, o: "Splats") -> "Splats":
        return Splats(*(torch.cat([a, b]) for a, b in [(self.xyz, o.xyz), (self.feats, o.feats), (self.opacity, o.opacity),
                                                         (self.scales, o.scales), (self.rots, o.rots), (self.objs, o.objs)]),
                      self.sh_degree)

    @property
    def n(self):
        return self.xyz.shape[0]


def load_gg(model_dir: str, iteration=40000, num_classes=200):
    from scene.gaussian_model import GaussianModel
    g = GaussianModel(3)
    g.load_ply(f"{model_dir}/point_cloud/iteration_{iteration}/point_cloud.ply")
    clf = torch.nn.Conv2d(g.num_objects, num_classes, kernel_size=1).to(DEV)
    clf.load_state_dict(torch.load(f"{model_dir}/point_cloud/iteration_{iteration}/classifier.pth", map_location=DEV))
    clf.eval()
    sp = Splats(g.get_xyz.detach(), g.get_features.detach(), g.get_opacity.detach(), g.get_scaling.detach(),
                g.get_rotation.detach(), g._objects_dc.detach(), g.max_sh_degree)
    with torch.no_grad():
        cls = clf(sp.objs.permute(2, 0, 1)).argmax(0).squeeze()                      # per-Gaussian class
    return sp, clf, cls


def load_langsplat(chkpnt: str):
    params, _ = torch.load(chkpnt, map_location=DEV, weights_only=False)
    sh, xyz, fdc, frest, sc, rot, op, lang = params[:8]
    n = xyz.shape[0]
    return Splats(xyz.detach(), torch.cat([fdc, frest], 1).detach(), torch.sigmoid(op).detach(), torch.exp(sc).detach(),
                  F.normalize(rot).detach(), torch.zeros(n, 1, 16, device=DEV), sh, lang.detach())


# ═══════════════════════════════ rendering ═══════════════════════════════
def render(view: View, sp: Splats, xyz: torch.Tensor | None = None, colors=None, bg=None):
    """Returns (rgb (3,H,W), objects (16,H,W)). Differentiable w.r.t. xyz."""
    xyz = sp.xyz if xyz is None else xyz
    if not bool(torch.isfinite(xyz).all()):                         # NaN/inf means crash the CUDA rasterizer
        raise FloatingPointError(f"render: {int((~torch.isfinite(xyz)).any(1).sum())} non-finite Gaussian positions")
    bg = torch.zeros(3, device=DEV) if bg is None else bg
    st = GaussianRasterizationSettings(image_height=view.H, image_width=view.W, tanfovx=math.tan(view.fovx / 2),
                                       tanfovy=math.tan(view.fovy / 2), bg=bg, scale_modifier=1.0, viewmatrix=view.wvt,
                                       projmatrix=view.fpt, sh_degree=sp.sh_degree, campos=view.center,
                                       prefiltered=False, debug=False)
    means2D = torch.zeros_like(xyz, requires_grad=True)
    img, _, objs = GaussianRasterizer(st)(means3D=xyz, means2D=means2D, shs=None if colors is not None else sp.feats,
                                         sh_objs=sp.objs, colors_precomp=colors, opacities=sp.opacity,
                                         scales=sp.scales, rotations=sp.rots, cov3D_precomp=None)
    return img, objs


def cam_depth(view: View, xyz: torch.Tensor) -> torch.Tensor:
    return (torch.cat([xyz, torch.ones_like(xyz[:, :1])], 1) @ view.wvt)[:, 2]


def render_depth_alpha(view: View, sp: Splats, xyz=None):
    """alpha-weighted depth and alpha (via colors_precomp = [z, 1, 0])."""
    xyz = sp.xyz if xyz is None else xyz
    z = cam_depth(view, xyz)
    col = torch.stack([z, torch.ones_like(z), torch.zeros_like(z)], 1)
    img, _ = render(view, sp, xyz, colors=col)
    a = img[1]
    return img[0] / a.clamp(min=1e-4), a


def classmap(view: View, sp: Splats, clf) -> np.ndarray:
    with torch.no_grad():
        _, objs = render(view, sp)
        return clf(objs).argmax(0).cpu().numpy().astype(np.int32)


def to_u8(t: torch.Tensor) -> np.ndarray:
    return (t.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)


def load_img(path, res=RES) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    return np.asarray(im if im.size == res else im.resize(res))


def to_t(img_u8: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img_u8).to(DEV).permute(2, 0, 1).float() / 255


# ═══════════════════════════════ A. identification ═══════════════════════════════
def class_scores_from_change(change_mask: np.ndarray, cmap: np.ndarray) -> dict[int, float]:
    """Exactly the scoring of detect_and_update.get_changed_gaussian_indices: fraction of each class's pixels inside the change mask."""
    ids, cnt = np.unique(cmap, return_counts=True); tot = dict(zip(ids.tolist(), cnt.tolist()))
    reg = cmap[change_mask > 0]
    u, c = np.unique(reg, return_counts=True)
    return {int(k): v / tot.get(k, 1) for k, v in zip(u.tolist(), c.tolist()) if k > 0}


def identify_pixeldiff(before_u8, after_u8, cmap):
    from detect_and_update import detect_changed_region
    m, _ = detect_changed_region(before_u8, after_u8, diff_thresh=25.0, sam2_helper=None)
    s = class_scores_from_change(m, cmap)
    return (max(s, key=s.get) if s else -1), m


def identify_dino(before_u8, after_u8, cmap):
    from dino_raft import detect_changed_region_dino
    m = detect_changed_region_dino(before_u8, after_u8, similarity_threshold=0.8)
    s = class_scores_from_change(m, cmap)
    return (max(s, key=s.get) if s else -1), m


# ═══════════════════════════════ B. selection ═══════════════════════════════
def sel_class(cls_ids: torch.Tensor, c: int) -> torch.Tensor:
    return cls_ids == c


def sel_depth_filter(sel: torch.Tensor, sp: Splats, view: View) -> torch.Tensor:
    """current pipeline (force_obj_id branch): keep camera depth within mean ± 3 std."""
    idx = torch.where(sel)[0]
    if len(idx) < 3:
        return sel
    z = cam_depth(view, sp.xyz[idx]); keep = (z > z.mean() - 3 * z.std()) & (z < z.mean() + 3 * z.std()) & (z > 0)
    out = torch.zeros_like(sel); out[idx[keep]] = True
    return out


def sel_clean3d(sel: torch.Tensor, sp: Splats, eps=0.03, min_samples=5) -> torch.Tensor:
    """3D cleanup: keep the largest DBSCAN cluster (+ opacity-weighted size), dropping scattered same-class floaters."""
    from sklearn.cluster import DBSCAN
    idx = torch.where(sel)[0]
    if len(idx) < 10:
        return sel
    X = sp.xyz[idx].cpu().numpy()
    lab = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    if (lab >= 0).sum() == 0:
        return sel
    w = sp.opacity[idx, 0].cpu().numpy()
    best = max(set(lab[lab >= 0].tolist()), key=lambda l: w[lab == l].sum())
    out = torch.zeros_like(sel); out[idx[torch.from_numpy(lab == best).to(DEV)]] = True
    return out


def sel_aabb(sp: Splats, aabb_corners_gg) -> torch.Tensor:
    """GT oracle: Gaussian centres inside the object's AABB (bottom raised slightly to exclude the support surface)."""
    c = np.array(aabb_corners_gg); lo, hi = c.min(0), c.max(0)
    lo = lo - [0.01, 0, 0.01]; hi = hi + [0.01, 0.005, 0.01]
    lo[1] += min(0.01, 0.3 * (hi[1] - lo[1]))
    lo_t, hi_t = torch.tensor(lo, device=DEV, dtype=torch.float32), torch.tensor(hi, device=DEV, dtype=torch.float32)
    return ((sp.xyz >= lo_t) & (sp.xyz <= hi_t)).all(1)


def sel_prf(sel: torch.Tensor, gt: torch.Tensor) -> dict:
    tp = int((sel & gt).sum()); p = tp / max(int(sel.sum()), 1); r = tp / max(int(gt.sum()), 1)
    return {"sel_n": int(sel.sum()), "gt_n": int(gt.sum()), "precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-9)}


# ═══════════════════════════════ C. displacement ═══════════════════════════════
def obj_centroid_2d(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    return (float(xs.mean()), float(ys.mean())) if len(xs) else (mask.shape[1] / 2, mask.shape[0] / 2)


def backproject_const_depth(view: View, sp: Splats, sel, cb, ca) -> np.ndarray:
    """Current method (detect_and_update.backproject_displacement): camera-plane shift at the mean object depth, dz_cam = 0."""
    z = max(float(cam_depth(view, sp.xyz[sel]).mean()), 0.1)
    d_cam = np.array([(ca[0] - cb[0]) / view.fx * z, (ca[1] - cb[1]) / view.fy * z, 0.0])
    d = view.R.T @ d_cam                                             # cam -> world rotation
    return d if np.isfinite(d).all() else np.zeros(3)


def shift_template(before_u8, after_u8, mask):
    from detect_and_update import estimate_2d_shift_template
    return estimate_2d_shift_template(before_u8, after_u8, mask.astype(np.uint8))


def shift_raft(before_u8, after_u8, mask):
    from dino_raft import estimate_displacement_raft
    dx, dy, conf, _ = estimate_displacement_raft(before_u8, after_u8, mask.astype(np.uint8))
    return dx, dy


def backproject_plane(view: View, sp: Splats, sel, shift) -> np.ndarray:
    """Support-plane model: the object slides on a horizontal surface. Project the 3D object centre, shift it by the
    2D motion, cast the ray through the new pixel and intersect it with the horizontal plane at the centre's height."""
    if not np.isfinite(shift).all():
        return np.zeros(3)
    c = sp.xyz[sel].median(0).values.cpu().numpy()
    (p,), _ = view.project(c[None])
    o, d = view.ray(p[0] + shift[0], p[1] + shift[1])
    if abs(d[1]) < 1e-3:
        return np.zeros(3)
    lam = (c[1] - o[1]) / d[1]
    if lam <= 0:
        return np.zeros(3)
    X = o + lam * d
    delta = X - c; delta[1] = 0.0
    return delta if np.isfinite(delta).all() else np.zeros(3)


def _down_nan(x: torch.Tensor, v):
    """Area-downsample an HxW map with NaN = unknown; a pixel is known only if its whole footprint is."""
    m = torch.isfinite(x).float()
    num = F.interpolate(torch.nan_to_num(x)[None, None], size=(v.H, v.W), mode="area")[0, 0]
    den = F.interpolate(m[None, None], size=(v.H, v.W), mode="area")[0, 0]
    return num / den.clamp_min(1e-6), den > 0.99


def keep_weight(exclude, v):
    if exclude is None:
        return torch.ones(1, device=DEV)
    m = cv2.resize((~exclude).astype(np.float32), (v.W, v.H), interpolation=cv2.INTER_AREA)
    return torch.tensor(m, device=DEV)[None]


def render_and_compare(view: View, sp: Splats, sel, after_t: torch.Tensor, init=(np.zeros(3),),
                       radius=0.5, step=0.04, scale=0.25, iters=60, ref_t: torch.Tensor | None = None,
                       exclude: np.ndarray | None = None, center=None, depth_delta: torch.Tensor | None = None,
                       depth_w: float = 1.0):
    """Analysis-by-synthesis: find the planar translation (tx, tz) of the selected Gaussians whose render best explains
    the after image. Coarse: wide grid with depth-aware compositing of a separately rendered object layer (fast).
    Fine: gradient descent on the full differentiable render. Returns (delta, loss_moved, loss_static).
    ref_t (a real BEFORE image of the same view) switches to the differential objective
    |(R(delta) - R(0)) - (after - before)|, which cancels the model-vs-reality appearance gap.
    exclude (bool HxW) drops pixels from the loss, e.g. the object's old footprint, whose never-observed hole otherwise
    outweighs the signal of a small or transparent object.
    center: local mode — the coarse grid is centred on this delta and the fine stage is clamped to within radius of it.
    depth_delta (HxW, NaN = unknown): external depth change after - before (e.g. aligned monocular depth); adds the
    differential depth loss depth_w * |(D(delta) - D(0)) - depth_delta| / D on pixels where it is known."""
    c0 = np.zeros(3) if center is None else np.asarray(center, float) * [1, 0, 1]
    obj, rest = sp.subset(sel), sp.subset(~sel)
    v_lo = view.scaled(scale)
    tgt_lo = F.interpolate(after_t[None], size=(v_lo.H, v_lo.W), mode="area")[0]
    with torch.no_grad():
        bg_rgb, _ = render(v_lo, rest)
        bg_d, bg_a = render_depth_alpha(v_lo, rest)
        bg_d = torch.where(bg_a > 0.05, bg_d, torch.full_like(bg_d, 1e4))

        def composite(delta3, with_depth=False):
            x = obj.xyz + torch.tensor(delta3, device=DEV, dtype=torch.float32)
            o_rgb, _ = render(v_lo, obj, x)
            o_d, o_a = render_depth_alpha(v_lo, obj, x)
            front = (o_d < bg_d) & (o_a > 0.01)
            rgb = torch.where(front[None], o_rgb + (1 - o_a)[None] * bg_rgb, bg_rgb)
            return (rgb, torch.where(front & (o_a > 0.5), o_d, bg_d)) if with_depth else rgb

        if ref_t is not None:                                        # differential target: R(0) + (after - before)
            ref_lo = F.interpolate(ref_t[None], size=(v_lo.H, v_lo.W), mode="area")[0]
            tgt_lo = composite(np.zeros(3)) + (tgt_lo - ref_lo)

        w_lo = keep_weight(exclude, v_lo)

        if depth_delta is not None:
            dz_lo, dm_lo = _down_nan(depth_delta, v_lo)
            dtgt_lo = composite(np.zeros(3), True)[1] + dz_lo
            dm_lo = dm_lo & (dtgt_lo > 0.05) & (dtgt_lo < 1e3)

        def comp_loss(delta3):
            if depth_delta is None:
                return float(((composite(delta3) - tgt_lo).abs() * w_lo).mean())
            rgb, d = composite(delta3, True)
            dl = torch.where(dm_lo, (d - dtgt_lo).abs() / dtgt_lo.clamp_min(0.1), torch.zeros_like(d))
            return float(((rgb - tgt_lo).abs() * w_lo).mean() + depth_w * (dl.clamp(max=1.0) * w_lo).mean())

        g = np.arange(-radius, radius + 1e-9, step)
        cands = [c0 + np.array([a, 0.0, b]) for a in g for b in g] + [np.asarray(i, float) * [1, 0, 1] for i in init if np.isfinite(i).all()]
        losses = [comp_loss(c3) for c3 in cands]
        best = cands[int(np.argmin(losses))]

    # fine: true render (correct occlusion) at half resolution, Adam on (tx, tz)
    v_md = view.scaled(0.5)
    tgt_md = F.interpolate(after_t[None], size=(v_md.H, v_md.W), mode="area")[0]
    if ref_t is not None:
        with torch.no_grad():
            tgt_md = render(v_md, sp)[0] + (tgt_md - F.interpolate(ref_t[None], size=(v_md.H, v_md.W), mode="area")[0])
    w_md = keep_weight(exclude, v_md)
    if depth_delta is not None:
        dz_md, dm_md = _down_nan(depth_delta, v_md)
        with torch.no_grad():
            dtgt_md = render_depth_alpha(v_md, sp)[0] + dz_md
        dm_md = dm_md & (dtgt_md > 0.05) & (dtgt_md < 1e3)
    t = torch.tensor([best[0], best[2]], device=DEV, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([t], lr=0.004)
    idx = torch.where(sel)[0]
    best_t, best_l = t.detach().clone(), 1e9
    for _ in range(iters):
        x = sp.xyz.clone()
        x[idx] = x[idx] + torch.stack([t[0], torch.zeros_like(t[0]), t[1]])
        img, _ = render(v_md, sp, x)
        loss = ((img - tgt_md).abs() * w_md).mean()
        if depth_delta is not None:
            d, a = render_depth_alpha(v_md, sp, x)
            m = dm_md & (a.detach() > 0.5)
            loss = loss + depth_w * (torch.where(m, (d - dtgt_md).abs() / dtgt_md.clamp_min(0.1), torch.zeros_like(d)).clamp(max=1.0) * w_md).mean()
        if loss.item() < best_l:
            best_l, best_t = loss.item(), t.detach().clone()
        opt.zero_grad(); loss.backward(); opt.step()
        if center is not None:                                       # local mode: stay within radius of the centre
            with torch.no_grad():
                t[0].clamp_(c0[0] - radius, c0[0] + radius); t[1].clamp_(c0[2] - radius, c0[2] + radius)
    with torch.no_grad():
        static = float(((render(v_md, sp)[0] - tgt_md).abs() * w_md).mean())
    return np.array([best_t[0].item(), 0.0, best_t[1].item()]), best_l, static


def visible_alpha(view: View, sp: Splats, sel, delta3=(0, 0, 0)) -> torch.Tensor:
    """Alpha of the selected Gaussians after translating them by delta3, zeroed where the rest of the scene is in front."""
    obj, rest = sp.subset(sel), sp.subset(~sel)
    with torch.no_grad():
        bg_d, bg_a = render_depth_alpha(view, rest)
        bg_d = torch.where(bg_a > 0.05, bg_d, torch.full_like(bg_d, 1e4))
        x = obj.xyz + torch.tensor(np.asarray(delta3, float), device=DEV, dtype=torch.float32)
        o_d, o_a = render_depth_alpha(view, obj, x)
        return torch.where(o_d < bg_d, o_a, torch.zeros_like(o_a))


def support_checker(sp: Splats, sel, band=(-0.03, 0.02), radius=0.015, q=0.05, down=-1):
    """Physical plausibility of a planar move: the object's footprint (its lowest Gaussians, GG y-min side) must land on
    scene geometry at the same height. Returns f(delta3) -> fraction of footprint points with a scene Gaussian within
    `radius` in xz and within `band` of the base height. Resolves the depth ambiguity of single-view fitting
    (sliding the object along the viewing ray changes the image little but makes it float or sink into a wall)."""
    from scipy.spatial import cKDTree
    xo = sp.xyz[sel].detach().cpu().numpy() * [1, -down, 1]       # flip so that 'down' is -y
    xr = sp.xyz[~sel].detach().cpu().numpy() * [1, -down, 1]
    yb = np.quantile(xo[:, 1], q)
    foot = xo[xo[:, 1] <= np.quantile(xo[:, 1], 0.15)][:, [0, 2]]
    foot = foot[np.random.default_rng(0).choice(len(foot), min(len(foot), 300), replace=False)] if len(foot) else foot
    ring = xr[(xr[:, 1] > yb + band[0]) & (xr[:, 1] < yb + band[1])][:, [0, 2]]
    tree = cKDTree(ring) if len(ring) else None

    def frac(d3):
        if tree is None or not len(foot):
            return 1.0
        dist, _ = tree.query(foot + np.asarray(d3, float)[[0, 2]], distance_upper_bound=radius)
        return float(np.isfinite(dist).mean())
    return frac


def fit_silhouette(view: View, sp: Splats, sel, change_mask: np.ndarray, init=(np.zeros(3),), radius=0.5, hard=None):
    """Appearance-free displacement: the real before/after change mask should equal old ∪ new object silhouette.
    Grid-search the planar move maximising soft IoU(A(0) ∪ A(delta), change), coarse (2 cm, 1/4 res) then fine
    (0.5 cm, 1/2 res). Works for low-contrast objects where colour render-and-compare has no signal. hard=thr binarises
    the object alpha (alpha > thr), for transparent objects whose Gaussians have low opacity.
    Returns (delta, iou_best, iou_zero)."""
    obj, rest = sp.subset(sel), sp.subset(~sel)

    def make(scale):
        v = view.scaled(scale)
        c = torch.tensor(cv2.resize(change_mask.astype(np.float32), (v.W, v.H), interpolation=cv2.INTER_AREA), device=DEV)
        with torch.no_grad():
            bg_d, bg_a = render_depth_alpha(v, rest)
            bg_d = torch.where(bg_a > 0.05, bg_d, torch.full_like(bg_d, 1e4))

            def alpha(d3):
                x = obj.xyz + torch.tensor(np.asarray(d3, float), device=DEV, dtype=torch.float32)
                o_d, o_a = render_depth_alpha(v, obj, x)
                a = torch.where(o_d < bg_d, o_a, torch.zeros_like(o_a)).clamp(0, 1)
                return a if hard is None else (a > hard).float()
            a0 = alpha(np.zeros(3))

            def iou(d3):
                p = torch.maximum(a0, alpha(d3))
                return float((p * c).sum() / (p + c - p * c).sum().clamp_min(1e-6))
        return iou

    with torch.no_grad():
        iou_lo = make(0.25)
        g = np.arange(-radius, radius + 1e-9, 0.02)
        cands = [np.array([a, 0.0, b]) for a in g for b in g] + [np.asarray(i, float) * [1, 0, 1] for i in init if np.isfinite(i).all()]
        sc = [iou_lo(c3) for c3 in cands]
        best = cands[int(np.argmax(sc))]
        iou_md = make(0.5)
        f = np.arange(-0.03, 0.03 + 1e-9, 0.005)
        fc = [best + np.array([a, 0.0, b]) for a in f for b in f]
        fs = [iou_md(c3) for c3 in fc]
        k = int(np.argmax(fs))
        return fc[k], fs[k], iou_md(np.zeros(3))


# ═══════════════════════════════ D. application ═══════════════════════════════
def apply_translation(sp: Splats, sel, delta) -> Splats:
    x = sp.xyz.clone(); x[sel] += torch.tensor(delta, device=DEV, dtype=torch.float32)
    return Splats(x, sp.feats, sp.opacity, sp.scales, sp.rots, sp.objs, sp.sh_degree)


def hole_fill(sp_moved: Splats, sp_orig: Splats, sel, view: View, after_u8: np.ndarray, stride=3) -> Splats:
    """Fill the surface revealed by the move: pixels covered by the object before but not after (input view) are cast
    onto the support plane (object bottom height) and become thin, opaque, horizontal Gaussians coloured from the after
    image. The support surface was never observed under the object, so the trained model has nothing there."""
    with torch.no_grad():
        ones = lambda s: torch.ones(s.n, 3, device=DEV)
        a0 = render(view, sp_orig.subset(sel), colors=ones(sp_orig.subset(sel)))[0][0]
        a1 = render(view, sp_moved.subset(sel), colors=ones(sp_moved.subset(sel)))[0][0]
        vac = ((a0 > 0.5) & (a1 < 0.3)).cpu().numpy()
    ys, xs = np.where(vac[::stride, ::stride]); ys, xs = ys * stride, xs * stride
    if len(xs) == 0:
        return sp_moved
    y_plane = float(torch.quantile(sp_orig.xyz[sel][:, 1], 0.02))
    o = -view.R.T @ view.t
    dc = np.stack([(xs - view.W / 2) / view.fx, (ys - view.H / 2) / view.fy, np.ones_like(xs, float)], 1) @ view.R
    ok = np.abs(dc[:, 1]) > 1e-4
    lam = (y_plane - o[1]) / dc[:, 1]
    ok &= lam > 0
    X = o + lam[:, None] * dc
    depth = (X[ok] - o) @ view.R[2]
    X, cols = X[ok], after_u8[ys[ok], xs[ok]].astype(np.float32) / 255
    k = len(X)
    s = np.clip(depth * stride / view.fx * 0.8, 1e-3, 0.05)
    feats = torch.zeros(k, 16, 3, device=DEV)
    feats[:, 0] = torch.tensor((cols - 0.5) / C0, device=DEV, dtype=torch.float32)
    new = Splats(torch.tensor(X, device=DEV, dtype=torch.float32), feats, torch.full((k, 1), 0.98, device=DEV),
                 torch.tensor(np.stack([s, np.full(k, 1e-3), s], 1), device=DEV, dtype=torch.float32),
                 torch.tensor([[1.0, 0, 0, 0]], device=DEV).repeat(k, 1), torch.zeros(k, 1, 16, device=DEV), sp_moved.sh_degree)
    return sp_moved.cat(new)


# ═══════════════════════════════ metrics ═══════════════════════════════
_W = None


def ssim_map(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    global _W
    if _W is None:
        g = torch.exp(-((torch.arange(11) - 5) ** 2) / (2 * 1.5 ** 2)); g = g / g.sum()
        _W = (g[:, None] @ g[None]).expand(3, 1, 11, 11).contiguous().to(DEV)
    a, b = a[None], b[None]
    mu1, mu2 = F.conv2d(a, _W, padding=5, groups=3), F.conv2d(b, _W, padding=5, groups=3)
    s11 = F.conv2d(a * a, _W, padding=5, groups=3) - mu1 ** 2
    s22 = F.conv2d(b * b, _W, padding=5, groups=3) - mu2 ** 2
    s12 = F.conv2d(a * b, _W, padding=5, groups=3) - mu1 * mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (s11 + s22 + C2)))[0].mean(0)


def image_metrics(pred: torch.Tensor, gt: torch.Tensor, roi: np.ndarray | None) -> dict:
    with torch.no_grad():
        mse = ((pred - gt) ** 2).mean(0)
        sm = ssim_map(pred, gt)
        out = {"psnr": float(-10 * torch.log10(mse.mean())), "ssim": float(sm.mean())}
        if roi is not None and roi.sum() > 50:
            r = torch.from_numpy(roi).to(DEV)
            out["psnr_roi"] = float(-10 * torch.log10(mse[r].mean()))
            out["ssim_roi"] = float(sm[r].mean())
    return out
