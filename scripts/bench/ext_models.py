"""
ext_models.py — external 2D/3D models for the displacement ablation (run_ablation.py). All lazily loaded.

  mono_depth(img)              Depth Anything V2 (Large): relative inverse depth, HxW
  sam_mask(img, pos, neg)      SAM ViT-H point-prompted masks: (3, H, W) bool + scores
  roma_match(a, b, n)          RoMa outdoor dense matcher: (n, 2) px in a, (n, 2) px in b, certainty
  mast3r_pair(a, b)            MASt3R (metric, 512): per-pixel depth in camera-1 frame for both images + matches
  raft_flow(a, b)              torchvision RAFT-large dense flow (same model as dino_raft.py), HxWx2

Images are uint8 HxWx3 RGB numpy arrays (crops are fine). Paths: /root/RoMa, /root/mast3r (see the server setup).
"""
from __future__ import annotations
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

DEV = "cuda"
_M = {}


def _get(name, make):
    if name not in _M:
        _M[name] = make()
    return _M[name]


# ── Depth Anything V2 ───────────────────────────────────────────────────────────
def mono_depth(img: np.ndarray) -> np.ndarray:
    def make():
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        mid = "depth-anything/Depth-Anything-V2-Large-hf"
        return AutoImageProcessor.from_pretrained(mid), AutoModelForDepthEstimation.from_pretrained(mid).to(DEV).eval()
    proc, model = _get("da", make)
    inp = proc(images=Image.fromarray(img), return_tensors="pt").to(DEV)
    with torch.no_grad():
        d = model(**inp).predicted_depth                          # (1, h, w) relative inverse depth
    return F.interpolate(d[None], size=img.shape[:2], mode="bicubic", align_corners=False)[0, 0].cpu().numpy()


# ── SAM ─────────────────────────────────────────────────────────────────────────
def sam_mask(img: np.ndarray, pos: list, neg: list = ()):
    def make():
        from transformers import SamModel, SamProcessor
        return SamProcessor.from_pretrained("facebook/sam-vit-huge"), SamModel.from_pretrained("facebook/sam-vit-huge").to(DEV).eval()
    proc, model = _get("sam", make)
    pts = [[list(map(float, p)) for p in list(pos) + list(neg)]]
    lab = [[1] * len(pos) + [0] * len(neg)]
    inp = proc(Image.fromarray(img), input_points=[pts], input_labels=[lab], return_tensors="pt").to(DEV)
    with torch.no_grad():
        out = model(**inp, multimask_output=True)
    m = proc.image_processor.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"].cpu(), inp["reshaped_input_sizes"].cpu())
    return m[0][0].numpy().astype(bool), out.iou_scores[0, 0].float().cpu().numpy()


# ── RoMa ────────────────────────────────────────────────────────────────────────
def roma_match(a: np.ndarray, b: np.ndarray, n=5000):
    def make():
        sys.path.insert(0, "/root/RoMa")
        from romatch import roma_outdoor
        return roma_outdoor(device=DEV)
    rm = _get("roma", make)
    warp, cert = rm.match(Image.fromarray(a), Image.fromarray(b), device=DEV)
    m, c = rm.sample(warp, cert, num=n)
    ka, kb = rm.to_pixel_coordinates(m, a.shape[0], a.shape[1], b.shape[0], b.shape[1])
    return ka.cpu().numpy(), kb.cpu().numpy(), c.cpu().numpy()


# ── MASt3R ──────────────────────────────────────────────────────────────────────
def mast3r_pair(a: np.ndarray, b: np.ndarray, size=(512, 384), query: np.ndarray | None = None):
    """Both crops resized to `size` (W, H; multiples of 16). Returns z1, z2 (H, W) depth in camera-1 frame (camera did
    not move, so z2 is image 2's depth), confidences, and descriptor matches (xy at `size` resolution): reciprocal
    nearest neighbours, or — with `query` (H, W bool) — one-way nearest neighbours of the query pixels of image 1
    (reciprocal matching drops almost every match on a moved object)."""
    def make():
        sys.path[:0] = ["/root/mast3r", "/root/mast3r/dust3r"]
        from mast3r.model import AsymmetricMASt3R
        return AsymmetricMASt3R.from_pretrained("naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric").to(DEV).eval()
    model = _get("mast3r", make)
    from dust3r.inference import inference
    from mast3r.fast_nn import fast_reciprocal_NNs
    W, H = size

    def prep(x, i):
        t = torch.tensor(np.asarray(Image.fromarray(x).resize((W, H), Image.BICUBIC)), dtype=torch.float32)
        return dict(img=(t.permute(2, 0, 1)[None] / 127.5 - 1), true_shape=np.int32([[H, W]]), idx=i, instance=str(i))
    out = inference([(prep(a, 0), prep(b, 1))], model, DEV, batch_size=1, verbose=False)
    p1, p2 = out["pred1"], out["pred2"]
    d1, d2 = p1["desc"][0].detach(), p2["desc"][0].detach()
    if query is None:
        m1, m2 = fast_reciprocal_NNs(d1, d2, subsample_or_initxy1=2, device=DEV, dist="dot", block_size=2 ** 13)
    else:
        ys, xs = np.where(query[::2, ::2]); ys, xs = ys * 2, xs * 2
        q = d1[torch.as_tensor(ys, device=d1.device), torch.as_tensor(xs, device=d1.device)]      # (N, C)
        idx = torch.cat([(c @ d2.reshape(-1, d2.shape[-1]).T).argmax(1) for c in q.split(1024)]).cpu().numpy()
        m1 = np.stack([xs, ys], 1); m2 = np.stack([idx % W, idx // W], 1)
    return dict(z1=p1["pts3d"][0, ..., 2].cpu().numpy(), z2=p2["pts3d_in_other_view"][0, ..., 2].cpu().numpy(),
                c1=p1["conf"][0].cpu().numpy(), c2=p2["conf"][0].cpu().numpy(), m1=np.asarray(m1), m2=np.asarray(m2))


# ── RAFT (dense) ────────────────────────────────────────────────────────────────
def raft_flow(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    def make():
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
        return raft_large(weights=Raft_Large_Weights.DEFAULT).to(DEV).eval()
    model = _get("raft", make)
    H, W = a.shape[:2]; Hp, Wp = (H + 7) // 8 * 8, (W + 7) // 8 * 8
    t = lambda x: F.pad(torch.tensor(x, dtype=torch.float32).permute(2, 0, 1)[None], (0, Wp - W, 0, Hp - H), mode="replicate").to(DEV) / 127.5 - 1
    with torch.no_grad():
        f = model(t(a), t(b))[-1][0, :, :H, :W]
    return f.permute(1, 2, 0).cpu().numpy()
