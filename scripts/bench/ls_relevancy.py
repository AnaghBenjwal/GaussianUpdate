#!/usr/bin/env python3
"""
ls_relevancy.py — per-Gaussian LangSplat relevancy for every tracked object name (dev env, needs open_clip).

Each Gaussian's 3-d language feature (normalised, as LangSplat's renderer does) is decoded by the scene's
autoencoder to CLIP space, and scored with LangSplat's own LERF-style relevancy (positive vs. the canonical
negatives object/things/stuff/texture). Relevancy > 0.5 means "closer to the query than to the negatives".

    conda run -n dev python scripts/bench/ls_relevancy.py --scene FloorPlan1 --tag FP1
Writes ~/Documents/displacement_bench/<scene>/langsplat_relevancy.npz with keys L<level>_<class_idx>.
"""
import argparse, json, re, sys
from pathlib import Path

import numpy as np
import torch

PKG = Path.home() / "Documents/vast_results/server_root"
sys.path.insert(0, str(PKG / "langsplat/eval")); sys.path.insert(0, str(PKG / "langsplat/autoencoder"))
from openclip_encoder import OpenCLIPNetwork          # LangSplat's encoder (ViT-B/16 laion2b)
from model import Autoencoder

ap = argparse.ArgumentParser()
ap.add_argument("--scene", default="FloorPlan1"); ap.add_argument("--tag", default="FP1")
args = ap.parse_args()
meta = json.load(open(Path.home() / f"Documents/dataset_1080p_fixed/{args.scene}/env/scene_meta.json"))
classes = [(o["class_idx"], re.sub(r"(?<!^)(?=[A-Z])", " ", o["object_id"].split("|")[0]).lower()) for o in meta["objects"]]
P = PKG / "output/langsplat_package" / args.tag
ae = Autoencoder([256, 128, 64, 32, 3], [16, 32, 64, 128, 256, 256, 512]).cuda()
ae.load_state_dict(torch.load(P / "autoencoder/best_ckpt.pth", map_location="cuda")); ae.eval()
clip = OpenCLIPNetwork("cuda"); clip.set_positives([n for _, n in classes])
out = {}
for L in (1, 2, 3):
    params, _ = torch.load(P / f"level_{L}/chkpnt30000.pth", map_location="cuda", weights_only=False)
    lang = params[7].detach()
    lang = lang / (lang.norm(dim=-1, keepdim=True) + 1e-9)
    rel = []
    with torch.no_grad():
        for ch in torch.split(lang, 200000):
            e = ae.decode(ch); e = e / (e.norm(dim=-1, keepdim=True) + 1e-9)
            rel.append(torch.stack([clip.get_relevancy(e.half(), k)[..., 0] for k in range(len(classes))], 0).float().cpu())
    rel = torch.cat(rel, 1).numpy()
    for k, (c, n) in enumerate(classes):
        out[f"L{L}_{c}"] = rel[k].astype(np.float16)
        print(f"level {L} {n:16s}: {(rel[k] > 0.5).sum():7d} Gaussians > 0.5 (of {rel.shape[1]})")
np.savez_compressed(Path.home() / f"Documents/displacement_bench/{args.scene}/langsplat_relevancy.npz", **out)
