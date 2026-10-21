#!/usr/bin/env python3
"""
predict.py — run a trained DispNet on a split and write per-sample predictions (gravity frame and GG world).

    python scripts/learn/predict.py --ckpt runs/r18/best.pt --split test_t1_gg --out preds_t1_gg.jsonl
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sample_io as io
import data as D
from data import DispDataset
from model import DispNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path.home() / "Documents/displacement_learn"))
    ap.add_argument("--split", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tta", action="store_true", help="average with the horizontally flipped input")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    A = json.load(open(Path(args.ckpt).parent / "args.json"))
    norm = A.get("norm", "fixed")                                            # r18 predates --norm
    D.RES = A.get("res", 256)
    ds = DispDataset([Path(args.data) / args.split], norm=norm)
    model = DispNet(pretrained=False, norm=norm, arch=A.get("arch", "r18"), planar=A.get("planar", False)).to(dev).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location=dev))
    dl = torch.utils.data.DataLoader(ds, batch_size=64, num_workers=8)
    with open(args.out, "w") as f, torch.no_grad():
        for b in dl:
            x, g = b["x"].to(dev), b["g"].to(dev)
            mu, logb = model(x, g)
            if args.tta:                                                     # flip: x of points, anchor and output
                xf, gf = x.flip(-1), g.clone()
                xf[:, 7] *= -1; gf[:, 1] *= -1
                mf, lf = model(xf, gf)
                mf[:, 0] *= -1
                mu, logb = (mu + mf) / 2, torch.logaddexp(logb, lf) - np.log(2)
            for i, d, s in zip(b["idx"].tolist(), mu.cpu().numpy(), logb.exp().cpu().numpy()):
                m = json.load(open(ds.dirs[i] / "meta.json"))
                du = io.grav_rotation(m["cam_yaw"]).T @ d                    # grav -> unity
                f.write(json.dumps(dict(sample_id=m["sample_id"], scene=m["scene"], trial=m.get("trial"), view=m.get("view"),
                                        pred_grav=d.tolist(), scale_grav=s.tolist(), pred_gg=(io.S_GG @ du).tolist(),
                                        gt_gg=m["delta_gg"], err_cm=float(np.linalg.norm(io.S_GG @ du - m["delta_gg"]) * 100))) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
