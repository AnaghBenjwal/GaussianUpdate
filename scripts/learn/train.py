#!/usr/bin/env python3
"""
train.py — train DispNet on displacement_learn/train, select on val, write checkpoints + log.

    python scripts/learn/train.py --data ~/Documents/displacement_learn --out ~/Documents/displacement_learn/runs/r18
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import DispDataset
from model import DispNet, laplace_nll


def evaluate(model, loader, dev):
    model.eval()
    E, B = [], []
    with torch.no_grad():
        for b in loader:
            mu, logb = model(b["x"].to(dev, non_blocking=True), b["g"].to(dev))
            E.append((mu.float().cpu() - b["y"]).norm(dim=1)); B.append(logb.float().exp().cpu())
    e = torch.cat(E).numpy() * 100
    return dict(median_cm=float(np.median(e)), mean_cm=float(e.mean()), lt5=float((e < 5).mean()), lt10=float((e < 10).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path.home() / "Documents/displacement_learn"))
    ap.add_argument("--train", default="train")
    ap.add_argument("--val", default="val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--norm", default="crop", choices=["crop", "fixed"])
    ap.add_argument("--limit", type=int, default=0, help="debug: use only N training samples")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out / "args.json", "w"), indent=1)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    tr = DispDataset([Path(args.data) / s for s in args.train.split(",")], train=True, norm=args.norm)
    va = DispDataset([Path(args.data) / s for s in args.val.split(",")], train=False, norm=args.norm)
    if args.limit:
        tr.dirs = tr.dirs[:args.limit]; va.dirs = va.dirs[:max(8, args.limit // 4)]
    print(f"train {len(tr)}  val {len(va)}", flush=True)
    mk = lambda ds, sh: torch.utils.data.DataLoader(ds, batch_size=args.bs, shuffle=sh, num_workers=args.workers,
                                                    pin_memory=dev == "cuda", drop_last=sh, persistent_workers=False)
    vl = mk(va, False)
    model = DispNet(norm=args.norm).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * max(1, len(tr) // args.bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, s / 500) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    best = math.inf
    log = open(out / "log.jsonl", "a")
    for ep in range(args.epochs):
        tr.epoch = ep
        model.train()
        t0, losses, errs = time.time(), [], []
        for b in mk(tr, True):
            x, g, y = b["x"].to(dev, non_blocking=True), b["g"].to(dev), b["y"].to(dev)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                mu, logb = model(x, g)
            loss = laplace_nll(mu.float(), logb.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step()
            losses.append(loss.item()); errs.append((mu.detach().float() - y).norm(dim=1).mean().item() * 100)
        v = evaluate(model, vl, dev)
        rec = dict(epoch=ep, loss=float(np.mean(losses)), train_err_cm=float(np.mean(errs)), time_s=time.time() - t0,
                   lr=sched.get_last_lr()[0], **{f"val_{k}": x for k, x in v.items()})
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(json.dumps(rec), flush=True)
        torch.save(model.state_dict(), out / "last.pt")
        if v["mean_cm"] < best:
            best = v["mean_cm"]; torch.save(model.state_dict(), out / "best.pt")
    print(f"best val mean {best:.2f} cm", flush=True)


if __name__ == "__main__":
    main()
