#!/usr/bin/env python3
"""bootstrap_ci.py — 95 % CIs by resampling moves (T2 has several views per move) + DispNet runtime. Paths are the Vast layout."""
import json, numpy as np, collections, sys, time, torch
sys.path.insert(0, "/root/Documents/gaussian-grouping/scripts/learn")
R = [json.loads(l) for l in open("/root/Documents/displacement_learn/results/per_input.jsonl")]
M = ["P1", "P2", "raft_crop", "roma", "depth", "sam", "rc", "agree", "dispnet_v1", "dispnet_v2", "dispnet_v2+rc", "agree+dispnet_v2", "agree→dispnet_v2+rc"]
rng = np.random.default_rng(0)
for st in ("test_t1", "test_t2"):
    rows = [r for r in R if r["set"] == st]
    moves = collections.defaultdict(list)
    for r in rows: moves[(r["scene"], r["trial"])].append(r)
    keys = list(moves)
    print(f"== {st}: 95% CI by resampling moves (n_moves={len(keys)})  [mean cm | <10cm | median]")
    for m in M:
        stats = []
        for _ in range(2000):
            s = [moves[keys[i]] for i in rng.integers(0, len(keys), len(keys))]
            e = np.array([r["err_cm"][m] for g in s for r in g])
            stats.append((e.mean(), (e < 10).mean() * 100, np.median(e)))
        a = np.array(stats); lo, hi = np.percentile(a, [2.5, 97.5], 0)
        e = np.array([r["err_cm"][m] for r in rows])
        print(f"  {m:22s} mean {e.mean():5.1f} [{lo[0]:5.1f},{hi[0]:5.1f}]  <10 {100*(e<10).mean():4.0f}% [{lo[1]:3.0f},{hi[1]:3.0f}]  median {np.median(e):4.1f} [{lo[2]:.1f},{hi[2]:.1f}]")
    # paired: agree+dispnet_v2 - agree
    for a_, b_ in (("agree+dispnet_v2", "agree"), ("dispnet_v2", "rc"), ("dispnet_v2", "dispnet_v1")):
        d = []
        for _ in range(2000):
            s = [moves[keys[i]] for i in rng.integers(0, len(keys), len(keys))]
            d.append(np.mean([r["err_cm"][a_] - r["err_cm"][b_] for g in s for r in g]))
        print(f"  paired mean diff {a_} - {b_}: {np.mean(d):+.2f} cm  95% CI [{np.percentile(d,2.5):+.2f}, {np.percentile(d,97.5):+.2f}]")
# DispNet runtime
from data import DispDataset; from model import DispNet
ds = DispDataset(["/root/Documents/displacement_learn/test_t2_gg"], norm="crop")
net = DispNet(pretrained=False).cuda().eval()
t = time.time(); xs = [ds[i] for i in range(50)]; tl = (time.time() - t) / 50
with torch.no_grad():
    for b in xs[:5]: net(b["x"][None].cuda(), b["g"][None].cuda())
    torch.cuda.synchronize(); t = time.time()
    for b in xs: net(b["x"][None].cuda(), b["g"][None].cuda())
    torch.cuda.synchronize(); tn = (time.time() - t) / 50
print(f"DispNet: input building {tl*1000:.0f} ms (CPU), network {tn*1000:.1f} ms (GPU, batch 1)")
