#!/usr/bin/env python3
"""bootstrap_ci.py — 95 % CIs by resampling moves (T2 has several views per move) and paired differences, for T1, T2 and
the fresh-scene subset (FloorPlan2/4/5).    python scripts/learn/bootstrap_ci.py results_v3/per_input.jsonl > bootstrap.md"""
import json, numpy as np, collections, sys
R = [json.loads(l) for l in open(sys.argv[1])]
M = ["rc", "agree", "raft_crop", "roma_pnp", "inerf", "cotracker", "dispnet_v3", "dispnet_v3+rc", "agree+dispnet_v3",
     "cotracker/dispnet_v3+rc", "cotracker→dispnet_v3+rc"]
PAIRS = [("cotracker/dispnet_v3+rc", "cotracker"), ("cotracker/dispnet_v3+rc", "agree+dispnet_v3"),
         ("cotracker/dispnet_v3+rc", "agree"), ("cotracker/dispnet_v3+rc", "roma_pnp"), ("cotracker/dispnet_v3+rc", "rc"), ("cotracker/dispnet_v3+rc", "dispnet_v3+rc"), ("agree+dispnet_v3", "agree")]
rng = np.random.default_rng(0)
SETS = [("T1 (FP1/3)", lambda r: r["set"] == "test_t1"), ("T2 all (FP1-5)", lambda r: r["set"] == "test_t2"),
        ("T2 new scenes only (FP2/4/5)", lambda r: r["set"] == "test_t2" and r["scene"] in ("FloorPlan2", "FloorPlan4", "FloorPlan5"))]
for name, f in SETS:
    rows = [r for r in R if f(r)]
    moves = collections.defaultdict(list)
    for r in rows: moves[(r["scene"], r["trial"])].append(r)
    keys = list(moves); S = [[moves[keys[i]] for i in rng.integers(0, len(keys), len(keys))] for _ in range(2000)]
    print(f"\n### {name}: {len(rows)} inputs, {len(keys)} moves — 95% CI by resampling moves\n")
    print("| method | mean cm [95% CI] | median cm [95% CI] | <10 cm [95% CI] |\n|---|---|---|---|")
    for m in M:
        e = np.array([r["err_cm"][m] for r in rows])
        st = np.array([(np.mean(x), np.median(x), (x < 10).mean() * 100) for x in
                       (np.array([r["err_cm"][m] for g in s for r in g]) for s in S)])
        lo, hi = np.percentile(st, [2.5, 97.5], 0)
        print(f"| {m} | {e.mean():.2f} [{lo[0]:.2f}, {hi[0]:.2f}] | {np.median(e):.2f} [{lo[1]:.2f}, {hi[1]:.2f}] | {100*(e<10).mean():.1f}% [{lo[2]:.0f}, {hi[2]:.0f}] |")
    print()
    for a, b in PAIRS:
        d = [np.mean([r["err_cm"][a] - r["err_cm"][b] for g in s for r in g]) for s in S]
        print(f"- paired mean diff {a} − {b}: {np.mean([r['err_cm'][a] - r['err_cm'][b] for r in rows]):+.2f} cm, 95% CI [{np.percentile(d, 2.5):+.2f}, {np.percentile(d, 97.5):+.2f}]")
