#!/usr/bin/env python3
"""
summarize_ablation.py — tables + plot for run_ablation.py.

    python scripts/bench/summarize_ablation.py --root ~/Documents/displacement_bench --scenes FloorPlan1 FloorPlan3

Writes <root>/summary/ablation.md and ablation_cdf.png. Paired comparisons are against rc|global (= the V2 pipeline's
displacement stage: differential render-and-compare initialised from the original RAFT estimate).
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=str(Path.home() / "Documents/displacement_bench"))
ap.add_argument("--scenes", nargs="+", default=["FloorPlan1", "FloorPlan3"])
args = ap.parse_args()
ROOT = Path(args.root); OUT = ROOT / "summary"; OUT.mkdir(exist_ok=True)
R = [json.loads(l) for s in args.scenes for fn in ("results_ablation.jsonl", "results_ablation2.jsonl")
     if (ROOT / s / fn).exists() for l in open(ROOT / s / fn)]
by = defaultdict(dict)                                    # config -> (scene, trial) -> record
for r in R:
    by[r["config"]][(r["scene"], r["trial"])] = r
keys = sorted(by["rc|global"])

# derived selection rule (no new renders — picks one of two stored records per trial):
#   sel|agree_<set>_{est,rc}: if >= 2 estimators of the set agree within 5 cm and their consensus is > 5 cm away from
#   the global RC answer, trust the consensus (est = unrefined, rc = locally refined); otherwise keep global RC.
SETS = {"all": ["raft_orig", "raft_crop", "roma", "depth", "sam", "mast3r", "mast3r_depth"],
        "new": ["raft_crop", "roma", "depth", "sam"], "match": ["raft_crop", "roma"],
        "nomast3r": ["raft_orig", "raft_crop", "roma", "depth", "sam"], "rds": ["roma", "depth", "sam"]}


def cons_group(v, thr=0.05):
    v = np.stack(v)
    D = np.linalg.norm(v[:, None, [0, 2]] - v[None, :, [0, 2]], axis=2)
    i = max(range(len(v)), key=lambda i: ((D[i] < thr).sum(), -D[i][D[i] < thr].sum()))
    m = D[i] < thr
    return np.median(v[m], 0), int(m.sum())


for k in keys:
    g = by["rc|global"][k]
    est = g["diag"]["est"]
    for name, ks in SETS.items():
        v = [np.array(est[q]) for q in ks if est.get(q) is not None]
        use = False
        if len(v) >= 2:
            c, n = cons_group(v)
            use = n >= 2 and np.hypot(*(c - np.array(g["delta_est"]))[[0, 2]]) > 0.05
        for kind in ("est", "rc"):
            src = by.get(f"{kind}|cons_{name}", {}).get(k)
            if src is not None:
                by[f"sel|agree_{name}_{kind}"][k] = dict(src if use else g, config=f"sel|agree_{name}_{kind}", used="cons" if use else "global")
BASE = "rc|global"
hard = [k for k in keys if by[BASE][k]["err_cm"] > 10]   # trials the baseline fails
f = lambda v, p=1: "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{p}f}"
pct = lambda v: "–" if v is None else f"{100 * v:.0f}%"


def stats(cfg, ks):
    rs = [by[cfg][k] for k in ks if k in by[cfg]]
    if not rs:
        return None
    e = np.array([r["err_cm"] for r in rs])
    base = np.array([by[BASE][k]["err_cm"] for k in ks if k in by[cfg]])
    nu = np.array([by["no_update"][k]["novel"]["psnr_roi"] for k in ks if k in by[cfg]])
    nov = np.array([r["novel"]["psnr_roi"] for r in rs])
    return dict(n=len(rs), avail=1 - np.mean([bool(r.get("est_none")) for r in rs]), med=float(np.median(e)),
                mean=float(e.mean()), s5=float(np.mean(e < 5)), s10=float(np.mean(e < 10)),
                better=int(np.sum(e < base - 1)), worse=int(np.sum(e > base + 1)), d_mean=float((e - base).mean()),
                nov=float(np.nanmean(nov)), gain=float(np.nanmean(nov - nu)))


def row(label, cfg, ks, avail=False):
    s = stats(cfg, ks)
    if s is None:
        return None
    a = f" {pct(s['avail'])} |" if avail else ""
    return (f"| {label} |{a} {f(s['med'])} | {f(s['mean'])} | {pct(s['s5'])} | {pct(s['s10'])} | "
            f"{s['better']} / {s['worse']} | {f(s['d_mean'])} | {f(s['nov'], 2)} | {f(s['gain'], 2)} |")


EST = [("raft_orig", "RAFT original (full image, median shift, centre → plane) — baseline matcher"),
       ("raft_crop", "RAFT dense on crop + per-match plane lift (control)"),
       ("roma", "RoMa on crop + per-match plane lift"),
       ("depth", "Depth Anything V2 (aligned), appeared-region vs old-footprint 3D median"),
       ("sam", "SAM outline on after photo, contact point → base plane"),
       ("mast3r", "MASt3R matches + MASt3R metric depth"),
       ("mast3r_depth", "MASt3R depth only (same estimator as Depth Anything)"),
       ("fuse_all", "median of all estimators")]
hdr = "| {} |{} median cm | mean cm | <5 cm | <10 cm | better / worse vs base | Δ mean err vs base | novel PSNR roi | Δ PSNR vs no update |\n|---|{}---|---|---|---|---|---|---|---|"
md = [f"# Displacement ablation — external cues vs. original RAFT matching\n",
      f"{len(keys)} trials ({', '.join(f'{s}: {sum(1 for k in keys if k[0] == s)}' for s in args.scenes)}). "
      f"Identification (V2 2-image differential RC, 100 % correct) and selection (class_clean3d) are fixed; only the "
      f"displacement stage varies. 'better / worse' counts trials where the error changes by more than 1 cm vs. "
      f"`rc|global` (the current V2 pipeline). Hard subset = the {len(hard)} trials where `rc|global` errs > 10 cm "
      f"({', '.join(sorted({by[BASE][k]['object'] for k in hard}))}).\n"]

for title, ks in [("all trials", keys), ("hard subset", hard)]:
    md.append(f"\n## A. Standalone estimators — {title}\n")
    md.append(hdr.format("estimator", " available |", "---|"))
    for k, lab in EST:
        r = row(f"{k} — {lab}", f"est|{k}", ks, avail=True)
        if r: md.append(r)

    md.append(f"\n## B. Pipelines (estimator → local render-and-compare refinement) — {title}\n")
    md.append(hdr.format("pipeline", "", ""))
    for cfg, lab in [("rc|global", "**baseline**: global RC (±50 cm grid), init {0, RAFT}"),
                     ("rc|global+depth", "global RC + mono-depth loss"),
                     ("rc|raft_orig", "RAFT original → local RC"), ("rc|raft_crop", "RAFT crop → local RC"),
                     ("rc|roma", "RoMa → local RC"), ("rc|depth", "Depth Anything → local RC"),
                     ("rc|sam", "SAM contact → local RC"), ("rc|mast3r", "MASt3R matches → local RC"),
                     ("rc|mast3r_depth", "MASt3R depth → local RC"),
                     ("rc|fuse_all", "**fusion** of all → local RC"), ("rc|fuse_all+depth", "fusion → local RC + depth loss")]:
        r = row(f"{cfg} — {lab}", cfg, ks)
        if r: md.append(r)

    md.append(f"\n## C. Leave-one-out from the fusion — {title}\n\nPositive Δ = removing the cue makes it worse (the cue helps).\n")
    md.append("| removed | median cm | mean cm | <5 cm | <10 cm | Δ mean err vs full fusion | Δ <10 cm vs full fusion |\n|---|---|---|---|---|---|---|")
    full = stats("rc|fuse_all", ks)
    for g in ["raft_orig", "roma", "depth", "sam", "mast3r"]:
        s = stats(f"rc|fuse_all-{g}", ks)
        if s and full:
            md.append(f"| {g} | {f(s['med'])} | {f(s['mean'])} | {pct(s['s5'])} | {pct(s['s10'])} | {f(s['mean'] - full['mean'])} | {100 * (full['s10'] - s['s10']):+.0f} pp |")

p2 = sorted({c for c in by if c.startswith(("rc|med_", "rc|cons_", "rc|hybrid_", "est|cons_", "est|hybrid_", "sel|agree_"))})
if p2:
    for title, ks in [("all trials", keys), ("hard subset", hard)]:
        md.append(f"\n## C2. Combination rules (pass 2: stored estimates → local RC) — {title}\n\n"
                  "med = median of the set; cons = median of the largest group agreeing within 5 cm; "
                  "hybrid = baseline unless it reports no motion (< 3 cm), then cons; "
                  "agree = cons if >= 2 estimators agree (5 cm) and disagree with the baseline by > 5 cm, else baseline; "
                  "est| = without, rc| = with the local render-and-compare refinement.\n")
        md.append(hdr.format("rule", "", ""))
        for c in [BASE, "rc|fuse_all"] + p2:
            r = row(c, c, ks)
            if r: md.append(r)

md.append("\n## D. Hole fill on top (novel-view PSNR in object region)\n\n| config | novel PSNR roi | Δ vs no update |\n|---|---|---|")
for c in ["no_update", "rc|global", "rc|global+fill", "rc|fuse_all", "rc|fuse_all+fill", "rc|fuse_all+depth", "rc|fuse_all+depth+fill"]:
    s = stats(c, keys)
    if s: md.append(f"| {c} | {f(s['nov'], 2)} | {f(s['gain'], 2)} |")

objs = sorted({(k[0], by[BASE][k]["object"]) for k in keys})
cols = ["rc|global", "est|roma", "est|depth", "est|sam", "est|mast3r_depth", "rc|fuse_all", "rc|fuse_all+depth"]
md.append("\n## E. Per object — median error cm\n\n| scene / object | n | " + " | ".join(cols) + " |\n|---|---|" + "---|" * len(cols))
for so in objs:
    ks = [k for k in keys if (k[0], by[BASE][k]["object"]) == so]
    cells = []
    for c in cols:
        e = [by[c][k]["err_cm"] for k in ks if k in by[c] and not by[c][k].get("est_none")]
        cells.append(f(float(np.median(e))) if e else "–")
    md.append(f"| {so[0]} / {so[1]} | {len(ks)} | " + " | ".join(cells) + " |")

tm = defaultdict(list)
for k in keys:
    for n, v in by[BASE][k].get("timing", {}).items():
        tm[n].append(v)
md.append("\n## F. Cost (seconds per trial, RTX 3090, mean)\n\n| step | s |\n|---|---|")
for n, v in tm.items():
    md.append(f"| {n} | {np.mean(v):.1f} |")

(OUT / "ablation.md").write_text("\n".join(md) + "\n")
print("\n".join(md))

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for c, lab in [("est|raft_orig", "RAFT original (alone)"), ("est|roma", "RoMa (alone)"), ("est|depth", "Depth Anything (alone)"),
               ("est|sam", "SAM contact (alone)"), ("est|mast3r_depth", "MASt3R depth (alone)"),
               ("rc|global", "baseline: global RC"), ("rc|fuse_all", "fusion → local RC")]:
    e = np.sort([by[c][k]["err_cm"] for k in keys if k in by[c]])
    ax[0].plot(e, np.arange(1, len(e) + 1) / len(e), label=lab, lw=2.2 if c.startswith("rc") else 1.2,
               ls="-" if c.startswith("rc") else "--")
ax[0].set_xlim(0, 30); ax[0].set_xlabel("3D displacement error (cm)"); ax[0].set_ylabel("fraction of trials")
ax[0].legend(fontsize=8); ax[0].grid(alpha=.3); ax[0].set_title("Error CDF, all trials")
g = ["raft_orig", "roma", "depth", "sam", "mast3r"]
full = stats("rc|fuse_all", keys)
ax[1].bar(g, [stats(f"rc|fuse_all-{x}", keys)["mean"] - full["mean"] for x in g])
ax[1].axhline(0, c="k", lw=.8); ax[1].set_ylabel("Δ mean error when removed (cm)")
ax[1].set_title("Leave-one-out: > 0 means the cue helps"); ax[1].grid(alpha=.3, axis="y")
plt.tight_layout(); plt.savefig(OUT / "ablation_cdf.png", dpi=130)
