#!/usr/bin/env python3
"""
summarize_eval.py — final comparison tables: every method on T1 and T2, scored per (trial, input view).

Reads <bench>/<scene>/eval_cues.jsonl and eval_learned_<NAME>.jsonl (eval_methods.py) and, for the learned model on
simulator inputs, runs/<run>/pred_test_t{1,2}_sim.jsonl (predict.py). Writes <out>/tables.md and <out>/per_input.jsonl.

    python scripts/learn/summarize_eval.py --data ~/Documents/displacement_learn --runs r18 r18crop --out results
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(Path.home() / "Documents/displacement_learn"))
ap.add_argument("--benches", nargs=2, default=[str(Path.home() / "Documents/displacement_bench"),
                                                str(Path.home() / "Documents/displacement_bench_t2")])
ap.add_argument("--runs", nargs="+", default=["r18"])
ap.add_argument("--labels", nargs="*", default=None, help="display name per run")
ap.add_argument("--out", required=True)
args = ap.parse_args()
OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
LAB = dict(zip(args.runs, args.labels or args.runs))

DESC = {
    "no_update": "predict zero motion",
    "P1": "current pipeline A: template-matched 2D shift, constant-depth back-projection",
    "P2": "current pipeline B: RAFT 2D shift, constant-depth back-projection",
    "raft_orig": "RAFT median shift + support-plane cast",
    "raft_crop": "RAFT dense flow on a crop + per-match plane lifting + RANSAC",
    "roma": "RoMa matches on the crop + plane lifting + RANSAC",
    "depth": "Depth Anything V2 (aligned to GG depth), appeared-region centroid",
    "sam": "SAM outline of the appeared object, contact point on the base plane",
    "rc": "global differential render-and-compare (V2 baseline)",
    "agree": "ablation winner: consensus of {raft_crop, roma, depth, sam} overrides rc when they disagree > 5 cm",
    "inerf": "[lit] iNeRF (IROS 2021): gradient descent on the object translation through the GG renderer, photometric loss",
    "lg_pnp": "[lit] DISK+LightGlue (ICCV 2023) matches + GG depth + PnP-RANSAC (pose step of 3DGS-CD, RA-L 2025)",
    "roma_pnp": "[lit] RoMa (CVPR 2024) matches + GG depth + PnP-RANSAC",
    "cotracker": "[lit] CoTracker3 (2024) point tracks + support-plane lifting + RANSAC",
    "icp": "[lit] point-to-point ICP: GG-depth object points vs SAM mask + Depth Anything V2 points of the AFTER photo",
    "mast3r": "[lit] MASt3R (ECCV 2024): metric depth (scaled to GG) + object matches, 3D differences + RANSAC",
}
LIT = ["inerf", "lg_pnp", "roma_pnp", "cotracker", "icp", "mast3r"]
HAND = ["no_update", "P1", "P2"] + LIT + ["raft_orig", "raft_crop", "roma", "depth", "sam", "rc", "agree"]


def load_set(tag, bench):
    """-> {(scene, trial, view): {method: delta or None}}, gt, meta"""
    E, GT, META = defaultdict(dict), {}, {}
    for sd in sorted(Path(bench).iterdir()):
        cp = sd / "eval_cues.jsonl"
        if not cp.exists():
            continue
        for l in open(cp):
            r = json.loads(l); k = (r["scene"], r["trial"], r["view"])
            GT[k] = np.array(r["gt_delta"]); META[k] = dict(object=r["object"], gt_cm=r["gt_norm_m"] * 100, diag=r["diag"],
                                                            timing=r["timing"])
            for m, e in r["est"].items():
                E[k][m] = e
        bp = sd / "eval_baselines.jsonl"
        if bp.exists():
            for l in open(bp):
                r = json.loads(l); k = (r["scene"], r["trial"], r["view"])
                E[k].update(r["est"])
                META.setdefault(k, {}).setdefault("timing", {}).update(r["timing"])
        for run in args.runs:
            lp = sd / f"eval_learned_{run}.jsonl"
            if lp.exists():
                for l in open(lp):
                    r = json.loads(l); k = (r["scene"], r["trial"], r["view"])
                    for m, e in r["est"].items():
                        E[k][m.replace(run, LAB[run])] = e
                    META[k][f"unc_{LAB[run]}"] = r["unc_cm"]
    for k in E:                                               # derived rule (fixed before looking at T2)
        for run in args.runs:
            L_ = LAB[run]
            if E[k].get(f"{L_}+rc") is not None and E[k].get("agree") is not None:
                a, d = np.asarray(E[k]["agree"]), np.asarray(E[k][L_])
                E[k][f"agree→{L_}+rc"] = E[k][f"{L_}+rc"] if np.hypot(*(a - d)[[0, 2]]) > 0.10 else E[k]["agree"]
            if E[k].get(f"{L_}+rc") is not None and "cotracker" in E[k]:  # added after the T2 baseline run, same 10 cm
                c, d = E[k]["cotracker"], np.asarray(E[k][L_])
                E[k][f"cotracker/{L_}+rc"] = E[k][f"{L_}+rc"] if c is None else c
                E[k][f"cotracker→{L_}+rc"] = (E[k][f"{L_}+rc"] if c is None or np.hypot(*(np.asarray(c) - d)[[0, 2]]) > 0.10
                                              else c)
    for run in args.runs:                                     # learned model on simulator inputs (no GG)
        pp = Path(args.data) / "runs" / run / f"pred_{tag}_sim.jsonl"
        if pp.exists():
            for l in open(pp):
                p = json.loads(l); k = (p["scene"], p["trial"], p["view"])
                if k in E:
                    E[k][f"{LAB[run]} (sim inputs)"] = p["pred_gg"]
    return E, GT, META


def errs(E, GT, m, keys):
    out = []
    for k in keys:
        e = E[k].get(m, "missing")
        if e == "missing":
            return None
        d = np.zeros(3) if e is None else np.asarray(e)            # a cue that gives up = no move
        out.append(np.linalg.norm(d - GT[k]) * 100)
    return np.array(out)


def fmt_row(name, e, n_none=None):
    if e is None or len(e) == 0:
        return None
    s = f"| {name} | {len(e)} | {np.median(e):.1f} | {e.mean():.1f} | {np.percentile(e, 90):.1f} | " \
        f"{(e < 2).mean() * 100:.0f}% | {(e < 5).mean() * 100:.0f}% | {(e < 10).mean() * 100:.0f}% |"
    return s + (f" {n_none} |" if n_none is not None else " |")


HDR = ["| method | n | median cm | mean cm | p90 cm | <2 cm | <5 cm | <10 cm | no answer |", "|---|---|---|---|---|---|---|---|---|"]


def methods_of(E):
    ms = []
    for k in E:
        for m in E[k]:
            if m not in ms:
                ms.append(m)
    hand = [m for m in HAND if m in ms]
    return hand + sorted(m for m in ms if m not in hand)


L = ["# Displacement estimation — all methods on the held-out test sets", "",
     "Error = ‖δ̂ − δ‖ (3D, cm) per (trial, input view). Inputs: real BEFORE + AFTER photo of one view, GG model of the "
     "BEFORE scene, ground-truth class (2-image identification is 100 % on T1). A cue that returns nothing counts as "
     "zero motion ('no answer' column). Learned rows use GG-rendered depth + mask unless marked (sim inputs).", ""]
L += ["Methods:", ""] + [f"- **{k}**: {v}" for k, v in DESC.items()] + [
    "- **<model>**: DispNet prediction; **<model>+rc**: local render-and-compare (±8 cm) started at it; "
    "**agree+<model>**: agree rule with DispNet as a fifth cue; **gate_<model>**: DispNet when its predicted "
    "uncertainty is < 3 cm, else agree; **agree→<model>+rc**: agree, unless DispNet disagrees with it by > 10 cm "
    "(planar), then <model>+rc; **cotracker/<model>+rc**: CoTracker3, or <model>+rc when it gives no answer; "
    "**cotracker→<model>+rc**: CoTracker3, unless it gives no answer or disagrees with DispNet by > 10 cm (planar), "
    "then <model>+rc (both added after the T2 baseline run; threshold reused, not tuned).", ""]
rows_all = []
for tag, bench, title in (("test_t1", args.benches[0], "T1 — original benchmark (37 moves, 1 input view each; methods were tuned on it)"),
                          ("test_t2", args.benches[1], "T2 — new held-out set (67 random moves, every visible view)")):
    E, GT, META = load_set(tag, bench)
    keys = sorted(k for k in E if "rc" in E[k])
    if not keys:
        continue
    ms = methods_of(E)
    L += [f"## {title}", "", f"{len(keys)} (trial, view) inputs, {len({k[:2] for k in keys})} moves, "
          f"median |δ| {np.median([META[k]['gt_cm'] for k in keys]):.1f} cm", ""] + HDR
    for m in ms:
        e = errs(E, GT, m, keys)
        nn = sum(E[k].get(m) is None for k in keys)
        r = fmt_row(m, e, nn)
        if r:
            L.append(r)
    # breakdowns for the main methods
    main = [m for m in ms if m in ("P1", "P2", "rc", "agree") or m in LIT or not m in HAND and "sim" not in m and "_v1" not in m and "gate" not in m]
    L += ["", f"### {tag}: median error (cm) by move size", "",
          "| method | " + " | ".join(f"{a}-{b} cm" for a, b in [(0, 5), (5, 10), (10, 20), (20, 45)]) + " |",
          "|---|" + "---|" * 4]
    g = np.array([META[k]["gt_cm"] for k in keys])
    for m in main:
        e = errs(E, GT, m, keys)
        if e is None:
            continue
        cells = [f"{np.median(e[(g >= a) & (g < b)]):.1f} (n={((g >= a) & (g < b)).sum()})" if ((g >= a) & (g < b)).any() else "–"
                 for a, b in [(0, 5), (5, 10), (10, 20), (20, 45)]]
        L.append(f"| {m} | " + " | ".join(cells) + " |")
    # where the baseline fails
    rc_e = errs(E, GT, "rc", keys); ag_e = errs(E, GT, "agree", keys)
    hard = rc_e > 10
    L += ["", f"### {tag}: inputs where global render-and-compare fails (rc error > 10 cm, n={hard.sum()})", ""] + HDR
    for m in main:
        e = errs(E, GT, m, keys)
        if e is not None and hard.any():
            L.append(fmt_row(m, e[hard], sum(E[k].get(m) is None for k, h in zip(keys, hard) if h)))
    # per object
    objs = sorted({META[k]["object"] for k in keys})
    L += ["", f"### {tag}: median error (cm) per object", "", "| object | n | " + " | ".join(main) + " |", "|---|---|" + "---|" * len(main)]
    for o in objs:
        ks = [k for k in keys if META[k]["object"] == o]
        cells = []
        for m in main:
            e = errs(E, GT, m, ks)
            cells.append("–" if e is None else f"{np.median(e):.1f}")
        L.append(f"| {o} | {len(ks)} | " + " | ".join(cells) + " |")
    T = defaultdict(list)
    for k in keys:
        for a, b in META[k]["timing"].items():
            T[a].append(b)
    L += ["", f"Mean runtime per input (s, RTX 3090, shared GPU): " + ", ".join(f"{a} {np.mean(b):.2f}" for a, b in T.items()), ""]
    for k in keys:
        rows_all.append(dict(set=tag, scene=k[0], trial=k[1], view=k[2], object=META[k]["object"], gt_cm=META[k]["gt_cm"],
                             err_cm={m: (None if (e := errs(E, GT, m, [k])) is None else round(float(e[0]), 2)) for m in ms}))
# ── rendering metrics (eval_render.py) ──────────────────────────────────────────
RK = ("psnr_roi", "ssim_roi", "lpips_roi", "iou")
for tag, bench in (("T1", args.benches[0]), ("T2", args.benches[1])):
    rows = [json.loads(l) for sd in sorted(Path(bench).iterdir()) if (sd / "eval_render.jsonl").exists()
            for l in open(sd / "eval_render.jsonl")]
    if not rows:
        continue
    by = defaultdict(dict)
    for r in rows:
        m = r["method"]
        for run in sorted(args.runs, key=len, reverse=True):     # "r18crop" before "r18"
            if run in m:
                m = m.replace(run, LAB[run]); break
        by[m][(r["scene"], r["trial"], r["view"])] = r
    E, _, _ = load_set("test_" + tag.lower(), bench)          # derived rules pick one component per input → reuse its row
    for k, est in E.items():
        for dm, d in est.items():
            if not ("→" in dm or "/" in dm) or d is None or k in by[dm]:
                continue
            for comp in ("cotracker", "agree") + tuple(f"{LAB[r]}+rc" for r in args.runs):
                if est.get(comp) is not None and np.allclose(est[comp], d) and k in by.get(comp, {}):
                    by[dm][k] = dict(by[comp][k], method=dm); break
    base = [m for m in by if by[m] and not ("→" in m or "/" in m)]
    common = set.intersection(*(set(by[m]) for m in base))
    by = {m: v for m, v in by.items() if m in base or (v and common <= set(v))}   # derived rows only if complete
    order = ["gt"] + [m for m in HAND if m in by] + sorted(m for m in by if m not in HAND and m != "gt")
    L += ["", f"## Rendering metrics — {tag} ({len(common)} inputs rendered by every method)", "",
          "The object's Gaussians are moved by the estimate and every view is rendered and compared with the real AFTER "
          "photo. ROI = dilated union of the true BEFORE/AFTER object masks; LPIPS on the ROI bounding box; IoU = rendered "
          "visible silhouette vs. true AFTER mask. 'novel' = mean over the other views that see the object. "
          "gt = true displacement (ceiling for this GG model + selection). ↑ higher is better, ↓ lower is better.", "",
          "| method | err median cm | input PSNR-ROI ↑ | input SSIM-ROI ↑ | input LPIPS-ROI ↓ | input IoU ↑ | "
          "novel PSNR-ROI ↑ | novel SSIM-ROI ↑ | novel LPIPS-ROI ↓ | novel IoU ↑ | novel PSNR full ↑ |",
          "|---|" + "---|" * 10]
    for m in order:
        R = [by[m][k] for k in sorted(common)]
        mean = lambda part, key: np.nanmean([np.nan if r[part].get(key) is None else r[part][key] for r in R])
        L.append(f"| {m} | {np.median([r['err_cm'] for r in R]):.1f} | " +
                 " | ".join(f"{mean('input', k):.3f}" if k != "psnr_roi" else f"{mean('input', k):.2f}" for k in RK) + " | " +
                 " | ".join(f"{mean('novel', k):.3f}" if k != "psnr_roi" else f"{mean('novel', k):.2f}" for k in RK) +
                 f" | {mean('novel', 'psnr'):.2f} |")
open(OUT / "tables.md", "w").write("\n".join(L))
with open(OUT / "per_input.jsonl", "w") as f:
    for r in rows_all:
        f.write(json.dumps(r) + "\n")
print("\n".join(L))
