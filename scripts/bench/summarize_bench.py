#!/usr/bin/env python3
"""
summarize_bench.py — tables + plots from displacement_bench/<scene>/results.jsonl (gaussian_grouping env).

    conda run -n gaussian_grouping python scripts/bench/summarize_bench.py --scenes FloorPlan1 FloorPlan3
Writes ~/Documents/displacement_bench/summary/{tables.md, summary.json, *.png}
"""
import argparse, json
from collections import defaultdict
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--scenes", nargs="+", default=["FloorPlan1", "FloorPlan3"])
args = ap.parse_args()
BENCH = Path.home() / "Documents/displacement_bench"
OUT = BENCH / "summary"; OUT.mkdir(exist_ok=True)
R = [json.loads(l) for s in args.scenes for fn in ("results.jsonl", "results_v2.jsonl") if (BENCH / s / fn).exists()
     for l in open(BENCH / s / fn)]
# paired quality gain: ROI PSNR minus the no-update ROI PSNR of the same trial/view (cancels the model-vs-reality gap)
_base = {(r["scene"], r["trial"]): r for r in R if r["config"] == "no_update"}
for r in R:
    b = _base.get((r["scene"], r["trial"]))
    if b:
        r["gain_in_roi"] = r["input"].get("psnr_roi", np.nan) - b["input"].get("psnr_roi", np.nan)
        r["gain_nov_roi"] = (r["novel"].get("psnr_roi") or np.nan) - (b["novel"].get("psnr_roi") or np.nan)
trials = sorted({(r["scene"], r["trial"]) for r in R})
by = defaultdict(list)
for r in R:
    by[r["config"]].append(r)

# move direction w.r.t. the input camera: lateral (mostly parallel to image x) vs depth (towards/away from camera)
views = {s: json.load(open(BENCH / s / "views.json")) for s in args.scenes}
def direction(r):
    yaw = np.deg2rad(views[r["scene"]]["views"][0]["yaw"])              # unity yaw: 0 = +z, 90 = +x
    fwd_gg = np.array([np.sin(yaw), 0, -np.cos(yaw)])                   # unity forward (sin, 0, cos) -> gg (z negated)
    d = np.array(r["gt_delta"]); d = d / (np.linalg.norm(d) + 1e-9)
    return "depth" if abs(d @ fwd_gg) > np.cos(np.deg2rad(45)) else "lateral"


def stats(rs):
    m = lambda k, sub=None: (lambda v: float(np.mean(v)) if v else None)([((x[sub] or {}).get(k) if sub else x.get(k)) for x in rs
                                                                          if ((x[sub] or {}).get(k) if sub else x.get(k)) is not None])
    e = [x["err_cm"] for x in rs if "err_cm" in x]
    o = {"n": len(rs),
         "id_acc": (lambda v: float(np.mean(v)) if v else None)([x["id_correct"] for x in rs if x.get("id_correct") is not None]),
         "f1": m("f1"), "precision": m("precision"), "recall": m("recall"),
         "err_med_cm": float(np.median(e)) if e else None, "err_mean_cm": float(np.mean(e)) if e else None,
         "succ5": float(np.mean([x < 5 for x in e])) if e else None, "succ10": float(np.mean([x < 10 for x in e])) if e else None,
         "vert_mean_cm": m("err_vertical_cm"), "gain_in_roi": m("gain_in_roi"), "gain_nov_roi": m("gain_nov_roi"),
         "in_psnr": m("psnr", "input"), "in_psnr_roi": m("psnr_roi", "input"), "in_ssim_roi": m("ssim_roi", "input"),
         "nov_psnr": m("psnr", "novel"), "nov_psnr_roi": m("psnr_roi", "novel"), "nov_ssim_roi": m("ssim_roi", "novel")}
    return o


f = lambda v, p=2: "–" if v is None else (f"{v:.{p}f}")
pct = lambda v: "–" if v is None else f"{100 * v:.0f}%"
S = {c: stats(rs) for c, rs in by.items()}
md = [f"# Object-displacement benchmark\n\n{len(trials)} trials ({', '.join(f'{s}: {sum(1 for t in trials if t[0] == s)}' for s in args.scenes)}), "
      f"input = one after-image (frame_00000); novel = held-out test views. ROI = moved object's before ∪ after footprint (dilated).\n"]

md.append("\n## 1. Identification accuracy (which object moved)\n\n| method | accuracy |\n|---|---|")
for k in ("pixeldiff", "dino", "render_compare"):
    v = [r["ids"][k] == r["class_idx"] for r in by["no_update"]]
    md.append(f"| 1-image: {k} | {pct(np.mean(v))} ({sum(v)}/{len(v)}) |")
_v2 = {(r["scene"], r["trial"]): r["ids"] for r in R if "2img_frac" in r.get("ids", {})}
for k in ("2img_frac", "2img_diffrc", "2img_drcm", "2img_sil"):
    v = [_v2[(r["scene"], r["trial"])].get(k) == r["class_idx"] for r in by["no_update"] if (r["scene"], r["trial"]) in _v2]
    if v:
        md.append(f"| 2-image: {k} | {pct(np.mean(v))} ({sum(v)}/{len(v)}) |")

md.append("\n## 2. Stage isolation — GG, oracle object id\n\n| selection | displacement | sel F1 | err median cm | succ@5cm | succ@10cm | vertical err cm | input PSNR roi | novel PSNR roi | novel SSIM roi | novel PSNR |\n|---|---|---|---|---|---|---|---|---|---|---|")
for c in sorted(k for k in S if k.startswith("oracleID|")):
    _, sel, dsp, app = c.split("|"); s = S[c]
    md.append(f"| {sel} | {dsp}{' + fill' if 'fill' in app else ''} | {f(s['f1'])} | {f(s['err_med_cm'], 1)} | {pct(s['succ5'])} | {pct(s['succ10'])} | {f(s['vert_mean_cm'], 1)} | {f(s['in_psnr_roi'])} | {f(s['nov_psnr_roi'])} | {f(s['nov_ssim_roi'], 3)} | {f(s['nov_psnr'])} |")

pipes = ["no_update", "P1_current_pixeldiff_template", "P2_current_dino_raft", "P3_dino_clean3d_raft_plane", "P4_render_compare",
         "P5_render_compare+fill", "V1_2img_frac_diffrc", "V1_2img_frac_diffrc+fill", "V2_2img_diffrc", "V2_2img_diffrc+fill",
         "V3_2img_drcm", "V3_2img_drcm+fill", "V4_2img_sil_drcm", "V4_2img_sil_drcm+fill",
         "oracleID|class_clean3d|gt|translate", "oracleID|class_clean3d|gt|translate+fill", "oracleID|aabb_oracle|gt|translate+fill"]
md.append("\n## 3. Full pipelines (no oracle) vs. bounds\n\n| pipeline | id acc | sel F1 | err median cm | err mean cm | succ@5cm | succ@10cm | input PSNR roi | novel PSNR roi | Δ input roi | Δ novel roi | novel SSIM roi | novel PSNR |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for c in pipes:
    if c in S:
        s = S[c]
        md.append(f"| {c} | {pct(s['id_acc'])} | {f(s['f1'])} | {f(s['err_med_cm'], 1)} | {f(s['err_mean_cm'], 1)} | {pct(s['succ5'])} | {pct(s['succ10'])} | {f(s['in_psnr_roi'])} | {f(s['nov_psnr_roi'])} | {f(s['gain_in_roi'])} | {f(s['gain_nov_roi'])} | {f(s['nov_ssim_roi'], 3)} | {f(s['nov_psnr'])} |")

md.append("\n## 4. Error by move direction relative to the input camera (median cm)\n\n| pipeline | lateral | depth (towards/away) |\n|---|---|---|")
for c in ["P1_current_pixeldiff_template", "P2_current_dino_raft", "P3_dino_clean3d_raft_plane", "P4_render_compare",
          "V2_2img_diffrc", "V3_2img_drcm", "V4_2img_sil_drcm", "oracleID|class_clean3d|diffrc|translate",
          "oracleID|class_clean3d|sil|translate", "oracleID|class_clean3d|drcm|translate",
          "oracleID|class|raft_constdepth|translate", "oracleID|class|raft_plane|translate"]:
    if c in by:
        g = defaultdict(list)
        for r in by[c]:
            if "err_cm" in r:
                g[direction(r)].append(r["err_cm"])
        md.append(f"| {c} | {f(np.median(g['lateral']) if g['lateral'] else None, 1)} (n={len(g['lateral'])}) | {f(np.median(g['depth']) if g['depth'] else None, 1)} (n={len(g['depth'])}) |")

ls_cfgs = sorted(k for k in S if k.startswith("LS|"))
if ls_cfgs:
    md.append("\n## 5. GG vs LangSplat (oracle object: GT class for GG, object-name text query for LangSplat)\n\n| model | selection | displacement | sel F1 | precision | recall | err median cm | input PSNR roi | novel PSNR roi | novel SSIM roi |\n|---|---|---|---|---|---|---|---|---|---|")
    for c in ["no_update", "oracleID|class_clean3d|gt|translate", "oracleID|class_clean3d|raft_plane|translate", "oracleID|class_clean3d|render_compare|translate",
              "oracleID|aabb_oracle|gt|translate"] + ls_cfgs:
        if c not in S:
            continue
        s = S[c]; parts = c.split("|")
        model = "LangSplat" if c.startswith("LS") else "GG"
        sel = parts[1] if len(parts) > 1 else "–"; dsp = parts[2] if len(parts) > 2 else "none"
        md.append(f"| {model} | {sel} | {dsp} | {f(s['f1'])} | {f(s['precision'])} | {f(s['recall'])} | {f(s['err_med_cm'], 1)} | {f(s['in_psnr_roi'])} | {f(s['nov_psnr_roi'])} | {f(s['nov_ssim_roi'], 3)} |")

md.append("\n## 6. Per scene — full pipelines (novel PSNR roi / median err cm / succ@10cm)\n")
md.append("| pipeline | " + " | ".join(args.scenes) + " |\n|---|" + "---|" * len(args.scenes))
for c in pipes:
    cells = []
    for sc in args.scenes:
        rs = [r for r in by.get(c, []) if r["scene"] == sc]
        s = stats(rs) if rs else None
        cells.append("–" if not s else f"{f(s['nov_psnr_roi'])} / {f(s['err_med_cm'], 1)} / {pct(s['succ10'])}")
    md.append(f"| {c} | " + " | ".join(cells) + " |")

md.append("\n## 7. Per object — median err cm (identification correct / n)\n")
_pc = ["P4_render_compare", "V2_2img_diffrc", "V3_2img_drcm", "V4_2img_sil_drcm", "oracleID|class_clean3d|drcm|translate"]
md.append("| scene / object | " + " | ".join(_pc) + " |\n|---|" + "---|" * len(_pc))
for so in sorted({(r["scene"], r["object"]) for r in by["no_update"]}):
    cells = []
    for c in _pc:
        rs = [r for r in by.get(c, []) if (r["scene"], r["object"]) == so and "err_cm" in r]
        cells.append("–" if not rs else f"{np.median([r['err_cm'] for r in rs]):.1f} ({sum(bool(r.get('id_correct')) for r in rs)}/{len(rs)})")
    md.append(f"| {so[0]} / {so[1]} | " + " | ".join(cells) + " |")

(OUT / "tables.md").write_text("\n".join(md) + "\n")
json.dump(S, open(OUT / "summary.json", "w"), indent=1)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for c, lab in [("P1_current_pixeldiff_template", "P1 current (pixel-diff + template)"), ("P2_current_dino_raft", "P2 current (DINO + RAFT)"),
               ("P3_dino_clean3d_raft_plane", "P3 support-plane"), ("P4_render_compare", "P4 render-and-compare"),
               ("V2_2img_diffrc", "V2 2-img differential RC"), ("V3_2img_drcm", "V3 2-img masked diff. RC")]:
    e = np.sort([r["err_cm"] for r in by.get(c, []) if "err_cm" in r])
    if len(e):
        ax[0].plot(e, np.arange(1, len(e) + 1) / len(e), label=lab)
ax[0].set_xlim(0, 60); ax[0].set_xlabel("3D displacement error (cm)"); ax[0].set_ylabel("fraction of trials"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[0].set_title("Displacement error CDF")
lab = [c for c in pipes if c in S]
ax[1].barh(range(len(lab)), [S[c]["nov_psnr_roi"] or 0 for c in lab]); ax[1].set_yticks(range(len(lab)))
ax[1].set_yticklabels([c.replace("oracleID|", "oracle: ").replace("|translate", "") for c in lab], fontsize=8); ax[1].invert_yaxis()
ax[1].set_xlabel("novel-view PSNR in object region (dB)"); ax[1].set_title("Update quality on held-out views")
lo = min(S[c]["nov_psnr_roi"] or 99 for c in lab); ax[1].set_xlim(lo - 2, None); ax[1].grid(alpha=.3, axis="x")
plt.tight_layout(); plt.savefig(OUT / "overview.png", dpi=130)
print("\n".join(md))
