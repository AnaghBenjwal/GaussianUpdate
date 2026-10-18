#!/usr/bin/env python3
"""
finalize_dataset.py — validate every sample, write index.jsonl + STATS.md for the displacement_learn dataset.

Checks per sample: all files readable and the same size, depth valid on the object, ground truth consistent
(|delta_grav| == |delta_unity|), and a geometry check: the ground-truth move projected into the crop vs. the shift
of the mask centroid (large values are normally occlusion, so this is recorded, not used to reject).
Broken samples are moved to <root>/_rejected/.

    python scripts/learn/finalize_dataset.py --root ~/Documents/displacement_learn
"""
from __future__ import annotations
import argparse, collections, json, shutil, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sample_io as io

SPLITS = ["train", "val", "test_t1_sim", "test_t2_sim"]
FILES = ["before.png", "after.png", "depth.png", "mask_before.png", "mask_after.png", "meta.json"]


def project(X, m):
    C = np.array(m["cam_pos_unity"])
    g = io.grav_rotation(m["cam_yaw"]) @ (np.asarray(X) - C)
    t = np.deg2rad(m["cam_horizon"])
    xc, yc, zc = g[0], -np.cos(t) * g[1] - np.sin(t) * g[2], -np.sin(t) * g[1] + np.cos(t) * g[2]
    fx, fy, cx, cy = m["K_crop"]
    return np.array([fx * xc / zc + cx, fy * yc / zc + cy])


def check(d: str):
    d = Path(d)
    try:
        if not all((d / f).exists() for f in FILES):
            return dict(dir=str(d), ok=False, why="missing file")
        m = json.load(open(d / "meta.json"))
        ims = {f: np.asarray(Image.open(d / f)) for f in FILES[:-1]}
        shapes = {v.shape[:2] for v in ims.values()}
        if len(shapes) != 1:
            return dict(dir=str(d), ok=False, why=f"shape mismatch {shapes}")
        mb, ma = ims["mask_before.png"] > 127, ims["mask_after.png"] > 127
        if mb.sum() == 0 or ma.sum() == 0:
            return dict(dir=str(d), ok=False, why="empty mask")
        dep = ims["depth.png"]
        if (dep[mb] > 0).mean() < 0.5:
            return dict(dir=str(d), ok=False, why="no depth on object")
        if abs(np.linalg.norm(m["delta_grav"]) - np.linalg.norm(m["delta_unity"])) > 1e-4:
            return dict(dir=str(d), ok=False, why="delta frames inconsistent")
        p0 = project(m["obj_pos_unity"], m)
        p1 = project(np.add(m["obj_pos_unity"], m["delta_unity"]), m)
        shift = np.argwhere(ma).mean(0)[::-1] - np.argwhere(mb).mean(0)[::-1]
        geo = float(np.linalg.norm((p1 - p0) - shift) / max(m.get("crop_scale", 1.0), 1e-6))
        row = dict(dir=str(d), ok=True, split=d.parent.parent.name, scene=m["scene"], sample_id=m["sample_id"],
                   object_type=m["object_type"], delta_norm_cm=round(m["delta_norm"] * 100, 2),
                   delta_grav=m["delta_grav"], px_before=m["px_before"], px_after=m["px_after"],
                   truncated=bool(m["truncated_before"] or m["truncated_after"]), view_source=m.get("view_source"),
                   config=m.get("config"), materials=m.get("materials"), lighting=m.get("lighting"),
                   trial=m.get("trial"), geo_check_px=round(geo, 2), crop=m["crop_size"])
        return row
    except Exception as e:
        return dict(dir=str(d), ok=False, why=repr(e)[:200])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path.home() / "Documents/displacement_learn"))
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    root = Path(args.root)
    dirs = [str(p.parent) for s in SPLITS for p in (root / s).glob("*/*/meta.json")]
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(check, dirs, chunksize=64))
    bad = [r for r in rows if not r["ok"]]
    for r in bad:
        dst = root / "_rejected" / Path(r["dir"]).relative_to(root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(r["dir"], dst)
    good = [r for r in rows if r["ok"]]
    with open(root / "index.jsonl", "w") as f:
        for r in good:
            r = dict(r); r["dir"] = str(Path(r["dir"]).relative_to(root)); r.pop("ok")
            f.write(json.dumps(r) + "\n")
    json.dump([dict(dir=str(Path(r["dir"]).relative_to(root)), why=r["why"]) for r in bad],
              open(root / "rejected.json", "w"), indent=1)

    L = ["# displacement_learn — dataset statistics", ""]
    L += ["| split | samples | scenes | object types | moves | median |δ| cm | median object px | truncated | geo check median px |",
          "|---|---|---|---|---|---|---|---|---|"]
    for s in SPLITS:
        R = [r for r in good if r["split"] == s]
        if not R:
            continue
        moves = {r["sample_id"].rsplit("_v", 1)[0] if r["trial"] is None else (r["scene"], r["trial"]) for r in R}
        L.append(f"| {s} | {len(R)} | {len({r['scene'] for r in R})} | {len({r['object_type'] for r in R})} | {len(moves)} | "
                 f"{np.median([r['delta_norm_cm'] for r in R]):.1f} | {np.median([r['px_before'] for r in R]):.0f} | "
                 f"{np.mean([r['truncated'] for r in R]) * 100:.0f}% | {np.median([r['geo_check_px'] for r in R]):.2f} |")
    L += ["", f"Rejected by validation: {len(bad)} ({collections.Counter(r['why'][:40] for r in bad).most_common(5)})", ""]
    for s in SPLITS:
        R = [r for r in good if r["split"] == s]
        if not R:
            continue
        L += [f"## {s}", "", "|δ| (cm) histogram: " + ", ".join(
            f"{a}-{b}: {sum(a <= r['delta_norm_cm'] < b for r in R)}" for a, b in [(0, 5), (5, 10), (10, 20), (20, 30), (30, 45)]), "",
              "object px (1600x900) quantiles 10/50/90: " + " / ".join(f"{q:.0f}" for q in np.percentile([r['px_before'] for r in R], [10, 50, 90])), "",
              "object types: " + ", ".join(f"{k} {v}" for k, v in collections.Counter(r["object_type"] for r in R).most_common()), "",
              "scenes: " + ", ".join(f"{k} {v}" for k, v in sorted(collections.Counter(r["scene"] for r in R).items())), ""]
    open(root / "STATS.md", "w").write("\n".join(L))
    print("\n".join(L[:12]))


if __name__ == "__main__":
    main()
