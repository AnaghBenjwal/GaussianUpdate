#!/usr/bin/env python3
"""
collect_scene_and_trials.py
Run in dev conda env:
    conda run -n dev python scripts/sim/collect_scene_and_trials.py --scene FloorPlan2
    conda run -n dev python scripts/sim/collect_scene_and_trials.py --scene FloorPlan3

Outputs to --out/<scene>/:
    env/images/         RGB training frames
    env/object_mask/    class-ID masks
    env/sparse/0/       COLMAP binary model
    env/scene_meta.json
    trials/<name>/      before.png, after.png, meta.json  (5 trials)
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "sim"))
from sim.collect_data import collect
from sim.before_after import render_before_after

# displacement table: try each in order until we get a visible shift
TRIAL_DX_OPTIONS = [
    ( 0.25,  0.00),   # right
    (-0.20,  0.00),   # left
    ( 0.00,  0.20),   # forward
    ( 0.00, -0.20),   # back
    ( 0.18,  0.18),   # diagonal
]
MIN_PIXEL_SHIFT = 5.0   # px — skip trials where object barely moved
NUM_TRIALS      = 5


def collect_trials(scene: str, out_root: Path, meta: dict) -> list[dict]:
    """Try each moveable object in turn; collect NUM_TRIALS visible displacements."""
    trials_dir = out_root / scene / "trials"
    objects = meta.get("objects", [])
    print(f"\n── Collecting trials for {scene} ({len(objects)} candidates) ──")

    collected: list[dict] = []
    attempted_objs = 0

    for obj_info in objects:
        if len(collected) >= NUM_TRIALS:
            break
        obj_idx   = objects.index(obj_info)
        obj_name  = obj_info["object_id"].split("|")[0].lower()
        attempted_objs += 1

        # Pick a displacement that gives a measurable pixel shift
        saved = None
        for dx, dz in TRIAL_DX_OPTIONS:
            out_dir = trials_dir / obj_name
            try:
                result = render_before_after(
                    scene=scene,
                    scene_meta=meta,
                    object_idx=obj_idx,
                    dx=dx, dz=dz,
                    out_dir=out_dir,
                )
                gt_dx = result.get("gt_pixel_dx") or 0
                gt_dy = result.get("gt_pixel_dy") or 0
                shift = (gt_dx**2 + gt_dy**2) ** 0.5
                if shift >= MIN_PIXEL_SHIFT:
                    saved = result
                    print(f"  [{len(collected)+1}/{NUM_TRIALS}] {obj_name}  "
                          f"dx={dx:+.2f} dz={dz:+.2f}  shift={shift:.1f}px  ✓")
                    break
                else:
                    print(f"  {obj_name}  dx={dx:+.2f} dz={dz:+.2f}  "
                          f"shift={shift:.1f}px  (too small, trying next)")
                    import shutil
                    shutil.rmtree(out_dir, ignore_errors=True)
            except Exception as e:
                print(f"  {obj_name}  dx={dx:+.2f}  error: {e}")
                import shutil
                shutil.rmtree(out_dir, ignore_errors=True)

        if saved is not None:
            collected.append(saved)
        else:
            print(f"  {obj_name}  — no valid displacement found, skipping")

    print(f"\nTrials collected: {len(collected)}/{NUM_TRIALS} "
          f"(tried {attempted_objs} objects)")
    return collected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True, help="e.g. FloorPlan2")
    parser.add_argument("--out", default=str(Path.home() / "Documents" / "dataset"))
    parser.add_argument("--skip_collect", action="store_true",
                        help="Skip image collection (use existing scene_meta.json)")
    parser.add_argument("--display", default=":99")
    args = parser.parse_args()

    if "DISPLAY" not in os.environ:
        os.environ["DISPLAY"] = args.display

    out_root = Path(args.out)
    meta_path = out_root / args.scene / "env" / "scene_meta.json"

    # ── Step 1: collect training images ──────────────────────────────────────
    if args.skip_collect and meta_path.exists():
        print(f"Skipping collection, using existing {meta_path}")
        meta = json.load(open(meta_path))
    else:
        print(f"\n{'='*55}")
        print(f"  Collecting training data: {args.scene}")
        print(f"{'='*55}")
        meta = collect(scene=args.scene, out_root=out_root)

    # ── Step 2: collect 5 displacement trials ────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Collecting displacement trials: {args.scene}")
    print(f"{'='*55}")
    trials = collect_trials(scene=args.scene, out_root=out_root, meta=meta)

    # ── Summary ───────────────────────────────────────────────────────────────
    env_dir    = out_root / args.scene / "env"
    trials_dir = out_root / args.scene / "trials"
    n_images   = len(list((env_dir / "images").glob("*.png"))) if (env_dir / "images").exists() else 0
    n_trials   = len(trials)

    print(f"\n{'='*55}")
    print(f"  Done: {args.scene}")
    print(f"  Training images : {n_images}")
    print(f"  Trials collected: {n_trials}/5")
    for t in trials:
        obj_name = t["object_id"].split("|")[0].lower()
        dx = t.get("gt_pixel_dx", 0) or 0
        dy = t.get("gt_pixel_dy", 0) or 0
        print(f"    {obj_name:<16}  shift ({dx:+.1f}, {dy:+.1f}) px")
    print(f"  Output → {out_root / args.scene}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
