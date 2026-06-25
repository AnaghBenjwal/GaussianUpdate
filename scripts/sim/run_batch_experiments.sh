#!/usr/bin/env bash
# run_batch_experiments.sh — 25-trial batch experiment on AI2THOR FloorPlan1
#
# 5 objects × 5 movements = 25 trials.
# Requires:
#   - data/sim/FloorPlan1/       (training data already collected)
#   - output/sim/FloorPlan1/     (GG model already trained)
#
# Usage:
#   bash scripts/sim/run_batch_experiments.sh [ITERATION]
#   bash scripts/sim/run_batch_experiments.sh 10000

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

ITERATION="${1:-10000}"
RUN_TAG="${2:-iter${ITERATION}}"   # e.g. "iter10000" or "iter30000"
SCENE="FloorPlan1"
DATA_DIR="$ROOT/data/sim/$SCENE"
MODEL_DIR="$ROOT/output/sim/$SCENE"
META="$DATA_DIR/scene_meta.json"
BA_ROOT="$DATA_DIR/before_after"
OUT_ROOT="$ROOT/output/sim_experiments/$SCENE/$RUN_TAG"
NUM_CLASSES=32

mkdir -p "$OUT_ROOT"
LOG="$OUT_ROOT/run.log"
echo "Batch experiment started: $(date)" | tee "$LOG"
echo "Model: $MODEL_DIR @ iter $ITERATION" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# ── Experiment table: obj_idx | cls_idx | obj_name | move_name | dx | dz ──────
# obj_idx maps to collect_data.py sorted object list (Apple=0, Book=1, ...)
# cls_idx = obj_idx + 1
#
# Objects chosen to have non-zero Gaussian coverage and be visible from cam0:
#   0  cls=1  Apple          54  Gaussians — kitchen table
#   1  cls=2  Book          698  Gaussians — kitchen table
#   3  cls=4  Bowl          252  Gaussians — kitchen table
#   4  cls=5  Bread         331  Gaussians — kitchen table
#   6  cls=7  CoffeeMachine 856  Gaussians — left counter (replaces ButterKnife cls=6 which has 0)
#
# Movements use forceAction=True TeleportObject fallback so physics collisions
# are bypassed. Displacements chosen to keep objects on visible surfaces.
#
# right    dx=+0.25  dz= 0.00
# left     dx=-0.20  dz= 0.00
# forward  dx= 0.00  dz=+0.20  (toward camera)
# back     dx= 0.00  dz=-0.20  (away from camera)
# diagonal dx=+0.18  dz=+0.18

declare -a EXPERIMENTS=(
    # obj_idx:cls_idx:obj_name:move_name:dx:dz
    "0:1:apple:right:0.25:0.00"
    "0:1:apple:left:-0.15:0.00"      # -0.20 pushed apple off counter; -0.15 stays visible
    "0:1:apple:forward:0.00:0.20"
    "0:1:apple:back:0.00:-0.20"
    "0:1:apple:diagonal:0.18:0.18"

    "1:2:book:right:0.25:0.00"
    "1:2:book:left:-0.20:0.00"
    "1:2:book:forward:0.00:0.20"
    "1:2:book:back:0.00:-0.20"
    "1:2:book:diagonal:0.18:0.18"

    "3:4:bowl:right:0.15:0.00"       # 0.25 pushed bowl off counter; 0.15 stays
    "3:4:bowl:left:-0.20:0.00"
    "3:4:bowl:forward:0.00:0.35"     # 0.20 gave only 4px shift; 0.35 gives more
    "3:4:bowl:back:0.00:-0.30"       # 0.20 gave only 3.6px shift; 0.30 gives more
    "3:4:bowl:diagonal:0.18:0.18"

    "4:5:bread:right:0.25:0.00"
    "4:5:bread:left:-0.15:0.00"      # -0.20 pushed bread off counter; -0.15 stays
    "4:5:bread:forward:0.00:0.20"
    "4:5:bread:back:0.00:-0.20"
    "4:5:bread:diagonal:0.18:0.18"

    "6:7:coffeemachine:right:0.25:0.00"
    "6:7:coffeemachine:left:-0.20:0.00"
    "6:7:coffeemachine:forward:0.00:0.20"
    "6:7:coffeemachine:back:0.00:-0.20"
    "6:7:coffeemachine:diagonal:0.18:0.18"
)

TOTAL=${#EXPERIMENTS[@]}
PASSED=0
FAILED=0

for exp in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r OBJ_IDX CLS_IDX OBJ_NAME MOVE DX DZ <<< "$exp"
    PAIR_NAME="${OBJ_NAME}_${MOVE}"
    BA_DIR="$BA_ROOT/$PAIR_NAME"
    OUT_DIR="$OUT_ROOT/$PAIR_NAME"
    EXP_LOG="$OUT_ROOT/${PAIR_NAME}.log"

    echo "────────────────────────────────────────────────────────"   | tee -a "$LOG"
    echo "  [$((PASSED+FAILED+1))/$TOTAL]  $PAIR_NAME  (cls=$CLS_IDX  dx=$DX  dz=$DZ)" | tee -a "$LOG"

    # Skip if already completed successfully
    if [ -f "$OUT_DIR/metrics.json" ] && python3 -c "
import json; m=json.load(open('$OUT_DIR/metrics.json'))
exit(0 if m.get('psnr_updated_vs_gt_after') is not None else 1)
" 2>/dev/null; then
        echo "  Already completed — skipping" | tee -a "$LOG"
        PASSED=$((PASSED+1))
        continue
    fi

    # Step A — generate before/after pair
    echo "  [A] before_after.py ..." | tee -a "$LOG"
    if conda run -n dev python "$ROOT/scripts/sim/before_after.py" \
        --scene "$SCENE" \
        --meta "$META" \
        --object_idx "$OBJ_IDX" \
        --dx "$DX" --dz "$DZ" \
        --out "$BA_DIR" \
        >> "$EXP_LOG" 2>&1; then
        echo "      OK → $BA_DIR" | tee -a "$LOG"
    else
        echo "      FAILED — skipping eval" | tee -a "$LOG"
        FAILED=$((FAILED+1))
        continue
    fi

    # Safety check: if gt_pixel shift is tiny the object didn't move
    SHIFT=$(python3 -c "
import json, math, sys
try:
    m = json.load(open('$BA_DIR/meta.json'))
    dx, dy = m.get('gt_pixel_dx', 0), m.get('gt_pixel_dy', 0)
    print(f'{math.sqrt(dx**2+dy**2):.1f}')
except Exception as e:
    print('0')
" 2>/dev/null)
    if python3 -c "exit(0 if float('$SHIFT') > 3 else 1)" 2>/dev/null; then
        echo "      pixel shift magnitude: ${SHIFT}px" | tee -a "$LOG"
    else
        echo "      WARNING: object barely moved (${SHIFT}px) — skipping" | tee -a "$LOG"
        FAILED=$((FAILED+1))
        continue
    fi

    # Step B — run GG detect_and_update + PSNR
    echo "  [B] eval_experiment.py ..." | tee -a "$LOG"
    mkdir -p "$OUT_DIR"
    if conda run -n gaussian_grouping python "$ROOT/scripts/sim/eval_experiment.py" \
        --model_path "$MODEL_DIR" \
        --source_path "$DATA_DIR" \
        --before_after_dir "$BA_DIR" \
        --iteration "$ITERATION" \
        --num_classes "$NUM_CLASSES" \
        --output "$OUT_DIR" \
        >> "$EXP_LOG" 2>&1; then

        # Extract and echo key metrics from metrics.json
        python3 -c "
import json
m = json.load(open('$OUT_DIR/metrics.json'))
gt_dx   = m.get('gt_pixel_dx', 0)
gt_dy   = m.get('gt_pixel_dy', 0)
det_dx  = m.get('detected_pixel_dx') or 0
det_dy  = m.get('detected_pixel_dy') or 0
psnr    = m.get('psnr_updated_vs_gt_after', 0)
ssim    = m.get('ssim_updated_vs_gt_after', 0)
import math
gt_mag  = math.sqrt(gt_dx**2 + gt_dy**2)
det_mag = math.sqrt(det_dx**2 + det_dy**2)
err     = math.sqrt((gt_dx-det_dx)**2 + (gt_dy-det_dy)**2)
print(f'      GT shift: ({gt_dx:+.1f}, {gt_dy:+.1f}) px  |{gt_mag:.1f}|')
print(f'      Detected: ({det_dx:+.1f}, {det_dy:+.1f}) px  |{det_mag:.1f}|')
print(f'      Error:    {err:.1f} px')
print(f'      PSNR:     {psnr:.2f} dB   SSIM: {ssim:.3f}')
" 2>/dev/null | tee -a "$LOG"
        PASSED=$((PASSED+1))
    else
        echo "      eval FAILED" | tee -a "$LOG"
        FAILED=$((FAILED+1))
    fi
done

echo "" | tee -a "$LOG"
echo "════════════════════════════════════════════════════════════" | tee -a "$LOG"
echo "  Done: $PASSED passed, $FAILED failed / $TOTAL total" | tee -a "$LOG"
echo "════════════════════════════════════════════════════════════" | tee -a "$LOG"

# ── Final summary table ────────────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "Running summary aggregation..." | tee -a "$LOG"
conda run -n gaussian_grouping python "$ROOT/scripts/sim/summarize_batch.py" \
    --batch_dir "$OUT_ROOT" \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "All outputs → $OUT_ROOT" | tee -a "$LOG"
echo "Log → $LOG" | tee -a "$LOG"
