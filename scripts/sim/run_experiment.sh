#!/usr/bin/env bash
# run_experiment.sh — Full pipeline with high-quality environment capture.
#
# Architecture:
#   ~/dataset/<scene>/env/      — static environment capture (~300 views, collected ONCE)
#   ~/dataset/<scene>/model/    — GG trained model (30k iters)
#   ~/dataset/<scene>/trials/   — before/after pairs (lightweight, per object)
#   ~/dataset/<scene>/results/  — detect_and_update outputs + metrics
#
# Usage:
#   bash scripts/sim/run_experiment.sh [SCENE]
#   bash scripts/sim/run_experiment.sh FloorPlan1
#
# Steps:
#   1. Collect ~300-view environment data (skip if env/scene_meta.json exists)
#   2. Train GG at 30k iterations
#   3. Generate before/after pairs for 5 objects
#   4. Run eval (detect_and_update + PSNR/SSIM) for each trial
#   5. Print summary table

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

SCENE="${1:-FloorPlan1}"
DATASET_DIR="$HOME/Documents/dataset"
ENV_DIR="$DATASET_DIR/$SCENE/env"
MODEL_DIR="$DATASET_DIR/$SCENE/model"
TRIALS_DIR="$DATASET_DIR/$SCENE/trials"
RESULTS_DIR="$DATASET_DIR/$SCENE/results"
ITERATIONS=30000
NUM_CLASSES=32
CONFIG="$ROOT/config/sim/train_30k.json"

echo "═══════════════════════════════════════════════════════════"
echo "  Experiment  scene=$SCENE  iters=$ITERATIONS"
echo "  Dataset: $DATASET_DIR/$SCENE"
echo "═══════════════════════════════════════════════════════════"

mkdir -p "$MODEL_DIR" "$TRIALS_DIR" "$RESULTS_DIR"

# ── 1. Collect environment data (skip if already done) ────────────────────────
echo ""
if [ -f "$ENV_DIR/scene_meta.json" ]; then
    N_IMGS=$(python3 -c "import json; m=json.load(open('$ENV_DIR/scene_meta.json')); print(m['num_frames'])" 2>/dev/null || echo "?")
    echo "[1/4] Environment data already exists ($N_IMGS frames) — skipping capture."
    echo "      Delete $ENV_DIR to force a re-capture."
else
    echo "[1/4] Collecting ~300-view environment data from AI2THOR..."
    conda run -n dev python "$ROOT/scripts/sim/collect_data.py" \
        --scene "$SCENE" \
        --out "$DATASET_DIR"
    N_IMGS=$(python3 -c "import json; m=json.load(open('$ENV_DIR/scene_meta.json')); print(m['num_frames'])" 2>/dev/null || echo "?")
    echo "      Captured $N_IMGS frames → $ENV_DIR"
fi

# ── 2. Train Gaussian Grouping ─────────────────────────────────────────────────
echo ""
CKPT="$MODEL_DIR/point_cloud/iteration_${ITERATIONS}/point_cloud.ply"
if [ -f "$CKPT" ]; then
    echo "[2/4] GG checkpoint already exists at iteration $ITERATIONS — skipping training."
    echo "      Delete $MODEL_DIR to force retraining."
else
    echo "[2/4] Training Gaussian Grouping ($ITERATIONS iters)..."
    conda run -n gaussian_grouping python "$ROOT/train.py" \
        -s "$ENV_DIR" \
        -m "$MODEL_DIR" \
        --iterations "$ITERATIONS" \
        --save_iterations "$ITERATIONS" \
        --config_file "$CONFIG" \
        2>&1 | tee "$MODEL_DIR/train.log"
    echo "      GG model → $MODEL_DIR"
fi

# ── 3. Generate before/after pairs ────────────────────────────────────────────
# Objects validated in FloorPlan1 batch experiments (obj_idx:cls_idx:name:dx:dz)
# One representative displacement per object ("right" or closest validated direction)
echo ""
echo "[3/4] Generating before/after trials for 5 objects..."
META="$ENV_DIR/scene_meta.json"

declare -a TRIALS=(
    "0:1:apple:0.25:0.00"
    "1:2:book:0.25:0.00"
    "3:4:bowl:0.15:0.00"
    "4:5:bread:0.25:0.00"
    "6:7:coffeemachine:0.25:0.00"
)

for t in "${TRIALS[@]}"; do
    IFS=':' read -r OBJ_IDX CLS_IDX NAME DX DZ <<< "$t"
    TRIAL_DIR="$TRIALS_DIR/$NAME"
    if [ -f "$TRIAL_DIR/after.png" ]; then
        echo "      $NAME — already exists, skipping"
        continue
    fi
    echo "      $NAME (obj_idx=$OBJ_IDX  dx=$DX  dz=$DZ)"
    conda run -n dev python "$ROOT/scripts/sim/before_after.py" \
        --scene "$SCENE" \
        --meta "$META" \
        --object_idx "$OBJ_IDX" \
        --dx "$DX" --dz "$DZ" \
        --out "$TRIAL_DIR" \
    || echo "      WARNING: $NAME failed — will skip eval"
done

# ── 4. Eval each trial ────────────────────────────────────────────────────────
echo ""
echo "[4/4] Running detect_and_update + PSNR/SSIM for each trial..."
PASSED=0; FAILED=0

for t in "${TRIALS[@]}"; do
    IFS=':' read -r OBJ_IDX CLS_IDX NAME DX DZ <<< "$t"
    TRIAL_DIR="$TRIALS_DIR/$NAME"
    OUT_DIR="$RESULTS_DIR/$NAME"
    EXP_LOG="$RESULTS_DIR/${NAME}.log"

    echo "  ── $NAME ──"

    if [ ! -f "$TRIAL_DIR/after.png" ]; then
        echo "      No after.png — skipping"
        FAILED=$((FAILED+1))
        continue
    fi

    if [ -f "$OUT_DIR/metrics.json" ]; then
        PSNR=$(python3 -c "import json; m=json.load(open('$OUT_DIR/metrics.json')); v=m.get('psnr_updated_vs_gt_after'); print(f'{v:.2f}' if v else 'None')" 2>/dev/null || echo "?")
        echo "      Already evaluated (PSNR=$PSNR dB) — skipping"
        PASSED=$((PASSED+1))
        continue
    fi

    mkdir -p "$OUT_DIR"
    if conda run -n gaussian_grouping python "$ROOT/scripts/sim/eval_experiment.py" \
        --model_path "$MODEL_DIR" \
        --source_path "$ENV_DIR" \
        --before_after_dir "$TRIAL_DIR" \
        --iteration "$ITERATIONS" \
        --num_classes "$NUM_CLASSES" \
        --output "$OUT_DIR" \
        > "$EXP_LOG" 2>&1; then
        python3 -c "
import json, math
m = json.load(open('$OUT_DIR/metrics.json'))
gt_dx, gt_dy   = m.get('gt_pixel_dx',0), m.get('gt_pixel_dy',0)
det_dx, det_dy = m.get('detected_pixel_dx') or 0, m.get('detected_pixel_dy') or 0
psnr = m.get('psnr_updated_vs_gt_after', 0)
ssim = m.get('ssim_updated_vs_gt_after', 0)
err  = math.sqrt((gt_dx-det_dx)**2 + (gt_dy-det_dy)**2)
print(f'      GT ({gt_dx:+.1f},{gt_dy:+.1f})px  det ({det_dx:+.1f},{det_dy:+.1f})px  err={err:.1f}px  PSNR={psnr:.2f}dB  SSIM={ssim:.3f}')
" 2>/dev/null
        PASSED=$((PASSED+1))
    else
        echo "      eval FAILED — see $EXP_LOG"
        FAILED=$((FAILED+1))
    fi
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
printf "%-18s  %-20s  %-20s  %8s  %6s\n" "Object" "GT shift (dx,dy)" "Det shift (dx,dy)" "Err (px)" "PSNR"
echo "────────────────────────────────────────────────────────────"
for t in "${TRIALS[@]}"; do
    IFS=':' read -r OBJ_IDX CLS_IDX NAME DX DZ <<< "$t"
    OUT_DIR="$RESULTS_DIR/$NAME"
    if [ -f "$OUT_DIR/metrics.json" ]; then
        python3 -c "
import json, math
m    = json.load(open('$OUT_DIR/metrics.json'))
gt_dx, gt_dy   = m.get('gt_pixel_dx',0), m.get('gt_pixel_dy',0)
det_dx, det_dy = m.get('detected_pixel_dx') or 0, m.get('detected_pixel_dy') or 0
psnr = m.get('psnr_updated_vs_gt_after') or 0
err  = math.sqrt((gt_dx-det_dx)**2 + (gt_dy-det_dy)**2)
print(f'$NAME'.ljust(18) + '  ' +
      f'({gt_dx:+.1f},{gt_dy:+.1f})'.ljust(20) + '  ' +
      f'({det_dx:+.1f},{det_dy:+.1f})'.ljust(20) + '  ' +
      f'{err:8.1f}  {psnr:6.2f}')
" 2>/dev/null || printf "%-18s  (no results)\n" "$NAME"
    else
        printf "%-18s  (no results)\n" "$NAME"
    fi
done
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Done: $PASSED passed, $FAILED failed"
echo "Results → $RESULTS_DIR"
