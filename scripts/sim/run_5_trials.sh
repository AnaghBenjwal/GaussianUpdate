#!/usr/bin/env bash
# run_5_trials.sh — Run eval on 5 pre-captured trials and summarize results.
# Uses the trained GG model at output/sim/FloorPlan1 and trial data from
# /home/unk/Documents/dataset/FloorPlan1/trials/{apple,book,bowl,bread,coffeemachine}
#
# Usage:  bash scripts/sim/run_5_trials.sh [ITERATION]

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

ITERATION="${1:-30000}"
SCENE="FloorPlan1"
SOURCE_PATH="$ROOT/data/sim/$SCENE"
MODEL_PATH="$ROOT/output/sim/$SCENE"
TRIALS_DIR="/home/unk/Documents/dataset/$SCENE/trials"
OUT_ROOT="$ROOT/results/${SCENE}_iter${ITERATION}"
NUM_CLASSES=32

mkdir -p "$OUT_ROOT"
LOG="$OUT_ROOT/run.log"
echo "=====================================================" | tee "$LOG"
echo "  Gaussian Grouping — Trial Evaluation" | tee -a "$LOG"
echo "  Scene:     $SCENE" | tee -a "$LOG"
echo "  Model:     $MODEL_PATH @ iter $ITERATION" | tee -a "$LOG"
echo "  Trials:    $TRIALS_DIR" | tee -a "$LOG"
echo "  Output:    $OUT_ROOT" | tee -a "$LOG"
echo "  Started:   $(date)" | tee -a "$LOG"
echo "=====================================================" | tee -a "$LOG"
echo "" | tee -a "$LOG"

TRIALS=(apple book bowl bread coffeemachine)
TOTAL=${#TRIALS[@]}
PASSED=0
FAILED=0

for TRIAL in "${TRIALS[@]}"; do
    BA_DIR="$TRIALS_DIR/$TRIAL"
    OUT_DIR="$OUT_ROOT/$TRIAL"
    EXP_LOG="$OUT_ROOT/${TRIAL}.log"

    echo "─────────────────────────────────────────────────────" | tee -a "$LOG"
    echo "  [$((PASSED+FAILED+1))/$TOTAL]  $TRIAL" | tee -a "$LOG"

    if [ ! -f "$BA_DIR/meta.json" ]; then
        echo "  ERROR: $BA_DIR/meta.json not found — skipping" | tee -a "$LOG"
        FAILED=$((FAILED+1))
        continue
    fi

    mkdir -p "$OUT_DIR"

    if conda run -n gaussian_grouping python "$ROOT/scripts/sim/eval_experiment.py" \
        --model_path "$MODEL_PATH" \
        --source_path "$SOURCE_PATH" \
        --before_after_dir "$BA_DIR" \
        --iteration "$ITERATION" \
        --num_classes "$NUM_CLASSES" \
        --output "$OUT_DIR" \
        > "$EXP_LOG" 2>&1; then

        conda run -n gaussian_grouping python - <<'PYEOF' 2>/dev/null | tee -a "$LOG"
import json, math, sys, os
out_dir = os.environ.get("EVAL_OUT_DIR", "")
PYEOF

        python3 - "$OUT_DIR" 2>/dev/null <<'PYEOF' | tee -a "$LOG"
import json, math, sys
out_dir = sys.argv[1]
m = json.load(open(f"{out_dir}/metrics.json"))
gt_dx  = m.get("gt_pixel_dx", 0) or 0
gt_dy  = m.get("gt_pixel_dy", 0) or 0
det_dx = m.get("detected_pixel_dx") or 0
det_dy = m.get("detected_pixel_dy") or 0
psnr   = m.get("psnr_updated_vs_gt_after")
ssim   = m.get("ssim_updated_vs_gt_after")
gt_mag = math.sqrt(gt_dx**2 + gt_dy**2)
err    = math.sqrt((gt_dx-det_dx)**2 + (gt_dy-det_dy)**2)
print(f"      GT shift:  ({gt_dx:+.1f}, {gt_dy:+.1f}) px  |{gt_mag:.1f}|")
print(f"      Detected:  ({det_dx:+.1f}, {det_dy:+.1f}) px")
print(f"      Error:     {err:.1f} px")
print(f"      PSNR:      {psnr:.2f} dB" if psnr else "      PSNR:      —")
print(f"      SSIM:      {ssim:.4f}" if ssim else "      SSIM:      —")
PYEOF

        echo "      → $OUT_DIR" | tee -a "$LOG"
        PASSED=$((PASSED+1))
    else
        echo "  FAILED — see $EXP_LOG" | tee -a "$LOG"
        tail -5 "$EXP_LOG" | tee -a "$LOG"
        FAILED=$((FAILED+1))
    fi
done

echo "" | tee -a "$LOG"
echo "=====================================================" | tee -a "$LOG"
echo "  Done: $PASSED passed, $FAILED failed / $TOTAL total" | tee -a "$LOG"
echo "  Finished: $(date)" | tee -a "$LOG"
echo "=====================================================" | tee -a "$LOG"

if [ "$PASSED" -gt 0 ]; then
    echo "" | tee -a "$LOG"
    echo "Running summary aggregation..." | tee -a "$LOG"
    conda run -n gaussian_grouping python "$ROOT/scripts/sim/summarize_batch.py" \
        --batch_dir "$OUT_ROOT" \
        2>&1 | tee -a "$LOG"
fi

echo "" | tee -a "$LOG"
echo "All results → $OUT_ROOT" | tee -a "$LOG"
