#!/usr/bin/env bash
# run_sim_pipeline.sh — Full pipeline: AI2THOR → GG training → before/after → detect_and_update
#
# Usage:
#   bash scripts/sim/run_sim_pipeline.sh [SCENE] [ITERATIONS]
#
# Example:
#   bash scripts/sim/run_sim_pipeline.sh FloorPlan1 15000
#
# All steps use cam_idx=0 (the canonical front-centre camera from collect_data.py).

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

SCENE="${1:-FloorPlan1}"
ITERATIONS="${2:-15000}"
DATA_DIR="$ROOT/data/sim/$SCENE"
MODEL_DIR="$ROOT/output/sim/$SCENE"
BA_DIR="$DATA_DIR/before_after"
RESULTS_DIR="$ROOT/output/sim_experiments/$SCENE"
CONFIG="$ROOT/config/sim/train.json"

echo "═══════════════════════════════════════════════════════════"
echo "  Sim Pipeline  scene=$SCENE  iters=$ITERATIONS"
echo "═══════════════════════════════════════════════════════════"

# ── 1. Collect training data ──────────────────────────────────────────────────
echo ""
echo "[1/5] Collecting multi-view data from AI2THOR (headless)..."
conda run -n dev python "$ROOT/scripts/sim/collect_data.py" \
    --scene "$SCENE" \
    --out "$ROOT/data/sim"

# ── 2. Train Gaussian Grouping ────────────────────────────────────────────────
echo ""
echo "[2/5] Training Gaussian Grouping ($ITERATIONS iters)..."
mkdir -p "$MODEL_DIR"
conda run -n gaussian_grouping python "$ROOT/train.py" \
    -s "$DATA_DIR" \
    -m "$MODEL_DIR" \
    --iterations "$ITERATIONS" \
    --save_iterations "$ITERATIONS" \
    --config_file "$CONFIG" \
    2>&1 | tee "$MODEL_DIR/train.log"

echo "  GG model → $MODEL_DIR"

# ── 3. Generate before/after pairs ───────────────────────────────────────────
echo ""
echo "[3/5] Generating before/after pairs..."
META="$DATA_DIR/scene_meta.json"

# Helper: run one before/after pair
run_pair() {
    local NAME="$1"; local OBJ_IDX="$2"; local DX="$3"; local DZ="$4"
    echo "  Pair: $NAME (obj$OBJ_IDX dx=$DX dz=$DZ)"
    conda run -n dev python "$ROOT/scripts/sim/before_after.py" \
        --scene "$SCENE" --meta "$META" \
        --object_idx "$OBJ_IDX" --dx "$DX" --dz "$DZ" \
        --out "$BA_DIR/$NAME" \
    || echo "  WARNING: $NAME failed (object may not be visible — skipping)"
}

run_pair "obj0_right"    0  0.35  0.0
run_pair "obj0_back"     0  0.0   0.30
run_pair "obj1_right"    1  0.30  0.0
run_pair "obj1_diag"     1  0.25  0.20
run_pair "obj2_right"    2  0.35  0.0

echo "  Before/after pairs → $BA_DIR/"

# ── 4. Run detect_and_update on each pair ────────────────────────────────────
echo ""
echo "[4/5] Running detect_and_update..."
mkdir -p "$RESULTS_DIR"

for PAIR_DIR in "$BA_DIR"/*/; do
    [ -f "$PAIR_DIR/after.png" ] || continue
    PAIR_NAME="$(basename "$PAIR_DIR")"
    PAIR_META="$PAIR_DIR/meta.json"

    CLASS_IDX=$(python3 -c "import json; d=json.load(open('$PAIR_META')); print(d['class_idx'])" 2>/dev/null || echo "1")

    echo "  → $PAIR_NAME (class $CLASS_IDX)"
    conda run -n gaussian_grouping python "$ROOT/scripts/detect_and_update.py" \
        --model_path "$MODEL_DIR" \
        --source_path "$DATA_DIR" \
        --iteration "$ITERATIONS" \
        --cam_idx 0 \
        --after "$PAIR_DIR/after.png" \
        --force_obj_id "$CLASS_IDX" \
        --output "$RESULTS_DIR/$PAIR_NAME" \
        --num_classes 32 \
        --no_sam2 \
        2>&1 | tee "$RESULTS_DIR/${PAIR_NAME}.log" \
    || echo "  WARNING: detect_and_update failed for $PAIR_NAME"
done

# ── 5. Summary ────────────────────────────────────────────────────────────────
echo ""
echo "[5/5] Results:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
printf "%-20s  %-18s  %-18s\n" "Pair" "GT (dx, dy) px" "Detected (dx, dy) px"
echo "────────────────────────────────────────────────────────"
for PAIR_DIR in "$BA_DIR"/*/; do
    [ -f "$PAIR_DIR/meta.json" ] || continue
    PAIR_NAME="$(basename "$PAIR_DIR")"
    REPORT="$RESULTS_DIR/$PAIR_NAME/report.json"
    GT=$(python3 -c "
import json
d=json.load(open('$PAIR_DIR/meta.json'))
dx,dy=d.get('gt_pixel_dx','?'),d.get('gt_pixel_dy','?')
print(f'({dx:.1f}, {dy:.1f})' if dx != '?' else '(unknown)')
" 2>/dev/null || echo "(error)")
    if [ -f "$REPORT" ]; then
        PRED=$(python3 -c "
import json
r=json.load(open('$REPORT'))
ps=r.get('px_shift',[None,None])
dx,dy=ps[0],ps[1]
print(f'({dx:.1f}, {dy:.1f})' if dx is not None else '(parse err)')
" 2>/dev/null || echo "(error)")
    else
        PRED="(no report)"
    fi
    printf "%-20s  %-18s  %-18s\n" "$PAIR_NAME" "$GT" "$PRED"
done
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Results → $RESULTS_DIR/"
echo "Visuals → $RESULTS_DIR/*/comparison.png"
