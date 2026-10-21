#!/bin/bash
cd /root/Documents/gaussian-grouping
OUT=/root/Documents/vast_results/server_root/output; mkdir -p $OUT
for n in ${@:-2 4 5}; do
  echo "== GG FloorPlan$n $(date)"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 /venv/main/bin/python train.py -s /root/Documents/dataset_1080p_fixed/FloorPlan$n/env -m $OUT/gg_FP${n}_fixed \
    --config_file config/sim/train_40k.json --eval --port $((6010+n)) \
    --iterations 40000 --save_iterations 40000 --test_iterations 10000 20000 40000 > $OUT/gg_train_log_FP${n}_fixed.txt 2>&1 \
    && echo "ok FP$n $(date)" || echo "FAIL FP$n"
done
echo GG_ALL_DONE
