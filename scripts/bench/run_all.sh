#!/bin/bash
# Full displacement benchmark (local): nohup bash scripts/bench/run_all.sh > ~/Documents/displacement_bench/run_all.log 2>&1 &
# Strictly sequential: AI2THOR (Vulkan) and GG rendering (CUDA) must never share the GPU — doing so on 2026-10-09
# ended in Xid 31 → Xid 62 and a hung GPU. Each step logs to ~/Documents/displacement_bench/logs/; any failure aborts.
set -ex
cd /home/unk/Documents/gaussian-grouping
LOGS=~/Documents/displacement_bench/logs; mkdir -p $LOGS
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
for s in FloorPlan1:FP1 FloorPlan3:FP3; do
  sc=${s%%:*}; tag=${s##*:}
  [ -f ~/Documents/displacement_bench/$sc/trials.json ] || conda run --no-capture-output -n dev python -u scripts/bench/gen_trials.py --scene $sc > $LOGS/gen_$tag.log 2>&1
  [ -f ~/Documents/displacement_bench/$sc/langsplat_relevancy.npz ] || conda run --no-capture-output -n dev python -u scripts/bench/ls_relevancy.py --scene $sc --tag $tag > $LOGS/rel_$tag.log 2>&1
done
for s in FloorPlan1:FP1 FloorPlan3:FP3; do
  sc=${s%%:*}; tag=${s##*:}
  conda run --no-capture-output -n gaussian_grouping python -u scripts/bench/run_bench.py --scene $sc --tag $tag > $LOGS/bench_$tag.log 2>&1
done
conda run --no-capture-output -n gaussian_grouping python -u scripts/bench/summarize_bench.py --scenes FloorPlan1 FloorPlan3 > $LOGS/summary.log 2>&1
echo ALL_DONE
