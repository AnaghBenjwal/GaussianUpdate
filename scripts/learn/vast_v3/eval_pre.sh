#!/bin/bash
# eval_pre.sh — evaluation stages that do not depend on the final DispNet (GPU-heavy; runs beside the CPU-bound tweaks)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/venv/main/bin/python; B=/root/Documents; cd $B/gaussian-grouping
step() { echo "== $* $(date)"; "$@" > /root/evlogs/$(echo "$*" | md5sum | cut -c1-8).log 2>&1 || echo "FAIL $*"; }
mkdir -p /root/evlogs
for bs in displacement_bench:1 displacement_bench:3 displacement_bench_t2:1 displacement_bench_t2:3 displacement_bench_t2:2 displacement_bench_t2:4 displacement_bench_t2:5; do
  b=${bs%:*}; n=${bs#*:}
  step $P scripts/learn/gg_inputs.py --bench $B/$b --scene FloorPlan$n --tag FP$n
done
step $P scripts/learn/export_bench.py --bench $B/displacement_bench --split test_t1 --variant gg --scenes FloorPlan1,FloorPlan3 --out $B/displacement_learn
for v in gg sim; do
  step $P scripts/learn/export_bench.py --bench $B/displacement_bench_t2 --split test_t2 --variant $v --scenes FloorPlan1,FloorPlan2,FloorPlan3,FloorPlan4,FloorPlan5 --out $B/displacement_learn
done
for n in 2 4 5; do
  step $P scripts/learn/eval_methods.py --bench $B/displacement_bench_t2 --scene FloorPlan$n --tag FP$n --stage cues
  step $P scripts/learn/eval_baselines.py --bench $B/displacement_bench_t2 --scene FloorPlan$n --tag FP$n
  step $P scripts/learn/eval_baselines.py --bench $B/displacement_bench_t2 --scene FloorPlan$n --tag FP$n --methods mast3r
done
echo EVAL_PRE_DONE $(date)
