#!/bin/bash
# final.sh RUN SRC [--tta] — the chosen DispNet (runs2/SRC/best.pt) on the test sets: T1 = FP1/3, T2 = FP1-5
RUN=$1; SRC=$2; TTA=$3
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/venv/main/bin/python; B=/root/Documents; L=$B/displacement_learn; R=$L/runs/$RUN; cd $B/gaussian-grouping
step() { echo "== $* $(date)"; "$@" > /root/evlogs/final_$(echo "$*" | md5sum | cut -c1-8).log 2>&1 || echo "FAIL $*"; }
mkdir -p $R /root/evlogs; cp $L/runs2/$SRC/best.pt $L/runs2/$SRC/args.json $L/runs2/$SRC/log.jsonl $R/; echo "$SRC $TTA" > $R/SOURCE
for s in test_t1_gg test_t2_gg test_t1_sim test_t2_sim; do
  step $P scripts/learn/predict.py --data $L --split $s --ckpt $R/best.pt --out $R/pred_$s.jsonl $TTA
done
for bs in displacement_bench:test_t1:1 displacement_bench:test_t1:3 displacement_bench_t2:test_t2:1 displacement_bench_t2:test_t2:3 \
          displacement_bench_t2:test_t2:2 displacement_bench_t2:test_t2:4 displacement_bench_t2:test_t2:5; do
  IFS=: read b t n <<< "$bs"
  step $P scripts/learn/eval_methods.py --bench $B/$b --scene FloorPlan$n --tag FP$n --stage learned --preds $RUN=$R/pred_${t}_gg.jsonl
  step $P scripts/learn/eval_render.py --bench $B/$b --scene FloorPlan$n --tag FP$n --runs $RUN
done
step $P scripts/learn/summarize_eval.py --data $L --runs $RUN --labels dispnet_v3 --out $L/results_v3
echo FINAL_DONE $(date)
