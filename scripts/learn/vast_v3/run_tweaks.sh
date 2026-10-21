#!/bin/bash
# usage: run_tweaks.sh name:"flags" ...   (sequential DispNet runs on the FP10-15 validation split)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/Documents/gaussian-grouping; R=/root/Documents/displacement_learn/runs2; mkdir -p $R
for job in "$@"; do n=${job%%:*}; f=${job#*:}
  [ -f $R/$n/DONE ] && continue
  echo "== $n ($f) $(date)"
  /venv/main/bin/python scripts/learn/train.py --data /root/Documents/displacement_learn --train train_v2 --val val_v2 \
     --out $R/$n --epochs 60 --workers 3 $f > $R/$n.log 2>&1 && touch $R/$n/DONE && tail -1 $R/$n.log
done
echo TWEAKS_DONE
