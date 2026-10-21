#!/bin/bash
# pipeline.sh — runs after the GG chain: FP2/4/5 test moves, then the DispNet tweak runs (one at a time: 1.9 CPU cores)
until grep -q GG_ALL_DONE /root/train_gg.log; do sleep 60; done
echo "== GG finished $(date)"; grep -e "^ok" -e FAIL /root/train_gg.log
cd /root/Documents/gaussian-grouping; mkdir -p /root/tmp_t1
for n in 2 4 5; do
  [ -d /root/Documents/displacement_bench_t2/FloorPlan$n ] && [ -f /root/gen_FP$n.done ] && continue
  echo "== gen_test FP$n $(date)"
  /venv/main/bin/python scripts/learn/gen_test.py --scene FloorPlan$n --out /root/Documents/displacement_bench_t2 --t1 /root/tmp_t1 \
    > /root/gen_FP$n.log 2>&1 && touch /root/gen_FP$n.done && tail -2 /root/gen_FP$n.log || echo "FAIL gen FP$n"
done
sed -i "s/--workers 6/--workers 3/" /root/run_tweaks.sh
/root/run_tweaks.sh base:"--arch r18 --preload" planar:"--arch r18 --planar --preload" \
  r34:"--arch r34 --preload" res384:"--arch r18 --res 384 --preload"
echo PIPE_DONE $(date)
