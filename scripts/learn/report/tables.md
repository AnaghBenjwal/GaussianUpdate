# Displacement estimation — all methods on the held-out test sets

Error = ‖δ̂ − δ‖ (3D, cm) per (trial, input view). Inputs: real BEFORE + AFTER photo of one view, GG model of the BEFORE scene, ground-truth class (2-image identification is 100 % on T1). A cue that returns nothing counts as zero motion ('no answer' column). Learned rows use GG-rendered depth + mask unless marked (sim inputs).

Methods:

- **no_update**: predict zero motion
- **P1**: current pipeline A: template-matched 2D shift, constant-depth back-projection
- **P2**: current pipeline B: RAFT 2D shift, constant-depth back-projection
- **raft_orig**: RAFT median shift + support-plane cast
- **raft_crop**: RAFT dense flow on a crop + per-match plane lifting + RANSAC
- **roma**: RoMa matches on the crop + plane lifting + RANSAC
- **depth**: Depth Anything V2 (aligned to GG depth), appeared-region centroid
- **sam**: SAM outline of the appeared object, contact point on the base plane
- **rc**: global differential render-and-compare (V2 baseline)
- **agree**: ablation winner: consensus of {raft_crop, roma, depth, sam} overrides rc when they disagree > 5 cm
- **<model>**: DispNet prediction; **<model>+rc**: local render-and-compare (±8 cm) started at it; **agree+<model>**: agree rule with DispNet as a fifth cue; **gate_<model>**: DispNet when its predicted uncertainty is < 3 cm, else agree; **agree→<model>+rc**: agree, unless DispNet disagrees with it by > 10 cm (planar), then <model>+rc.

## T1 — original benchmark (37 moves, 1 input view each; methods were tuned on it)

37 (trial, view) inputs, 37 moves, median |δ| 15.0 cm

| method | n | median cm | mean cm | p90 cm | <2 cm | <5 cm | <10 cm | no answer |
|---|---|---|---|---|---|---|---|---|
| no_update | 37 | 15.0 | 17.3 | 30.0 | 0% | 0% | 0% | 0 |
| P1 | 37 | 96.8 | 91.7 | 208.5 | 0% | 5% | 19% | 0 |
| P2 | 37 | 35.2 | 73.3 | 198.3 | 0% | 0% | 0% | 0 |
| raft_orig | 37 | 19.9 | 91.8 | 307.6 | 0% | 3% | 11% | 0 |
| raft_crop | 37 | 0.9 | 5.2 | 15.0 | 62% | 76% | 78% | 4 |
| roma | 37 | 0.8 | 6.9 | 15.0 | 57% | 65% | 65% | 12 |
| depth | 37 | 5.4 | 11.3 | 18.7 | 5% | 38% | 68% | 2 |
| sam | 37 | 6.5 | 14.6 | 15.5 | 11% | 46% | 62% | 6 |
| rc | 37 | 1.3 | 7.1 | 15.0 | 57% | 62% | 62% | 0 |
| agree | 37 | 0.8 | 2.5 | 5.3 | 68% | 86% | 92% | 0 |
| agree+dispnet_v1 | 37 | 0.9 | 3.1 | 11.6 | 62% | 81% | 86% | 0 |
| agree+dispnet_v2 | 37 | 0.9 | 3.2 | 6.9 | 62% | 89% | 92% | 0 |
| agree→dispnet_v1+rc | 37 | 2.7 | 5.2 | 14.9 | 46% | 70% | 78% | 0 |
| agree→dispnet_v2+rc | 37 | 1.3 | 4.1 | 13.0 | 57% | 78% | 86% | 0 |
| dispnet_v1 | 37 | 8.3 | 8.6 | 15.3 | 3% | 35% | 62% | 0 |
| dispnet_v1 (sim inputs) | 37 | 7.2 | 8.5 | 15.1 | 5% | 38% | 65% | 0 |
| dispnet_v1+rc | 37 | 2.1 | 5.5 | 14.7 | 49% | 68% | 76% | 0 |
| dispnet_v2 | 37 | 6.8 | 8.0 | 13.4 | 8% | 38% | 78% | 0 |
| dispnet_v2 (sim inputs) | 37 | 5.5 | 7.6 | 14.0 | 5% | 46% | 76% | 0 |
| dispnet_v2+rc | 37 | 1.8 | 5.1 | 14.7 | 51% | 68% | 81% | 0 |
| gate_dispnet_v1 | 37 | 6.1 | 7.1 | 12.8 | 11% | 43% | 70% | 0 |
| gate_dispnet_v2 | 37 | 5.1 | 6.2 | 12.6 | 19% | 49% | 81% | 0 |

### test_t1: median error (cm) by move size

| method | 0-5 cm | 5-10 cm | 10-20 cm | 20-45 cm |
|---|---|---|---|---|
| P1 | – | – | 102.0 (n=31) | 59.5 (n=6) |
| P2 | – | – | 31.4 (n=31) | 77.2 (n=6) |
| rc | – | – | 1.3 (n=31) | 1.1 (n=6) |
| agree | – | – | 0.8 (n=31) | 1.1 (n=6) |
| agree+dispnet_v1 | – | – | 0.9 (n=31) | 2.7 (n=6) |
| agree+dispnet_v2 | – | – | 0.8 (n=31) | 3.5 (n=6) |
| agree→dispnet_v1+rc | – | – | 2.0 (n=31) | 7.3 (n=6) |
| agree→dispnet_v2+rc | – | – | 0.8 (n=31) | 5.6 (n=6) |
| dispnet_v1 | – | – | 7.8 (n=31) | 15.3 (n=6) |
| dispnet_v1+rc | – | – | 2.0 (n=31) | 7.3 (n=6) |
| dispnet_v2 | – | – | 6.2 (n=31) | 12.7 (n=6) |
| dispnet_v2+rc | – | – | 1.4 (n=31) | 5.4 (n=6) |
| gate_dispnet_v1 | – | – | 6.1 (n=31) | 6.6 (n=6) |
| gate_dispnet_v2 | – | – | 4.9 (n=31) | 11.2 (n=6) |

### test_t1: inputs where global render-and-compare fails (rc error > 10 cm, n=14)

| method | n | median cm | mean cm | p90 cm | <2 cm | <5 cm | <10 cm | no answer |
|---|---|---|---|---|---|---|---|---|
| P1 | 14 | 14.3 | 71.6 | 168.3 | 0% | 14% | 36% | 0 |
| P2 | 14 | 24.8 | 92.4 | 300.1 | 0% | 0% | 0% | 0 |
| rc | 14 | 15.0 | 17.6 | 29.0 | 0% | 0% | 0% | 0 |
| agree | 14 | 3.0 | 4.4 | 12.1 | 36% | 71% | 86% | 0 |
| agree+dispnet_v1 | 14 | 3.0 | 4.5 | 12.1 | 29% | 71% | 86% | 0 |
| agree+dispnet_v2 | 14 | 3.5 | 5.0 | 13.4 | 21% | 79% | 86% | 0 |
| agree→dispnet_v1+rc | 14 | 3.3 | 6.5 | 15.0 | 29% | 64% | 71% | 0 |
| agree→dispnet_v2+rc | 14 | 3.0 | 4.9 | 14.0 | 36% | 71% | 79% | 0 |
| dispnet_v1 | 14 | 4.8 | 6.8 | 13.8 | 7% | 57% | 71% | 0 |
| dispnet_v1+rc | 14 | 6.0 | 8.1 | 15.2 | 29% | 50% | 57% | 0 |
| dispnet_v2 | 14 | 6.0 | 5.6 | 8.9 | 14% | 43% | 100% | 0 |
| dispnet_v2+rc | 14 | 6.9 | 7.4 | 14.8 | 21% | 43% | 64% | 0 |
| gate_dispnet_v1 | 14 | 4.8 | 6.8 | 13.8 | 7% | 57% | 71% | 0 |
| gate_dispnet_v2 | 14 | 5.1 | 5.0 | 8.6 | 21% | 50% | 100% | 0 |

### test_t1: median error (cm) per object

| object | n | P1 | P2 | rc | agree | agree+dispnet_v1 | agree+dispnet_v2 | agree→dispnet_v1+rc | agree→dispnet_v2+rc | dispnet_v1 | dispnet_v1+rc | dispnet_v2 | dispnet_v2+rc | gate_dispnet_v1 | gate_dispnet_v2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| apple | 1 | 15.3 | 16.8 | 0.6 | 0.6 | 0.6 | 0.6 | 0.6 | 0.6 | 7.8 | 0.3 | 1.0 | 0.5 | 7.8 | 1.0 |
| book | 4 | 112.8 | 16.2 | 0.1 | 0.3 | 0.3 | 0.1 | 8.1 | 0.3 | 12.3 | 5.2 | 9.9 | 0.3 | 11.6 | 12.9 |
| bottle | 3 | 9.0 | 334.8 | 15.0 | 0.8 | 0.8 | 1.1 | 0.8 | 0.8 | 10.0 | 15.3 | 6.4 | 9.7 | 10.0 | 6.4 |
| bowl | 8 | 113.1 | 66.4 | 0.6 | 0.6 | 0.6 | 0.6 | 1.1 | 0.6 | 5.1 | 0.9 | 5.2 | 0.8 | 5.1 | 5.2 |
| bread | 4 | 18.2 | 17.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 6.4 | 0.3 | 6.0 | 0.1 | 6.4 | 6.0 |
| butter knife | 4 | 158.9 | 15.5 | 15.0 | 9.8 | 9.8 | 9.8 | 9.8 | 8.2 | 7.0 | 13.9 | 6.2 | 12.4 | 7.0 | 6.2 |
| coffee machine | 4 | 281.6 | 45.4 | 2.6 | 1.3 | 1.3 | 1.3 | 7.4 | 2.0 | 13.7 | 7.5 | 8.8 | 8.9 | 10.5 | 1.3 |
| dish sponge | 2 | 7.6 | 99.5 | 22.5 | 4.0 | 3.8 | 3.6 | 4.0 | 4.0 | 4.0 | 1.1 | 1.0 | 1.0 | 4.0 | 1.0 |
| house plant | 3 | 151.2 | 80.2 | 0.0 | 0.0 | 0.0 | 0.9 | 2.1 | 6.3 | 13.0 | 2.1 | 14.5 | 6.3 | 8.3 | 4.7 |
| kettle | 4 | 9.5 | 14.9 | 15.0 | 2.2 | 2.7 | 2.6 | 2.2 | 2.2 | 3.3 | 2.7 | 5.9 | 2.6 | 3.3 | 5.9 |

Mean runtime per input (s, RTX 3090, shared GPU): P 1.37, raft 0.68, roma 3.13, depth 3.15, sam 2.89, rc 10.19

## T2 — new held-out set (67 random moves, every visible view)

368 (trial, view) inputs, 67 moves, median |δ| 7.3 cm

| method | n | median cm | mean cm | p90 cm | <2 cm | <5 cm | <10 cm | no answer |
|---|---|---|---|---|---|---|---|---|
| no_update | 368 | 7.3 | 11.0 | 20.7 | 0% | 32% | 60% | 0 |
| P1 | 368 | 6.6 | 48.4 | 174.7 | 15% | 40% | 60% | 0 |
| P2 | 368 | 52.4 | 68.2 | 134.5 | 1% | 3% | 8% | 0 |
| raft_orig | 368 | 65.1 | 229.1 | 493.8 | 1% | 4% | 15% | 0 |
| raft_crop | 368 | 0.4 | 2.1 | 4.6 | 85% | 90% | 92% | 29 |
| roma | 368 | 0.2 | 4.5 | 19.5 | 74% | 79% | 82% | 90 |
| depth | 368 | 6.7 | 10.8 | 22.2 | 6% | 38% | 64% | 46 |
| sam | 368 | 2.9 | 8.1 | 13.7 | 41% | 74% | 86% | 97 |
| rc | 368 | 0.6 | 3.5 | 13.7 | 74% | 82% | 87% | 0 |
| agree | 368 | 0.5 | 1.7 | 3.4 | 82% | 93% | 96% | 0 |
| agree+dispnet_v1 | 368 | 0.5 | 1.6 | 3.6 | 82% | 93% | 96% | 0 |
| agree+dispnet_v2 | 368 | 0.5 | 1.6 | 3.3 | 82% | 94% | 97% | 0 |
| agree→dispnet_v1+rc | 368 | 0.6 | 2.2 | 5.3 | 79% | 89% | 94% | 0 |
| agree→dispnet_v2+rc | 368 | 0.5 | 1.8 | 4.2 | 79% | 91% | 95% | 0 |
| dispnet_v1 | 368 | 3.6 | 4.9 | 10.1 | 30% | 63% | 90% | 0 |
| dispnet_v1 (sim inputs) | 368 | 3.5 | 4.8 | 9.7 | 30% | 64% | 90% | 0 |
| dispnet_v1+rc | 368 | 0.6 | 2.4 | 7.1 | 77% | 86% | 93% | 0 |
| dispnet_v2 | 368 | 2.8 | 4.0 | 8.9 | 38% | 71% | 92% | 0 |
| dispnet_v2 (sim inputs) | 368 | 2.4 | 3.7 | 7.7 | 42% | 73% | 94% | 0 |
| dispnet_v2+rc | 368 | 0.6 | 2.2 | 6.3 | 77% | 85% | 94% | 0 |
| gate_dispnet_v1 | 368 | 3.2 | 4.4 | 9.2 | 33% | 68% | 92% | 0 |
| gate_dispnet_v2 | 368 | 2.4 | 3.6 | 7.8 | 43% | 76% | 93% | 0 |

### test_t2: median error (cm) by move size

| method | 0-5 cm | 5-10 cm | 10-20 cm | 20-45 cm |
|---|---|---|---|---|
| P1 | 3.1 (n=116) | 6.2 (n=103) | 65.2 (n=92) | 23.6 (n=57) |
| P2 | 64.1 (n=116) | 66.5 (n=103) | 43.1 (n=92) | 30.5 (n=57) |
| rc | 0.5 (n=116) | 0.5 (n=103) | 1.3 (n=92) | 0.9 (n=57) |
| agree | 0.5 (n=116) | 0.4 (n=103) | 0.7 (n=92) | 0.5 (n=57) |
| agree+dispnet_v1 | 0.5 (n=116) | 0.4 (n=103) | 0.8 (n=92) | 0.5 (n=57) |
| agree+dispnet_v2 | 0.5 (n=116) | 0.4 (n=103) | 0.6 (n=92) | 0.6 (n=57) |
| agree→dispnet_v1+rc | 0.5 (n=116) | 0.4 (n=103) | 1.0 (n=92) | 1.0 (n=57) |
| agree→dispnet_v2+rc | 0.5 (n=116) | 0.4 (n=103) | 0.8 (n=92) | 0.6 (n=57) |
| dispnet_v1 | 1.6 (n=116) | 3.0 (n=103) | 6.2 (n=92) | 6.5 (n=57) |
| dispnet_v1+rc | 0.5 (n=116) | 0.5 (n=103) | 1.1 (n=92) | 1.0 (n=57) |
| dispnet_v2 | 1.2 (n=116) | 2.3 (n=103) | 4.9 (n=92) | 7.0 (n=57) |
| dispnet_v2+rc | 0.5 (n=116) | 0.5 (n=103) | 1.0 (n=92) | 0.9 (n=57) |
| gate_dispnet_v1 | 1.6 (n=116) | 2.8 (n=103) | 5.3 (n=92) | 5.8 (n=57) |
| gate_dispnet_v2 | 1.1 (n=116) | 2.0 (n=103) | 4.6 (n=92) | 5.8 (n=57) |

### test_t2: inputs where global render-and-compare fails (rc error > 10 cm, n=47)

| method | n | median cm | mean cm | p90 cm | <2 cm | <5 cm | <10 cm | no answer |
|---|---|---|---|---|---|---|---|---|
| P1 | 47 | 81.5 | 82.8 | 170.8 | 6% | 13% | 21% | 0 |
| P2 | 47 | 23.7 | 26.7 | 41.6 | 0% | 2% | 13% | 0 |
| rc | 47 | 18.4 | 19.3 | 24.6 | 0% | 0% | 0% | 0 |
| agree | 47 | 2.4 | 6.4 | 18.9 | 43% | 68% | 72% | 0 |
| agree+dispnet_v1 | 47 | 2.4 | 5.8 | 18.4 | 45% | 68% | 74% | 0 |
| agree+dispnet_v2 | 47 | 2.5 | 5.9 | 18.9 | 43% | 72% | 79% | 0 |
| agree→dispnet_v1+rc | 47 | 4.6 | 8.4 | 19.4 | 38% | 53% | 60% | 0 |
| agree→dispnet_v2+rc | 47 | 2.2 | 6.4 | 18.6 | 45% | 68% | 70% | 0 |
| dispnet_v1 | 47 | 6.7 | 8.5 | 16.0 | 4% | 28% | 66% | 0 |
| dispnet_v1+rc | 47 | 6.4 | 9.0 | 19.2 | 36% | 45% | 51% | 0 |
| dispnet_v2 | 47 | 6.2 | 7.1 | 12.8 | 9% | 38% | 74% | 0 |
| dispnet_v2+rc | 47 | 5.6 | 7.9 | 18.8 | 34% | 47% | 62% | 0 |
| gate_dispnet_v1 | 47 | 5.8 | 7.3 | 13.0 | 6% | 45% | 72% | 0 |
| gate_dispnet_v2 | 47 | 6.2 | 7.1 | 12.8 | 11% | 40% | 72% | 0 |

### test_t2: median error (cm) per object

| object | n | P1 | P2 | rc | agree | agree+dispnet_v1 | agree+dispnet_v2 | agree→dispnet_v1+rc | agree→dispnet_v2+rc | dispnet_v1 | dispnet_v1+rc | dispnet_v2 | dispnet_v2+rc | gate_dispnet_v1 | gate_dispnet_v2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| apple | 38 | 6.0 | 59.7 | 0.6 | 0.6 | 0.6 | 0.5 | 0.6 | 0.6 | 2.8 | 0.6 | 1.5 | 0.6 | 2.5 | 1.5 |
| book | 31 | 4.9 | 26.9 | 0.5 | 0.5 | 0.5 | 0.5 | 0.8 | 0.6 | 3.7 | 0.8 | 3.3 | 0.8 | 3.3 | 1.1 |
| bottle | 16 | 7.3 | 245.5 | 4.4 | 3.2 | 3.2 | 3.2 | 4.4 | 3.2 | 7.4 | 4.5 | 3.5 | 4.9 | 7.4 | 3.5 |
| bowl | 63 | 6.2 | 61.0 | 0.4 | 0.4 | 0.4 | 0.4 | 0.4 | 0.4 | 2.3 | 0.4 | 1.6 | 0.4 | 2.3 | 1.6 |
| bread | 55 | 15.5 | 41.9 | 0.3 | 0.3 | 0.3 | 0.3 | 0.3 | 0.3 | 2.3 | 0.4 | 3.1 | 0.4 | 2.3 | 3.1 |
| butter knife | 25 | 137.0 | 20.2 | 18.4 | 4.3 | 4.4 | 3.5 | 18.3 | 13.0 | 10.7 | 13.4 | 7.9 | 13.5 | 7.4 | 7.6 |
| coffee machine | 24 | 12.9 | 115.5 | 0.4 | 0.4 | 0.4 | 0.4 | 0.4 | 0.4 | 3.7 | 0.4 | 2.5 | 0.4 | 3.5 | 1.6 |
| credit card | 4 | 7.7 | 37.2 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 4.1 | 0.4 | 1.5 | 0.5 | 4.1 | 1.5 |
| cup | 4 | 293.9 | 39.5 | 2.1 | 1.1 | 1.1 | 1.1 | 1.1 | 1.1 | 6.5 | 2.1 | 3.9 | 1.5 | 3.4 | 2.6 |
| dish sponge | 17 | 2.9 | 89.6 | 1.2 | 1.2 | 1.2 | 1.2 | 1.1 | 1.2 | 1.9 | 1.1 | 1.4 | 1.1 | 1.9 | 1.4 |
| egg | 20 | 2.3 | 65.2 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 1.5 | 0.5 | 1.2 | 0.5 | 1.5 | 1.2 |
| garbage can | 9 | 68.5 | 20.4 | 2.2 | 0.5 | 0.5 | 0.5 | 1.2 | 1.2 | 8.2 | 1.1 | 4.1 | 1.0 | 6.1 | 4.1 |
| house plant | 39 | 5.7 | 61.1 | 0.4 | 0.3 | 0.3 | 0.3 | 0.3 | 0.3 | 7.1 | 0.6 | 3.7 | 0.4 | 6.4 | 3.6 |
| kettle | 23 | 6.3 | 23.7 | 15.1 | 1.9 | 1.9 | 1.9 | 1.7 | 1.6 | 3.2 | 1.2 | 4.1 | 1.1 | 3.0 | 4.1 |

Mean runtime per input (s, RTX 3090, shared GPU): P 0.37, raft 0.28, roma 0.98, depth 0.64, sam 0.73, rc 7.36
