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
- **inerf**: [lit] iNeRF (IROS 2021): gradient descent on the object translation through the GG renderer, photometric loss
- **lg_pnp**: [lit] DISK+LightGlue (ICCV 2023) matches + GG depth + PnP-RANSAC (pose step of 3DGS-CD, RA-L 2025)
- **roma_pnp**: [lit] RoMa (CVPR 2024) matches + GG depth + PnP-RANSAC
- **cotracker**: [lit] CoTracker3 (2024) point tracks + support-plane lifting + RANSAC
- **icp**: [lit] point-to-point ICP: GG-depth object points vs SAM mask + Depth Anything V2 points of the AFTER photo
- **mast3r**: [lit] MASt3R (ECCV 2024): metric depth (scaled to GG) + object matches, 3D differences + RANSAC
- **<model>**: DispNet prediction; **<model>+rc**: local render-and-compare (±8 cm) started at it; **agree+<model>**: agree rule with DispNet as a fifth cue; **gate_<model>**: DispNet when its predicted uncertainty is < 3 cm, else agree; **agree→<model>+rc**: agree, unless DispNet disagrees with it by > 10 cm (planar), then <model>+rc; **cotracker/<model>+rc**: CoTracker3, or <model>+rc when it gives no answer; **cotracker→<model>+rc**: CoTracker3, unless it gives no answer or disagrees with DispNet by > 10 cm (planar), then <model>+rc (both added after the T2 baseline run; threshold reused, not tuned).

## T1 — original benchmark (37 moves, 1 input view each; methods were tuned on it)

37 (trial, view) inputs, 37 moves, median |δ| 15.0 cm

| method | n | median cm | mean cm | p90 cm | <2 cm | <5 cm | <10 cm | no answer |
|---|---|---|---|---|---|---|---|---|
| no_update | 37 | 15.0 | 17.3 | 30.0 | 0% | 0% | 0% | 0 |
| P1 | 37 | 96.8 | 91.7 | 208.5 | 0% | 5% | 19% | 0 |
| P2 | 37 | 35.2 | 73.3 | 198.3 | 0% | 0% | 0% | 0 |
| inerf | 37 | 13.9 | 14.7 | 28.6 | 16% | 27% | 38% | 0 |
| lg_pnp | 37 | 15.0 | 15.2 | 30.0 | 11% | 16% | 19% | 24 |
| roma_pnp | 37 | 2.5 | 9.0 | 21.5 | 43% | 54% | 62% | 6 |
| cotracker | 37 | 1.0 | 4.4 | 15.0 | 62% | 81% | 81% | 5 |
| icp | 37 | 6.5 | 16.6 | 31.0 | 8% | 46% | 59% | 5 |
| mast3r | 37 | 15.0 | 17.4 | 30.0 | 0% | 3% | 3% | 30 |
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
| cotracker/dispnet_v1+rc | 37 | 0.9 | 3.2 | 12.9 | 68% | 86% | 86% | 0 |
| cotracker/dispnet_v2+rc | 37 | 0.9 | 3.1 | 12.0 | 68% | 86% | 86% | 0 |
| cotracker→dispnet_v1+rc | 37 | 1.2 | 5.0 | 14.4 | 54% | 70% | 76% | 0 |
| cotracker→dispnet_v2+rc | 37 | 1.0 | 3.8 | 12.3 | 62% | 81% | 86% | 0 |
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
| inerf | – | – | 11.3 (n=31) | 28.1 (n=6) |
| lg_pnp | – | – | 15.0 (n=31) | 30.0 (n=6) |
| roma_pnp | – | – | 2.4 (n=31) | 18.4 (n=6) |
| cotracker | – | – | 0.9 (n=31) | 2.7 (n=6) |
| icp | – | – | 7.8 (n=31) | 3.1 (n=6) |
| mast3r | – | – | 15.0 (n=31) | 30.0 (n=6) |
| rc | – | – | 1.3 (n=31) | 1.1 (n=6) |
| agree | – | – | 0.8 (n=31) | 1.1 (n=6) |
| agree+dispnet_v2 | – | – | 0.8 (n=31) | 3.5 (n=6) |
| agree→dispnet_v2+rc | – | – | 0.8 (n=31) | 5.6 (n=6) |
| cotracker/dispnet_v2+rc | – | – | 0.9 (n=31) | 1.7 (n=6) |
| cotracker→dispnet_v2+rc | – | – | 0.9 (n=31) | 3.8 (n=6) |
| dispnet_v2 | – | – | 6.2 (n=31) | 12.7 (n=6) |
| dispnet_v2+rc | – | – | 1.4 (n=31) | 5.4 (n=6) |

### test_t1: inputs where global render-and-compare fails (rc error > 10 cm, n=14)

| method | n | median cm | mean cm | p90 cm | <2 cm | <5 cm | <10 cm | no answer |
|---|---|---|---|---|---|---|---|---|
| P1 | 14 | 14.3 | 71.6 | 168.3 | 0% | 14% | 36% | 0 |
| P2 | 14 | 24.8 | 92.4 | 300.1 | 0% | 0% | 0% | 0 |
| inerf | 14 | 14.8 | 16.2 | 24.3 | 0% | 7% | 14% | 0 |
| lg_pnp | 14 | 15.0 | 16.7 | 28.7 | 0% | 7% | 14% | 8 |
| roma_pnp | 14 | 12.5 | 12.2 | 25.5 | 29% | 43% | 50% | 4 |
| cotracker | 14 | 2.4 | 7.5 | 15.0 | 36% | 64% | 64% | 5 |
| icp | 14 | 5.3 | 19.6 | 30.5 | 7% | 50% | 57% | 2 |
| mast3r | 14 | 15.0 | 17.1 | 25.5 | 0% | 0% | 0% | 13 |
| rc | 14 | 15.0 | 17.6 | 29.0 | 0% | 0% | 0% | 0 |
| agree | 14 | 3.0 | 4.4 | 12.1 | 36% | 71% | 86% | 0 |
| agree+dispnet_v2 | 14 | 3.5 | 5.0 | 13.4 | 21% | 79% | 86% | 0 |
| agree→dispnet_v2+rc | 14 | 3.0 | 4.9 | 14.0 | 36% | 71% | 79% | 0 |
| cotracker/dispnet_v2+rc | 14 | 1.7 | 4.0 | 12.7 | 50% | 79% | 79% | 0 |
| cotracker→dispnet_v2+rc | 14 | 1.7 | 4.0 | 12.7 | 50% | 79% | 79% | 0 |
| dispnet_v2 | 14 | 6.0 | 5.6 | 8.9 | 14% | 43% | 100% | 0 |
| dispnet_v2+rc | 14 | 6.9 | 7.4 | 14.8 | 21% | 43% | 64% | 0 |

### test_t1: median error (cm) per object

| object | n | P1 | P2 | inerf | lg_pnp | roma_pnp | cotracker | icp | mast3r | rc | agree | agree+dispnet_v2 | agree→dispnet_v2+rc | cotracker/dispnet_v2+rc | cotracker→dispnet_v2+rc | dispnet_v2 | dispnet_v2+rc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| apple | 1 | 15.3 | 16.8 | 8.8 | 15.0 | 15.0 | 0.9 | 4.2 | 15.0 | 0.6 | 0.6 | 0.6 | 0.6 | 0.9 | 0.9 | 1.0 | 0.5 |
| book | 4 | 112.8 | 16.2 | 16.0 | 15.0 | 15.6 | 0.7 | 9.3 | 15.0 | 0.1 | 0.3 | 0.1 | 0.3 | 0.7 | 0.2 | 9.9 | 0.3 |
| bottle | 3 | 9.0 | 334.8 | 19.5 | 15.0 | 1.4 | 2.4 | 2.8 | 15.0 | 15.0 | 0.8 | 1.1 | 0.8 | 2.4 | 2.4 | 6.4 | 9.7 |
| bowl | 8 | 113.1 | 66.4 | 19.7 | 15.0 | 2.0 | 0.7 | 4.0 | 15.0 | 0.6 | 0.6 | 0.6 | 0.6 | 0.7 | 0.7 | 5.2 | 0.8 |
| bread | 4 | 18.2 | 17.1 | 17.4 | 0.4 | 0.8 | 0.2 | 2.6 | 19.5 | 0.1 | 0.1 | 0.1 | 0.1 | 0.2 | 0.2 | 6.0 | 0.1 |
| butter knife | 4 | 158.9 | 15.5 | 12.4 | 15.0 | 15.0 | 15.0 | 4.5 | 15.0 | 15.0 | 9.8 | 9.8 | 8.2 | 12.4 | 12.4 | 6.2 | 12.4 |
| coffee machine | 4 | 281.6 | 45.4 | 4.5 | 6.7 | 1.2 | 1.7 | 71.5 | 14.9 | 2.6 | 1.3 | 1.3 | 2.0 | 1.7 | 1.9 | 8.8 | 8.9 |
| dish sponge | 2 | 7.6 | 99.5 | 22.7 | 22.5 | 22.5 | 15.5 | 2.6 | 22.5 | 22.5 | 4.0 | 3.6 | 4.0 | 0.8 | 0.8 | 1.0 | 1.0 |
| house plant | 3 | 151.2 | 80.2 | 0.1 | 15.0 | 0.5 | 0.6 | 15.0 | 15.0 | 0.0 | 0.0 | 0.9 | 6.3 | 0.6 | 6.3 | 14.5 | 6.3 |
| kettle | 4 | 9.5 | 14.9 | 16.2 | 17.7 | 2.7 | 1.7 | 19.0 | 15.0 | 15.0 | 2.2 | 2.6 | 2.2 | 1.7 | 1.7 | 5.9 | 2.6 |

Mean runtime per input (s, RTX 3090, shared GPU): P 1.37, raft 0.68, roma 3.13, depth 3.15, sam 2.89, rc 10.19, inerf 10.63, lg_pnp 0.11, roma_pnp 1.38, cotracker 0.44, icp 1.59, mast3r 1.63

## T2 — new held-out set (67 random moves, every visible view)

368 (trial, view) inputs, 67 moves, median |δ| 7.3 cm

| method | n | median cm | mean cm | p90 cm | <2 cm | <5 cm | <10 cm | no answer |
|---|---|---|---|---|---|---|---|---|
| no_update | 368 | 7.3 | 11.0 | 20.7 | 0% | 32% | 60% | 0 |
| P1 | 368 | 6.6 | 48.4 | 174.7 | 15% | 40% | 60% | 0 |
| P2 | 368 | 52.4 | 68.2 | 134.5 | 1% | 3% | 8% | 0 |
| inerf | 368 | 5.1 | 9.0 | 26.9 | 30% | 49% | 72% | 0 |
| lg_pnp | 368 | 3.5 | 7.0 | 19.5 | 38% | 61% | 73% | 151 |
| roma_pnp | 368 | 1.5 | 7.3 | 19.7 | 56% | 74% | 79% | 36 |
| cotracker | 368 | 0.3 | 1.9 | 3.1 | 88% | 92% | 92% | 33 |
| icp | 368 | 5.2 | 10.7 | 19.7 | 11% | 46% | 74% | 94 |
| mast3r | 368 | 7.7 | 11.4 | 21.3 | 0% | 30% | 58% | 351 |
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
| cotracker/dispnet_v1+rc | 368 | 0.3 | 1.6 | 2.1 | 89% | 93% | 94% | 0 |
| cotracker/dispnet_v2+rc | 368 | 0.3 | 1.4 | 2.1 | 90% | 93% | 95% | 0 |
| cotracker→dispnet_v1+rc | 368 | 0.3 | 1.8 | 4.7 | 86% | 90% | 94% | 0 |
| cotracker→dispnet_v2+rc | 368 | 0.3 | 1.6 | 4.4 | 86% | 90% | 95% | 0 |
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
| inerf | 4.2 (n=116) | 2.4 (n=103) | 12.3 (n=92) | 25.6 (n=57) |
| lg_pnp | 3.2 (n=116) | 2.3 (n=103) | 11.8 (n=92) | 17.4 (n=57) |
| roma_pnp | 0.7 (n=116) | 0.9 (n=103) | 4.6 (n=92) | 20.1 (n=57) |
| cotracker | 0.3 (n=116) | 0.3 (n=103) | 0.6 (n=92) | 0.3 (n=57) |
| icp | 3.8 (n=116) | 6.5 (n=103) | 8.0 (n=92) | 7.3 (n=57) |
| mast3r | 3.7 (n=116) | 7.1 (n=103) | 15.9 (n=92) | 21.7 (n=57) |
| rc | 0.5 (n=116) | 0.5 (n=103) | 1.3 (n=92) | 0.9 (n=57) |
| agree | 0.5 (n=116) | 0.4 (n=103) | 0.7 (n=92) | 0.5 (n=57) |
| agree+dispnet_v2 | 0.5 (n=116) | 0.4 (n=103) | 0.6 (n=92) | 0.6 (n=57) |
| agree→dispnet_v2+rc | 0.5 (n=116) | 0.4 (n=103) | 0.8 (n=92) | 0.6 (n=57) |
| cotracker/dispnet_v2+rc | 0.3 (n=116) | 0.3 (n=103) | 0.5 (n=92) | 0.3 (n=57) |
| cotracker→dispnet_v2+rc | 0.3 (n=116) | 0.3 (n=103) | 0.6 (n=92) | 0.3 (n=57) |
| dispnet_v2 | 1.2 (n=116) | 2.3 (n=103) | 4.9 (n=92) | 7.0 (n=57) |
| dispnet_v2+rc | 0.5 (n=116) | 0.5 (n=103) | 1.0 (n=92) | 0.9 (n=57) |

### test_t2: inputs where global render-and-compare fails (rc error > 10 cm, n=47)

| method | n | median cm | mean cm | p90 cm | <2 cm | <5 cm | <10 cm | no answer |
|---|---|---|---|---|---|---|---|---|
| P1 | 47 | 81.5 | 82.8 | 170.8 | 6% | 13% | 21% | 0 |
| P2 | 47 | 23.7 | 26.7 | 41.6 | 0% | 2% | 13% | 0 |
| inerf | 47 | 19.5 | 20.8 | 30.5 | 2% | 2% | 11% | 0 |
| lg_pnp | 47 | 17.4 | 15.4 | 24.1 | 13% | 19% | 23% | 29 |
| roma_pnp | 47 | 18.4 | 15.4 | 23.5 | 11% | 17% | 23% | 19 |
| cotracker | 47 | 13.0 | 10.0 | 19.7 | 43% | 45% | 45% | 25 |
| icp | 47 | 8.6 | 15.2 | 30.3 | 9% | 38% | 53% | 6 |
| mast3r | 47 | 19.5 | 19.7 | 24.5 | 0% | 0% | 0% | 47 |
| rc | 47 | 18.4 | 19.3 | 24.6 | 0% | 0% | 0% | 0 |
| agree | 47 | 2.4 | 6.4 | 18.9 | 43% | 68% | 72% | 0 |
| agree+dispnet_v2 | 47 | 2.5 | 5.9 | 18.9 | 43% | 72% | 79% | 0 |
| agree→dispnet_v2+rc | 47 | 2.2 | 6.4 | 18.6 | 45% | 68% | 70% | 0 |
| cotracker/dispnet_v2+rc | 47 | 3.3 | 7.1 | 18.8 | 47% | 55% | 66% | 0 |
| cotracker→dispnet_v2+rc | 47 | 3.5 | 7.2 | 18.8 | 43% | 53% | 66% | 0 |
| dispnet_v2 | 47 | 6.2 | 7.1 | 12.8 | 9% | 38% | 74% | 0 |
| dispnet_v2+rc | 47 | 5.6 | 7.9 | 18.8 | 34% | 47% | 62% | 0 |

### test_t2: median error (cm) per object

| object | n | P1 | P2 | inerf | lg_pnp | roma_pnp | cotracker | icp | mast3r | rc | agree | agree+dispnet_v2 | agree→dispnet_v2+rc | cotracker/dispnet_v2+rc | cotracker→dispnet_v2+rc | dispnet_v2 | dispnet_v2+rc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| apple | 38 | 6.0 | 59.7 | 8.1 | 6.1 | 1.6 | 0.4 | 3.7 | 7.1 | 0.6 | 0.6 | 0.5 | 0.6 | 0.4 | 0.5 | 1.5 | 0.6 |
| book | 31 | 4.9 | 26.9 | 2.4 | 0.9 | 1.6 | 0.2 | 5.2 | 6.3 | 0.5 | 0.5 | 0.5 | 0.6 | 0.2 | 0.2 | 3.3 | 0.8 |
| bottle | 16 | 7.3 | 245.5 | 3.9 | 5.5 | 1.3 | 0.9 | 7.7 | 7.2 | 4.4 | 3.2 | 3.2 | 3.2 | 0.9 | 1.2 | 3.5 | 4.9 |
| bowl | 63 | 6.2 | 61.0 | 3.3 | 8.8 | 2.0 | 0.2 | 4.5 | 8.8 | 0.4 | 0.4 | 0.4 | 0.4 | 0.2 | 0.2 | 1.6 | 0.4 |
| bread | 55 | 15.5 | 41.9 | 5.2 | 0.9 | 0.5 | 0.2 | 4.8 | 15.9 | 0.3 | 0.3 | 0.3 | 0.3 | 0.2 | 0.2 | 3.1 | 0.4 |
| butter knife | 25 | 137.0 | 20.2 | 18.4 | 18.4 | 18.5 | 18.4 | 4.7 | 18.4 | 18.4 | 4.3 | 3.5 | 13.0 | 13.4 | 13.4 | 7.9 | 13.5 |
| coffee machine | 24 | 12.9 | 115.5 | 2.3 | 0.7 | 0.4 | 0.3 | 10.4 | 9.1 | 0.4 | 0.4 | 0.4 | 0.4 | 0.3 | 0.3 | 2.5 | 0.4 |
| credit card | 4 | 7.7 | 37.2 | 7.1 | 9.6 | 9.6 | 0.3 | 5.6 | 5.8 | 0.5 | 0.5 | 0.5 | 0.5 | 0.3 | 0.3 | 1.5 | 0.5 |
| cup | 4 | 293.9 | 39.5 | 11.5 | 5.6 | 0.3 | 0.3 | 50.2 | 5.6 | 2.1 | 1.1 | 1.1 | 1.1 | 0.3 | 0.3 | 3.9 | 1.5 |
| dish sponge | 17 | 2.9 | 89.6 | 8.6 | 4.6 | 4.1 | 0.5 | 3.5 | 4.6 | 1.2 | 1.2 | 1.2 | 1.2 | 0.5 | 0.5 | 1.4 | 1.1 |
| egg | 20 | 2.3 | 65.2 | 5.5 | 3.3 | 2.8 | 0.2 | 3.3 | 3.3 | 0.5 | 0.5 | 0.5 | 0.5 | 0.2 | 0.2 | 1.2 | 0.5 |
| garbage can | 9 | 68.5 | 20.4 | 18.7 | 5.3 | 2.1 | 0.3 | 23.4 | 18.2 | 2.2 | 0.5 | 0.5 | 1.2 | 0.3 | 0.6 | 4.1 | 1.0 |
| house plant | 39 | 5.7 | 61.1 | 0.7 | 0.4 | 0.2 | 0.4 | 5.2 | 8.8 | 0.4 | 0.3 | 0.3 | 0.3 | 0.4 | 0.4 | 3.7 | 0.4 |
| kettle | 23 | 6.3 | 23.7 | 19.3 | 5.9 | 5.0 | 1.1 | 22.5 | 15.1 | 15.1 | 1.9 | 1.9 | 1.6 | 0.9 | 1.1 | 4.1 | 1.1 |

Mean runtime per input (s, RTX 3090, shared GPU): P 0.37, raft 0.28, roma 0.98, depth 0.64, sam 0.73, rc 7.36, inerf 8.67, lg_pnp 0.05, roma_pnp 0.75, cotracker 0.37, icp 1.03, mast3r 0.99


## Rendering metrics — T1 (37 inputs rendered by every method)

The object's Gaussians are moved by the estimate and every view is rendered and compared with the real AFTER photo. ROI = dilated union of the true BEFORE/AFTER object masks; LPIPS on the ROI bounding box; IoU = rendered visible silhouette vs. true AFTER mask. 'novel' = mean over the other views that see the object. gt = true displacement (ceiling for this GG model + selection). ↑ higher is better, ↓ lower is better.

| method | err median cm | input PSNR-ROI ↑ | input SSIM-ROI ↑ | input LPIPS-ROI ↓ | input IoU ↑ | novel PSNR-ROI ↑ | novel SSIM-ROI ↑ | novel LPIPS-ROI ↓ | novel IoU ↑ | novel PSNR full ↑ |
|---|---|---|---|---|---|---|---|---|---|---|
| gt | 0.0 | 21.11 | 0.707 | 0.172 | 0.737 | 20.97 | 0.717 | 0.155 | 0.745 | 31.83 |
| no_update | 15.0 | 14.48 | 0.428 | 0.331 | 0.169 | 14.80 | 0.417 | 0.314 | 0.229 | 30.54 |
| P1 | 96.8 | 16.98 | 0.543 | 0.280 | 0.238 | 16.19 | 0.465 | 0.300 | 0.160 | 29.80 |
| inerf | 13.9 | 18.11 | 0.575 | 0.242 | 0.377 | 16.95 | 0.505 | 0.264 | 0.314 | 31.13 |
| lg_pnp | 15.0 | 16.12 | 0.494 | 0.297 | 0.303 | 15.99 | 0.473 | 0.277 | 0.322 | 30.87 |
| roma_pnp | 2.5 | 18.55 | 0.584 | 0.229 | 0.510 | 18.12 | 0.564 | 0.222 | 0.505 | 31.31 |
| cotracker | 1.0 | 19.97 | 0.635 | 0.193 | 0.653 | 19.66 | 0.638 | 0.181 | 0.666 | 31.62 |
| icp | 6.5 | 17.45 | 0.512 | 0.244 | 0.489 | 16.77 | 0.469 | 0.249 | 0.422 | 31.00 |
| mast3r | 15.0 | 14.70 | 0.433 | 0.329 | 0.172 | 14.97 | 0.421 | 0.313 | 0.233 | 30.54 |
| raft_crop | 0.9 | 19.97 | 0.632 | 0.191 | 0.657 | 19.62 | 0.636 | 0.181 | 0.652 | 31.62 |
| roma | 0.8 | 18.85 | 0.601 | 0.226 | 0.543 | 18.79 | 0.600 | 0.210 | 0.572 | 31.39 |
| depth | 5.4 | 17.36 | 0.515 | 0.251 | 0.439 | 17.10 | 0.480 | 0.236 | 0.439 | 31.12 |
| sam | 6.5 | 17.16 | 0.513 | 0.262 | 0.446 | 17.11 | 0.490 | 0.238 | 0.464 | 31.04 |
| rc | 1.3 | 19.90 | 0.631 | 0.194 | 0.588 | 19.15 | 0.614 | 0.196 | 0.578 | 31.69 |
| agree | 0.8 | 20.50 | 0.661 | 0.181 | 0.705 | 19.79 | 0.639 | 0.173 | 0.696 | 31.70 |
| agree+dispnet_v2 | 0.9 | 20.69 | 0.669 | 0.178 | 0.694 | 19.80 | 0.641 | 0.171 | 0.685 | 31.70 |
| agree→dispnet_v2+rc | 1.3 | 19.96 | 0.637 | 0.188 | 0.653 | 19.52 | 0.621 | 0.179 | 0.657 | 31.59 |
| cotracker/dispnet_v2+rc | 0.9 | 20.12 | 0.643 | 0.186 | 0.668 | 19.83 | 0.646 | 0.173 | 0.685 | 31.63 |
| cotracker→dispnet_v2+rc | 1.0 | 20.04 | 0.643 | 0.188 | 0.658 | 19.86 | 0.645 | 0.174 | 0.681 | 31.63 |
| dispnet_v2 | 6.8 | 17.40 | 0.532 | 0.254 | 0.475 | 17.43 | 0.505 | 0.230 | 0.528 | 31.12 |
| dispnet_v2+rc | 1.8 | 19.96 | 0.637 | 0.192 | 0.621 | 19.44 | 0.619 | 0.185 | 0.634 | 31.58 |

## Rendering metrics — T2 (368 inputs rendered by every method)

The object's Gaussians are moved by the estimate and every view is rendered and compared with the real AFTER photo. ROI = dilated union of the true BEFORE/AFTER object masks; LPIPS on the ROI bounding box; IoU = rendered visible silhouette vs. true AFTER mask. 'novel' = mean over the other views that see the object. gt = true displacement (ceiling for this GG model + selection). ↑ higher is better, ↓ lower is better.

| method | err median cm | input PSNR-ROI ↑ | input SSIM-ROI ↑ | input LPIPS-ROI ↓ | input IoU ↑ | novel PSNR-ROI ↑ | novel SSIM-ROI ↑ | novel LPIPS-ROI ↓ | novel IoU ↑ | novel PSNR full ↑ |
|---|---|---|---|---|---|---|---|---|---|---|
| gt | 0.0 | 22.89 | 0.765 | 0.124 | 0.813 | 23.00 | 0.769 | 0.121 | 0.815 | 31.47 |
| no_update | 7.3 | 15.93 | 0.435 | 0.245 | 0.361 | 16.01 | 0.442 | 0.244 | 0.358 | 30.36 |
| P1 | 6.6 | 20.20 | 0.638 | 0.172 | 0.553 | 18.39 | 0.543 | 0.206 | 0.448 | 30.50 |
| inerf | 5.1 | 20.86 | 0.656 | 0.166 | 0.613 | 19.27 | 0.576 | 0.192 | 0.521 | 31.01 |
| lg_pnp | 3.5 | 19.57 | 0.616 | 0.186 | 0.570 | 18.92 | 0.574 | 0.195 | 0.543 | 31.01 |
| roma_pnp | 1.5 | 21.07 | 0.679 | 0.163 | 0.652 | 20.36 | 0.639 | 0.169 | 0.626 | 31.16 |
| cotracker | 0.3 | 22.36 | 0.732 | 0.131 | 0.774 | 22.23 | 0.722 | 0.129 | 0.771 | 31.39 |
| icp | 5.2 | 18.52 | 0.530 | 0.177 | 0.606 | 17.72 | 0.498 | 0.203 | 0.522 | 30.69 |
| mast3r | 7.7 | 15.98 | 0.436 | 0.246 | 0.362 | 16.03 | 0.442 | 0.246 | 0.353 | 30.34 |
| raft_crop | 0.4 | 22.18 | 0.722 | 0.132 | 0.766 | 22.01 | 0.712 | 0.132 | 0.759 | 31.38 |
| roma | 0.2 | 21.38 | 0.691 | 0.159 | 0.668 | 21.42 | 0.689 | 0.156 | 0.675 | 31.25 |
| depth | 6.7 | 17.54 | 0.481 | 0.228 | 0.450 | 17.60 | 0.489 | 0.222 | 0.451 | 30.59 |
| sam | 2.9 | 18.91 | 0.554 | 0.173 | 0.620 | 18.99 | 0.557 | 0.174 | 0.606 | 30.87 |
| rc | 0.6 | 21.86 | 0.699 | 0.139 | 0.734 | 21.37 | 0.677 | 0.142 | 0.720 | 31.27 |
| agree | 0.5 | 22.16 | 0.716 | 0.129 | 0.784 | 21.73 | 0.694 | 0.132 | 0.769 | 31.33 |
| agree+dispnet_v2 | 0.5 | 22.18 | 0.717 | 0.128 | 0.788 | 21.75 | 0.695 | 0.131 | 0.771 | 31.34 |
| agree→dispnet_v2+rc | 0.5 | 22.11 | 0.713 | 0.130 | 0.778 | 21.67 | 0.690 | 0.133 | 0.764 | 31.31 |
| cotracker/dispnet_v2+rc | 0.3 | 22.46 | 0.737 | 0.129 | 0.782 | 22.30 | 0.725 | 0.128 | 0.778 | 31.39 |
| cotracker→dispnet_v2+rc | 0.3 | 22.39 | 0.732 | 0.130 | 0.778 | 22.20 | 0.718 | 0.130 | 0.772 | 31.37 |
| dispnet_v2 | 2.8 | 19.47 | 0.577 | 0.171 | 0.639 | 19.55 | 0.574 | 0.166 | 0.642 | 30.93 |
| dispnet_v2+rc | 0.6 | 22.06 | 0.709 | 0.131 | 0.772 | 21.59 | 0.684 | 0.135 | 0.752 | 31.29 |