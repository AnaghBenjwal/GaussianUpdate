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
| agree+dispnet_v3 | 37 | 0.9 | 2.5 | 5.1 | 68% | 89% | 92% | 0 |
| agree→dispnet_v3+rc | 37 | 1.3 | 4.0 | 15.0 | 57% | 73% | 81% | 0 |
| cotracker/dispnet_v3+rc | 37 | 0.9 | 2.9 | 6.8 | 70% | 89% | 89% | 0 |
| cotracker→dispnet_v3+rc | 37 | 1.0 | 3.5 | 11.8 | 62% | 78% | 84% | 0 |
| dispnet_v3 | 37 | 5.6 | 7.0 | 15.9 | 27% | 46% | 78% | 0 |
| dispnet_v3 (sim inputs) | 37 | 5.1 | 6.8 | 16.2 | 19% | 49% | 81% | 0 |
| dispnet_v3+rc | 37 | 1.4 | 4.6 | 14.0 | 59% | 65% | 78% | 0 |
| gate_dispnet_v3 | 37 | 4.8 | 5.1 | 10.0 | 35% | 57% | 89% | 0 |

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
| agree+dispnet_v3 | – | – | 0.9 (n=31) | 1.1 (n=6) |
| agree→dispnet_v3+rc | – | – | 0.8 (n=31) | 6.2 (n=6) |
| cotracker/dispnet_v3+rc | – | – | 0.9 (n=31) | 1.7 (n=6) |
| cotracker→dispnet_v3+rc | – | – | 0.9 (n=31) | 2.7 (n=6) |
| dispnet_v3 | – | – | 4.9 (n=31) | 9.0 (n=6) |
| dispnet_v3+rc | – | – | 1.3 (n=31) | 4.6 (n=6) |

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
| agree+dispnet_v3 | 14 | 3.0 | 4.5 | 12.1 | 36% | 79% | 86% | 0 |
| agree→dispnet_v3+rc | 14 | 3.0 | 5.6 | 15.0 | 36% | 64% | 71% | 0 |
| cotracker/dispnet_v3+rc | 14 | 1.7 | 3.4 | 11.4 | 57% | 86% | 86% | 0 |
| cotracker→dispnet_v3+rc | 14 | 2.1 | 4.1 | 13.9 | 50% | 79% | 79% | 0 |
| dispnet_v3 | 14 | 4.1 | 5.4 | 10.3 | 29% | 64% | 86% | 0 |
| dispnet_v3+rc | 14 | 7.2 | 7.3 | 15.0 | 43% | 43% | 57% | 0 |

### test_t1: median error (cm) per object

| object | n | P1 | P2 | inerf | lg_pnp | roma_pnp | cotracker | icp | mast3r | rc | agree | agree+dispnet_v3 | agree→dispnet_v3+rc | cotracker/dispnet_v3+rc | cotracker→dispnet_v3+rc | dispnet_v3 | dispnet_v3+rc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| apple | 1 | 15.3 | 16.8 | 8.8 | 15.0 | 15.0 | 0.9 | 4.2 | 15.0 | 0.6 | 0.6 | 0.6 | 0.6 | 0.9 | 0.9 | 0.3 | 1.2 |
| book | 4 | 112.8 | 16.2 | 16.0 | 15.0 | 15.6 | 0.7 | 9.3 | 15.0 | 0.1 | 0.3 | 0.3 | 9.1 | 0.7 | 9.1 | 13.9 | 8.3 |
| bottle | 3 | 9.0 | 334.8 | 19.5 | 15.0 | 1.4 | 2.4 | 2.8 | 15.0 | 15.0 | 0.8 | 1.1 | 0.8 | 2.4 | 2.4 | 3.7 | 10.7 |
| bowl | 8 | 113.1 | 66.4 | 19.7 | 15.0 | 2.0 | 0.7 | 4.0 | 15.0 | 0.6 | 0.6 | 0.6 | 0.6 | 0.7 | 0.5 | 3.6 | 0.5 |
| bread | 4 | 18.2 | 17.1 | 17.4 | 0.4 | 0.8 | 0.2 | 2.6 | 19.5 | 0.1 | 0.1 | 0.1 | 0.1 | 0.2 | 0.2 | 5.8 | 0.1 |
| butter knife | 4 | 158.9 | 15.5 | 12.4 | 15.0 | 15.0 | 15.0 | 4.5 | 15.0 | 15.0 | 9.8 | 9.8 | 9.2 | 8.5 | 8.5 | 6.4 | 8.5 |
| coffee machine | 4 | 281.6 | 45.4 | 4.5 | 6.7 | 1.2 | 1.7 | 71.5 | 14.9 | 2.6 | 1.3 | 1.3 | 4.1 | 1.7 | 3.9 | 13.0 | 4.1 |
| dish sponge | 2 | 7.6 | 99.5 | 22.7 | 22.5 | 22.5 | 15.5 | 2.6 | 22.5 | 22.5 | 4.0 | 3.6 | 4.0 | 0.9 | 0.9 | 1.2 | 1.1 |
| house plant | 3 | 151.2 | 80.2 | 0.1 | 15.0 | 0.5 | 0.6 | 15.0 | 15.0 | 0.0 | 0.0 | 0.0 | 0.9 | 0.6 | 0.6 | 9.9 | 1.9 |
| kettle | 4 | 9.5 | 14.9 | 16.2 | 17.7 | 2.7 | 1.7 | 19.0 | 15.0 | 15.0 | 2.2 | 2.3 | 2.2 | 1.7 | 1.7 | 4.1 | 4.5 |

Mean runtime per input (s, RTX 3090, shared GPU): P 1.37, raft 0.68, roma 3.13, depth 3.15, sam 2.89, rc 10.19, inerf 10.63, lg_pnp 0.11, roma_pnp 1.38, cotracker 0.44, icp 1.59, mast3r 1.63

## T2 — held-out random moves, every visible view

891 (trial, view) inputs, 172 moves, median |δ| 8.3 cm

| method | n | median cm | mean cm | p90 cm | <2 cm | <5 cm | <10 cm | no answer |
|---|---|---|---|---|---|---|---|---|
| no_update | 891 | 8.3 | 11.2 | 21.4 | 0% | 29% | 59% | 0 |
| P1 | 891 | 9.9 | 55.0 | 190.3 | 11% | 32% | 51% | 0 |
| P2 | 891 | 63.9 | 100.6 | 257.8 | 0% | 5% | 10% | 0 |
| inerf | 891 | 4.8 | 9.3 | 27.4 | 29% | 51% | 71% | 0 |
| lg_pnp | 891 | 3.7 | 7.7 | 20.5 | 38% | 57% | 71% | 318 |
| roma_pnp | 891 | 1.5 | 7.2 | 20.7 | 53% | 67% | 75% | 108 |
| cotracker | 891 | 0.3 | 2.2 | 3.7 | 86% | 91% | 93% | 75 |
| icp | 891 | 5.4 | 14.1 | 27.6 | 11% | 46% | 71% | 208 |
| mast3r | 891 | 8.8 | 11.6 | 22.3 | 1% | 27% | 57% | 826 |
| raft_orig | 891 | 52.3 | 235.9 | 380.1 | 1% | 8% | 19% | 0 |
| raft_crop | 891 | 0.4 | 2.4 | 4.4 | 82% | 90% | 93% | 87 |
| roma | 891 | 0.2 | 4.8 | 19.5 | 70% | 76% | 80% | 247 |
| depth | 891 | 6.9 | 14.8 | 28.1 | 6% | 36% | 62% | 90 |
| sam | 891 | 3.0 | 13.4 | 20.5 | 40% | 69% | 81% | 218 |
| rc | 891 | 0.5 | 3.8 | 13.0 | 72% | 83% | 88% | 0 |
| agree | 891 | 0.5 | 1.8 | 3.7 | 80% | 94% | 96% | 0 |
| agree+dispnet_v3 | 891 | 0.5 | 1.6 | 3.5 | 81% | 95% | 97% | 0 |
| agree→dispnet_v3+rc | 891 | 0.5 | 2.1 | 4.7 | 76% | 91% | 95% | 0 |
| cotracker/dispnet_v3+rc | 891 | 0.3 | 1.6 | 2.7 | 88% | 93% | 95% | 0 |
| cotracker→dispnet_v3+rc | 891 | 0.4 | 1.9 | 4.8 | 83% | 90% | 94% | 0 |
| dispnet_v3 | 891 | 2.6 | 4.1 | 9.6 | 41% | 74% | 90% | 0 |
| dispnet_v3 (sim inputs) | 891 | 2.4 | 4.0 | 9.8 | 43% | 75% | 90% | 0 |
| dispnet_v3+rc | 891 | 0.6 | 2.3 | 6.3 | 75% | 87% | 94% | 0 |
| gate_dispnet_v3 | 891 | 2.3 | 3.6 | 8.0 | 46% | 79% | 93% | 0 |

### test_t2: median error (cm) by move size

| method | 0-5 cm | 5-10 cm | 10-20 cm | 20-45 cm |
|---|---|---|---|---|
| P1 | 3.8 (n=260) | 6.5 (n=266) | 16.2 (n=217) | 35.3 (n=148) |
| P2 | 58.8 (n=260) | 70.3 (n=266) | 68.0 (n=217) | 47.3 (n=148) |
| inerf | 3.2 (n=260) | 2.7 (n=266) | 9.1 (n=217) | 26.5 (n=148) |
| lg_pnp | 3.2 (n=260) | 4.1 (n=266) | 10.5 (n=217) | 20.5 (n=148) |
| roma_pnp | 0.7 (n=260) | 1.0 (n=266) | 5.2 (n=217) | 20.7 (n=148) |
| cotracker | 0.3 (n=260) | 0.3 (n=266) | 0.5 (n=217) | 0.5 (n=148) |
| icp | 4.2 (n=260) | 6.3 (n=266) | 8.6 (n=217) | 4.8 (n=148) |
| mast3r | 3.8 (n=260) | 7.1 (n=266) | 13.9 (n=217) | 24.1 (n=148) |
| rc | 0.5 (n=260) | 0.5 (n=266) | 0.7 (n=217) | 0.9 (n=148) |
| agree | 0.5 (n=260) | 0.4 (n=266) | 0.5 (n=217) | 0.5 (n=148) |
| agree+dispnet_v3 | 0.5 (n=260) | 0.5 (n=266) | 0.5 (n=217) | 0.5 (n=148) |
| agree→dispnet_v3+rc | 0.5 (n=260) | 0.5 (n=266) | 0.6 (n=217) | 1.0 (n=148) |
| cotracker/dispnet_v3+rc | 0.3 (n=260) | 0.3 (n=266) | 0.5 (n=217) | 0.5 (n=148) |
| cotracker→dispnet_v3+rc | 0.3 (n=260) | 0.3 (n=266) | 0.6 (n=217) | 0.8 (n=148) |
| dispnet_v3 | 1.2 (n=260) | 2.3 (n=266) | 4.3 (n=217) | 5.6 (n=148) |
| dispnet_v3+rc | 0.5 (n=260) | 0.5 (n=266) | 0.8 (n=217) | 1.0 (n=148) |

### test_t2: inputs where global render-and-compare fails (rc error > 10 cm, n=107)

| method | n | median cm | mean cm | p90 cm | <2 cm | <5 cm | <10 cm | no answer |
|---|---|---|---|---|---|---|---|---|
| P1 | 107 | 65.7 | 88.8 | 193.1 | 4% | 10% | 16% | 0 |
| P2 | 107 | 36.0 | 97.4 | 282.5 | 0% | 2% | 7% | 0 |
| inerf | 107 | 19.7 | 21.2 | 39.3 | 5% | 7% | 16% | 0 |
| lg_pnp | 107 | 18.4 | 17.1 | 36.3 | 16% | 22% | 27% | 62 |
| roma_pnp | 107 | 18.4 | 17.2 | 36.1 | 14% | 20% | 24% | 39 |
| cotracker | 107 | 1.3 | 9.1 | 20.7 | 57% | 59% | 60% | 37 |
| icp | 107 | 9.2 | 22.5 | 50.9 | 9% | 36% | 51% | 14 |
| mast3r | 107 | 20.1 | 21.5 | 37.3 | 0% | 1% | 3% | 106 |
| rc | 107 | 19.9 | 22.4 | 36.5 | 0% | 0% | 0% | 0 |
| agree | 107 | 2.1 | 7.3 | 20.3 | 49% | 69% | 72% | 0 |
| agree+dispnet_v3 | 107 | 1.7 | 5.3 | 18.7 | 58% | 79% | 81% | 0 |
| agree→dispnet_v3+rc | 107 | 2.7 | 8.2 | 20.6 | 41% | 58% | 64% | 0 |
| cotracker/dispnet_v3+rc | 107 | 1.2 | 6.4 | 18.4 | 62% | 68% | 72% | 0 |
| cotracker→dispnet_v3+rc | 107 | 1.8 | 8.3 | 20.6 | 50% | 57% | 62% | 0 |
| dispnet_v3 | 107 | 7.2 | 8.9 | 18.7 | 13% | 39% | 66% | 0 |
| dispnet_v3+rc | 107 | 6.7 | 9.0 | 20.6 | 34% | 49% | 60% | 0 |

### test_t2: median error (cm) per object

| object | n | P1 | P2 | inerf | lg_pnp | roma_pnp | cotracker | icp | mast3r | rc | agree | agree+dispnet_v3 | agree→dispnet_v3+rc | cotracker/dispnet_v3+rc | cotracker→dispnet_v3+rc | dispnet_v3 | dispnet_v3+rc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| apple | 69 | 5.5 | 59.0 | 6.5 | 6.1 | 1.0 | 0.4 | 3.1 | 7.1 | 0.4 | 0.4 | 0.4 | 0.5 | 0.4 | 0.4 | 1.8 | 0.5 |
| book | 31 | 4.9 | 26.9 | 2.4 | 0.9 | 1.6 | 0.2 | 5.2 | 6.3 | 0.5 | 0.5 | 0.5 | 0.8 | 0.2 | 0.2 | 1.6 | 0.9 |
| bottle | 16 | 7.3 | 245.5 | 3.9 | 5.5 | 1.3 | 0.9 | 7.7 | 7.2 | 4.4 | 3.2 | 3.2 | 3.2 | 0.9 | 0.9 | 3.8 | 4.1 |
| bowl | 139 | 28.5 | 82.5 | 3.0 | 9.3 | 2.8 | 0.2 | 5.4 | 8.9 | 0.4 | 0.4 | 0.4 | 0.4 | 0.2 | 0.3 | 1.7 | 0.4 |
| bread | 141 | 17.9 | 49.7 | 5.5 | 0.7 | 0.4 | 0.2 | 5.3 | 14.2 | 0.3 | 0.3 | 0.3 | 0.3 | 0.2 | 0.2 | 2.0 | 0.3 |
| butter knife | 47 | 117.3 | 12.0 | 15.0 | 13.0 | 13.0 | 13.0 | 3.9 | 13.0 | 13.0 | 3.6 | 3.5 | 3.9 | 4.7 | 4.7 | 4.9 | 4.7 |
| cell phone | 24 | 6.1 | 47.9 | 2.8 | 9.8 | 5.4 | 0.4 | 4.1 | 5.1 | 2.5 | 1.4 | 1.4 | 1.4 | 0.4 | 0.4 | 1.5 | 1.2 |
| chair | 13 | 6.7 | 34.8 | 7.5 | 7.7 | 1.5 | 0.5 | 19.9 | 5.4 | 5.4 | 4.3 | 1.2 | 0.6 | 0.5 | 0.5 | 4.0 | 0.7 |
| coffee machine | 57 | 12.6 | 105.8 | 1.5 | 0.6 | 0.3 | 0.4 | 9.1 | 10.3 | 0.4 | 0.4 | 0.4 | 0.4 | 0.4 | 0.4 | 3.5 | 0.4 |
| credit card | 4 | 7.7 | 37.2 | 7.1 | 9.6 | 9.6 | 0.3 | 5.6 | 5.8 | 0.5 | 0.5 | 0.5 | 0.5 | 0.3 | 0.3 | 1.8 | 0.4 |
| cup | 16 | 29.4 | 20.5 | 6.0 | 5.7 | 3.4 | 0.3 | 6.4 | 5.7 | 0.9 | 0.6 | 0.6 | 0.9 | 0.3 | 0.3 | 2.3 | 0.5 |
| dish sponge | 77 | 4.6 | 93.5 | 5.7 | 8.4 | 7.2 | 0.4 | 4.0 | 7.3 | 0.9 | 0.9 | 0.9 | 0.9 | 0.4 | 0.4 | 2.5 | 0.8 |
| egg | 38 | 4.1 | 64.4 | 9.4 | 4.2 | 12.8 | 0.6 | 3.2 | 3.3 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.4 | 1.4 | 0.5 |
| fork | 20 | 114.8 | 36.0 | 10.6 | 10.6 | 11.9 | 5.4 | 15.0 | 10.6 | 10.5 | 5.2 | 4.5 | 5.2 | 3.7 | 3.7 | 6.7 | 8.3 |
| garbage can | 44 | 10.7 | 63.0 | 2.1 | 5.5 | 1.0 | 0.4 | 12.4 | 13.9 | 1.0 | 0.7 | 0.7 | 1.1 | 0.4 | 0.6 | 3.0 | 1.0 |
| house plant | 95 | 6.4 | 85.1 | 1.1 | 0.6 | 0.2 | 0.4 | 7.8 | 8.8 | 1.3 | 0.4 | 0.5 | 0.9 | 0.4 | 0.5 | 4.2 | 1.1 |
| kettle | 51 | 3.9 | 27.0 | 4.1 | 1.3 | 1.3 | 0.4 | 10.1 | 6.8 | 0.6 | 0.5 | 0.5 | 0.5 | 0.4 | 0.4 | 2.7 | 0.5 |
| knife | 9 | 14.8 | 15.6 | 10.5 | 9.6 | 9.6 | 5.0 | 11.6 | 9.6 | 4.3 | 4.3 | 1.9 | 4.3 | 5.0 | 9.9 | 8.4 | 2.6 |

Mean runtime per input (s, RTX 3090, shared GPU): P 0.45, raft 0.36, roma 1.21, depth 0.68, sam 0.88, rc 7.95, inerf 12.63, lg_pnp 0.08, roma_pnp 1.15, cotracker 0.53, icp 1.44, mast3r 1.65


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
| agree+dispnet_v3 | 0.9 | 20.45 | 0.657 | 0.182 | 0.703 | 19.69 | 0.630 | 0.173 | 0.696 | 31.69 |
| agree→dispnet_v3+rc | 1.3 | 19.85 | 0.638 | 0.191 | 0.620 | 19.36 | 0.609 | 0.188 | 0.640 | 31.53 |
| cotracker/dispnet_v3+rc | 0.9 | 20.14 | 0.645 | 0.186 | 0.673 | 19.87 | 0.648 | 0.172 | 0.687 | 31.63 |
| cotracker→dispnet_v3+rc | 1.0 | 19.88 | 0.636 | 0.189 | 0.641 | 19.65 | 0.627 | 0.179 | 0.670 | 31.57 |
| dispnet_v3 | 5.6 | 17.54 | 0.544 | 0.251 | 0.469 | 17.91 | 0.518 | 0.218 | 0.560 | 31.20 |
| dispnet_v3+rc | 1.4 | 19.78 | 0.632 | 0.191 | 0.605 | 19.26 | 0.604 | 0.190 | 0.613 | 31.53 |

## Rendering metrics — T2 (891 inputs rendered by every method)

The object's Gaussians are moved by the estimate and every view is rendered and compared with the real AFTER photo. ROI = dilated union of the true BEFORE/AFTER object masks; LPIPS on the ROI bounding box; IoU = rendered visible silhouette vs. true AFTER mask. 'novel' = mean over the other views that see the object. gt = true displacement (ceiling for this GG model + selection). ↑ higher is better, ↓ lower is better.

| method | err median cm | input PSNR-ROI ↑ | input SSIM-ROI ↑ | input LPIPS-ROI ↓ | input IoU ↑ | novel PSNR-ROI ↑ | novel SSIM-ROI ↑ | novel LPIPS-ROI ↓ | novel IoU ↑ | novel PSNR full ↑ |
|---|---|---|---|---|---|---|---|---|---|---|
| gt | 0.0 | 23.77 | 0.770 | 0.123 | 0.792 | 23.92 | 0.773 | 0.119 | 0.787 | 31.77 |
| no_update | 8.3 | 17.10 | 0.458 | 0.239 | 0.348 | 17.28 | 0.467 | 0.236 | 0.341 | 30.94 |
| P1 | 9.9 | 20.64 | 0.634 | 0.184 | 0.462 | 19.14 | 0.553 | 0.210 | 0.375 | 30.95 |
| inerf | 4.8 | 21.74 | 0.663 | 0.168 | 0.570 | 20.39 | 0.596 | 0.187 | 0.496 | 31.41 |
| lg_pnp | 3.7 | 20.69 | 0.632 | 0.179 | 0.546 | 20.10 | 0.595 | 0.186 | 0.513 | 31.41 |
| roma_pnp | 1.5 | 21.75 | 0.675 | 0.165 | 0.595 | 21.23 | 0.645 | 0.169 | 0.569 | 31.51 |
| cotracker | 0.3 | 22.97 | 0.727 | 0.133 | 0.731 | 22.92 | 0.719 | 0.131 | 0.724 | 31.68 |
| icp | 5.4 | 19.24 | 0.537 | 0.185 | 0.557 | 18.80 | 0.520 | 0.201 | 0.481 | 31.13 |
| mast3r | 8.8 | 17.17 | 0.460 | 0.240 | 0.351 | 17.31 | 0.468 | 0.238 | 0.340 | 30.91 |
| raft_crop | 0.4 | 22.82 | 0.717 | 0.133 | 0.726 | 22.76 | 0.710 | 0.132 | 0.716 | 31.68 |
| roma | 0.2 | 22.12 | 0.690 | 0.158 | 0.627 | 22.22 | 0.691 | 0.154 | 0.627 | 31.60 |
| depth | 6.9 | 18.38 | 0.497 | 0.230 | 0.411 | 18.58 | 0.508 | 0.222 | 0.402 | 31.05 |
| sam | 3.0 | 19.79 | 0.562 | 0.179 | 0.576 | 19.90 | 0.567 | 0.179 | 0.557 | 31.26 |
| rc | 0.5 | 22.76 | 0.703 | 0.137 | 0.711 | 22.38 | 0.685 | 0.139 | 0.691 | 31.59 |
| agree | 0.5 | 23.00 | 0.719 | 0.129 | 0.754 | 22.70 | 0.701 | 0.129 | 0.735 | 31.65 |
| agree+dispnet_v3 | 0.5 | 23.01 | 0.719 | 0.127 | 0.757 | 22.70 | 0.701 | 0.128 | 0.738 | 31.65 |
| agree→dispnet_v3+rc | 0.5 | 22.92 | 0.713 | 0.130 | 0.743 | 22.56 | 0.693 | 0.131 | 0.722 | 31.62 |
| cotracker/dispnet_v3+rc | 0.3 | 23.14 | 0.734 | 0.130 | 0.746 | 23.06 | 0.724 | 0.128 | 0.738 | 31.68 |
| cotracker→dispnet_v3+rc | 0.4 | 23.09 | 0.728 | 0.131 | 0.743 | 22.94 | 0.716 | 0.129 | 0.730 | 31.66 |
| dispnet_v3 | 2.6 | 20.57 | 0.591 | 0.166 | 0.616 | 20.67 | 0.592 | 0.160 | 0.617 | 31.35 |
| dispnet_v3+rc | 0.6 | 22.96 | 0.713 | 0.130 | 0.742 | 22.55 | 0.691 | 0.132 | 0.717 | 31.61 |