# Object displacement after a move — literature baselines vs. our methods (final, v3)

Task: a GG model of a kitchen exists; one object is moved on its support surface. Given a real BEFORE and AFTER
photo from one camera (plus the GG model), estimate the object's 3D displacement δ. Identification is fixed to the
true class (2-image identification is 100 % on T1) so only the displacement stage is compared. Two kinds of score:

- **Displacement error** ‖δ̂ − δ‖ in cm, per (move, input view).
- **Rendering quality of the updated scene**: the object's Gaussians are moved by δ̂, every view is rendered and
  compared with the real AFTER photo — PSNR / SSIM / LPIPS in the object region and silhouette IoU against the true
  AFTER mask, on the input view and on the *novel* views (the other cameras that see the object).

**What changed since v2 (2026-10-18 report):** the test set grew from 2 to 5 kitchens (GG models trained for
FloorPlan2, 4, 5), the data split was redone (validation = FloorPlan10–15), and DispNet was retrained with two
changes chosen on validation (v3). All methods were re-run on the new test scenes.

## Data split
| split | scenes | samples / inputs | used for |
|---|---|---|---|
| train | FloorPlan6, 7, 9, 16–30 (18 kitchens) | 7,735 simulator samples | DispNet training |
| validation | FloorPlan10–15 (6 kitchens) | 2,532 simulator samples | choosing the DispNet variant and TTA |
| test T1 | FloorPlan1, 3 | 37 moves, 37 inputs (frame_00000) | original benchmark; the hand-built rules were tuned on it |
| test T2 | FloorPlan1–5 | 172 random moves, 891 inputs (every view that sees the object) | held-out test |

FloorPlan8 is empty. FloorPlan6–10 are reserved as a possible later test set. They are in train and validation
now, so they cannot be used as test data for this model. The T2 moves on FloorPlan2/4/5 (105 moves, 523 inputs) were
generated after every design choice was frozen, including the proposed pipeline. They are the clean confirmation
the v2 report asked for.

## DispNet v3 — variants compared on validation (FloorPlan10–15)
| variant | change | val mean cm | val median cm | <10 cm |
|---|---|---|---|---|
| base | v2 architecture (ResNet-18), retrained on the new split | 2.78 | 1.61 | 96 % |
| planar | predict only the floor-plane move (dy = 0) | 1.94 | 1.07 | 98 % |
| r34 | ResNet-34 backbone | 2.30 | 1.33 | 97 % |
| res384 | 384 px input | – | – | – |
| planar + r34 | both | 1.75 | – | – |
| **planar + r34 + flip TTA = v3** | average with the horizontally flipped input | **1.67** | **0.89** | **98.7 %** |

res384 ran out of the container's 45.8 GB RAM limit during data preloading and was not retried. On the 368 T2
inputs of FloorPlan1/3, v3 alone improves over v2 (median 2.8 → 2.0 cm, mean 4.0 → 3.7 cm). It does so despite six fewer
training scenes. Inside the proposed pipeline the difference is negligible (mean 1.40 vs. 1.41 cm), because the
model is only used when CoTracker3 gives no answer. Network time: 7.9 ms for both TTA passes on an RTX 3090; with
input building ≈ 0.03 s.

## Main result — T2 (FloorPlan1–5, 891 inputs, 172 moves), 95 % CIs from resampling moves

| method | median cm | mean cm | <2 cm | <10 cm | no answer | time / input |
|---|---|---|---|---|---|---|
| no update | 8.3 | 11.2 | 0 % | 59 % | – | – |
| P1 current pipeline | 9.9 | 55.0 | 11 % | 51 % | – | 0.45 s |
| *iNeRF* | 4.8 | 9.3 [7.8–10.9] | 29 % | 71 % [65–76] | 0 | 12.6 s |
| *LightGlue + PnP* | 3.7 | 7.7 | 38 % | 71 % | 318 | 0.08 s |
| *RoMa + PnP* | 1.5 | 7.2 [5.6–9.1] | 53 % | 75 % [69–81] | 108 | 1.15 s |
| *CoTracker3* | **0.3** | 2.2 [1.6–3.0] | 86 % | 93 % [89–96] | 75 | 0.53 s |
| *ICP (DA-V2 + SAM)* | 5.4 | 14.1 | 11 % | 71 % | 208 | 1.44 s |
| *MASt3R* | 8.8 | 11.6 | 1 % | 57 % | 826 | 1.65 s |
| global rc | 0.5 | 3.8 [2.8–4.8] | 72 % | 88 % [83–92] | 0 | 7.9 s |
| agree rule (4 cues + rc) | 0.5 | 1.8 [1.4–2.3] | 80 % | 96 % [94–98] | 0 | ≈ 10 s |
| agree + DispNet v3 (5 cues + rc) | 0.5 | 1.6 [1.3–2.0] | 81 % | 97 % [96–98] | 0 | ≈ 10 s |
| DispNet v3 | 2.6 | 4.1 [3.6–4.7] | 41 % | 90 % [87–93] | 0 | 0.03 s |
| DispNet v3 + local rc | 0.6 | 2.3 [1.9–2.9] | 75 % | 94 % [91–96] | 0 | 2.3 s |
| **CoTracker3 → DispNet v3 + local rc (proposed)** | **0.3** | **1.6 [1.2–2.1]** | **88 %** | **95 % [93–97]** | 0 | **≈ 0.7 s** |

(*italics* = literature baselines.) Proposed = use the CoTracker3 estimate; when it gives no answer (8.4 % of T2
inputs), use DispNet v3 refined by local rc. Expected runtime 0.53 s + 0.084 × (0.03 + 2.3) s. Runtimes were measured on
this run's GPU, which had only 1.9 CPU cores, so the CPU-bound steps are slower than in the v2 report.

**Fresh scenes only (FloorPlan2/4/5, 523 inputs, 105 moves; nothing was ever chosen on them):** proposed median
0.37 cm, mean 1.77 cm [1.26–2.42], 95 % < 10 cm. CoTracker3 alone: 0.37 / 2.46 cm. agree: 0.47 / 1.88 cm. The
ranking is the same as on all of T2. Full numbers are in `bootstrap.md`.

**Paired differences of the proposed pipeline (mean error, cm; negative = proposed better):**

| vs. | T2 all (FP1–5) | T2 fresh (FP2/4/5) | T1 |
|---|---|---|---|
| CoTracker3 alone (best literature baseline) | −0.59 [−1.04, −0.27] | −0.68 [−1.36, −0.20] | −1.54 [−3.53, −0.00] |
| RoMa + PnP | −5.60 [−7.36, −4.14] | −5.36 [−6.88, −3.99] | −6.13 [−9.89, −3.00] |
| global rc | −2.13 [−3.01, −1.38] | −2.15 [−3.38, −1.13] | −4.21 [−7.62, −0.75] |
| DispNet v3 + local rc | −0.71 [−1.09, −0.37] | −0.71 [−1.31, −0.19] | −1.71 [−3.23, −0.41] |
| agree rule | −0.19 [−0.48, +0.11] (tie) | −0.11 [−0.49, +0.31] (tie) | +0.40 [−1.22, +2.48] (tie) |
| agree + DispNet v3 | +0.03 [−0.26, +0.34] (tie) | +0.16 [−0.26, +0.59] (tie) | +0.33 [−1.27, +2.44] (tie) |

## Rendering metrics — T2 (all 891 inputs)
| method | err median | input PSNR-ROI ↑ | input LPIPS-ROI ↓ | input IoU ↑ | novel PSNR-ROI ↑ | novel SSIM-ROI ↑ | novel LPIPS-ROI ↓ | novel IoU ↑ |
|---|---|---|---|---|---|---|---|---|
| gt δ (ceiling) | 0.0 | 23.77 | 0.123 | 0.792 | 23.92 | 0.773 | 0.119 | 0.787 |
| no update | 8.3 | 17.10 | 0.239 | 0.348 | 17.28 | 0.467 | 0.236 | 0.341 |
| P1 | 9.9 | 20.64 | 0.184 | 0.462 | 19.14 | 0.553 | 0.210 | 0.375 |
| *iNeRF* | 4.8 | 21.74 | 0.168 | 0.570 | 20.39 | 0.596 | 0.187 | 0.496 |
| *LightGlue + PnP* | 3.7 | 20.69 | 0.179 | 0.546 | 20.10 | 0.595 | 0.186 | 0.513 |
| *RoMa + PnP* | 1.5 | 21.75 | 0.165 | 0.595 | 21.23 | 0.645 | 0.169 | 0.569 |
| *CoTracker3* | 0.3 | 22.97 | 0.133 | 0.731 | 22.92 | 0.719 | 0.131 | 0.724 |
| *ICP* | 5.4 | 19.24 | 0.185 | 0.557 | 18.80 | 0.520 | 0.201 | 0.481 |
| *MASt3R* | 8.8 | 17.17 | 0.240 | 0.351 | 17.31 | 0.468 | 0.238 | 0.340 |
| global rc | 0.5 | 22.76 | 0.137 | 0.711 | 22.38 | 0.685 | 0.139 | 0.691 |
| agree | 0.5 | 23.00 | 0.129 | 0.754 | 22.70 | 0.701 | 0.129 | 0.735 |
| agree + DispNet v3 | 0.5 | 23.01 | **0.127** | **0.757** | 22.70 | 0.701 | **0.128** | 0.738 |
| DispNet v3 | 2.6 | 20.57 | 0.166 | 0.616 | 20.67 | 0.592 | 0.160 | 0.617 |
| DispNet v3 + local rc | 0.6 | 22.96 | 0.130 | 0.742 | 22.55 | 0.691 | 0.132 | 0.717 |
| **proposed** | 0.3 | **23.14** | 0.130 | 0.746 | **23.06** | **0.724** | **0.128** | **0.738** |

The rendering metrics rank the methods the same way as the displacement error. The proposed pipeline is closest to the gt-δ
ceiling on novel-view PSNR-ROI: −0.9 dB, against −1.2 dB for agree and −3.3 dB for DispNet alone. The proposed
pipeline and agree + DispNet v3 are within noise of each other on every metric. T1 tables are in `tables.md`.

## Findings
1. **The v2 conclusions hold on 2.5× more test data and on three unseen kitchens.** CoTracker3 is the strongest
   literature baseline (T2 median 0.3 cm, mean 2.2 cm). The proposed pipeline (CoTracker3, DispNet v3 + local rc
   when it gives no answer) is significantly better than every literature baseline and every single component, on
   all of T2, on the fresh scenes alone and on T1.
2. **The proposed pipeline ties the agree rule** in accuracy, both with and without DispNet as a fifth cue. It needs
   ≈ 0.7 s per input instead of ≈ 10 s, and two networks instead of four external models plus a global search.
   On the subset where global rc fails (107 T2 inputs) the proposed pipeline has the lowest median (1.2 cm), but
   agree + DispNet v3 has the lowest mean (5.3 vs. 6.4 cm).
3. **DispNet v3 is a better model than v2** (planar output + ResNet-34 + flip TTA; −27 % validation error from planar
   alone). Its value in the pipeline is robustness: it always answers, 90 % within 10 cm on T2. As a fifth cue it
   gives a small, significant gain for the agree rule (−0.22 cm [−0.43, −0.05]).
4. **The other literature methods are far behind.** The PnP methods depend on enough matches on the object, and
   iNeRF gets stuck beyond ≈ 10 cm. **MASt3R fails on this task** (its matches on the moved object stay at the old pixel
   position; no answer on 93 % of T2).
5. **The existing pipelines (P1/P2) are unusable** for metric displacement (55–101 cm mean on T2).
6. **Remaining weakness: thin objects.** The butter knife has a median error of 13.0 cm on FloorPlan1/3 (25 inputs) and
   3.1 cm on FloorPlan2/4/5 (22 inputs). The fork has 3.7 cm and the knife 5.0 cm. CoTracker3 often gives up on
   them, and local rc can pull DispNet the wrong way.

## Caveats
- **T2 views are correlated within a move**, so all CIs resample moves rather than views.
- **T1 is small (37 inputs) and was used to design the hand rules**, so its numbers are optimistic for those methods.
- **All data is synthetic** (AI2-THOR renders, GG models trained on them). Real-camera performance is untested.
- **The literature methods are adaptations**, not their original systems: CoTracker3 and RoMa use our support-plane
  lifting, and the PnP methods use GG depth.
- **Runtimes are indicative only** (shared RTX 3090, 1.9 CPU cores).
- The FloorPlan1/3 baseline and cue results are the v2-run results. They use the same GG models, test moves and code, so
  they were not recomputed. Only the learned rows and their rendering rows are new on FloorPlan1/3.

## Files
- `tables.md`: full tables (all methods, by move size, by object, rc-failure subset, rendering metrics for T1 and T2).
- `bootstrap.md`: 95 % CIs and paired differences for T1, T2 and the fresh-scene subset.
- `per_input.jsonl`: per-input errors for every method.
- DispNet v3: `~/Documents/displacement_learn/runs/v3/` (`best.pt`, `args.json`, training log, predictions; `SOURCE` =
  runs2/planar_r34 + TTA). All variant runs: `~/Documents/displacement_learn/runs2/`.
- GG models: `~/Documents/vast_results/server_root/output/gg_FP{1..5}_fixed`.
- Raw per-input estimates: `~/Documents/displacement_bench_t2/FloorPlan{1..5}/eval_*.jsonl` (T2) and
  `~/Documents/displacement_bench_vast_v3/FloorPlan{1,3}/` (T1).

Code is in `gaussian-grouping/scripts/learn/`: `train.py` (`--arch`, `--planar`, `--res`, `--preload`), `predict.py`
(`--tta`), `eval_methods.py`, `eval_baselines.py`, `eval_render.py`, `summarize_eval.py`, `bootstrap_ci.py`.
