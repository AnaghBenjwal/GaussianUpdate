# Object displacement after a move — literature baselines vs. our methods (final)

Task: a GG model of a kitchen exists; one object is moved on its support surface. Given a real BEFORE and AFTER
photo from one camera (plus the GG model), estimate the object's 3D displacement δ. Identification is fixed to the
true class (2-image identification is 100 % on T1) so only the displacement stage is compared. Two kinds of score:

- **Displacement error** ‖δ̂ − δ‖ in cm, per (move, input view).
- **Rendering quality of the updated scene**: the object's Gaussians are moved by δ̂, every view is rendered and
  compared with the real AFTER photo — PSNR / SSIM / LPIPS in the object region and silhouette IoU against the true
  AFTER mask, on the input view and on the *novel* views (the other cameras that see the object).

## Test sets
| set | moves | inputs | scenes | |δ| | notes |
|---|---|---|---|---|---|
| T1 | 37 | 37 (frame_00000 only) | FloorPlan1, 3 | 15 / 30 cm, fixed directions | original benchmark; our hand-built rules were tuned on it |
| T2 | 67 | 368 (every view that sees the object) | FloorPlan1, 3 | random 3–40 cm, random direction (median 7.3) | generated after all design choices were frozen → the held-out test |

DispNet never saw FloorPlan1 or FloorPlan3 (training: 10,267 samples from FloorPlan6–30; model selection on
FloorPlan2/4/5).

## Methods compared
**Literature baselines** (adapted to the task — each turns its native output into a 3D object translation):

| name | method | adaptation |
|---|---|---|
| iNeRF | Yen-Chen et al., IROS 2021 | gradient descent on the object translation through the differentiable GG render, photometric loss in the object region |
| LightGlue+PnP | DISK + LightGlue (ICCV 2023) → PnP-RANSAC; = pose step of 3DGS-CD (RA-L 2025) | BEFORE-photo 3D points from GG depth on the object, matched to the AFTER photo |
| RoMa+PnP | RoMa (CVPR 2024) dense matches → PnP-RANSAC | as above |
| CoTracker3 | Karaev et al., 2024, offline point tracker | tracks object points through a BEFORE→AFTER clip, lifts them to the support plane, RANSAC |
| ICP | Besl & McKay 1992, point-to-point | BEFORE points from GG depth vs. AFTER points from a SAM mask + Depth Anything V2 depth (aligned to GG) |
| MASt3R | Leroy et al., ECCV 2024 | metric pointmaps of the pair (scaled to GG) + descriptor matches on the object, 3D differences + RANSAC |

Not run: **FoundationPose** (needs an object mesh and a depth image of the AFTER scene, plus nvdiffrast/kaolin builds
on CUDA 13 — not feasible in the time available).

**Ours**: P1/P2 (the existing pipelines), the hand-built cues, global render-and-compare (rc), the agree rule, and
the learned model **DispNet v2** (ResNet-18 on an 11-channel crop, crop-normalised output, Laplace NLL), alone and
with a local render-and-compare refinement (±8 cm around its prediction).

## Main result — T2 (held out), 95 % CIs from resampling moves

| method | median cm | mean cm | <2 cm | <10 cm | no answer | time / input |
|---|---|---|---|---|---|---|
| no update | 7.3 | 11.0 | 0 % | 60 % | – | – |
| P1 current pipeline | 6.6 | 48.4 | 15 % | 60 % | – | 0.4 s |
| *iNeRF* | 5.1 | 9.0 [7.1–11.2] | 30 % | 72 % [64–80] | 0 | 8.7 s |
| *LightGlue + PnP* | 3.5 | 7.0 [5.3–8.8] | 38 % | 73 % [64–82] | 151 | 0.05 s |
| *RoMa + PnP* | 1.5 | 7.3 [4.4–11.4] | 56 % | 79 % [69–87] | 36 | 0.75 s |
| *CoTracker3* | **0.3** | 1.9 [0.9–3.1] | 88 % | 92 % [86–98] | 33 | 0.37 s |
| *ICP (DA-V2 + SAM)* | 5.2 | 10.7 [8.1–14.4] | 11 % | 74 % [68–80] | 94 | 1.0 s |
| *MASt3R* | 7.7 | 11.4 [9.4–13.5] | 0 % | 58 % [44–70] | 351 | 1.0 s |
| global rc | 0.6 | 3.5 [2.1–5.1] | 74 % | 87 % [79–94] | 0 | 7.4 s |
| agree rule (4 cues + rc) | 0.5 | 1.7 [1.2–2.4] | 82 % | 96 % [93–99] | 0 | ≈ 10 s |
| DispNet v2 | 2.8 | 4.0 [3.3–4.8] | 38 % | 92 % [87–96] | 0 | 0.025 s |
| DispNet v2 + local rc | 0.6 | 2.2 [1.5–3.1] | 77 % | 94 % [89–98] | 0 | 2.3 s |
| **CoTracker3 → DispNet v2 + local rc (proposed)** | **0.3** | **1.4 [0.7–2.3]** | **90 %** | **95 % [91–99]** | 0 | **≈ 0.6 s** |

(*italics* = literature baselines.) Proposed = use the CoTracker3 estimate; when it gives no answer (9 % of T2
inputs), use DispNet v2 refined by local rc. Expected runtime 0.37 s + 0.09 × (0.025 + 2.3) s. Local rc was timed
on 58 T2 inputs: 2.3 s mean vs. 3.9 s for a global rc on the same inputs and GPU.

**Paired differences of the proposed pipeline (mean error, cm; negative = proposed better):**

| vs. | T2 | T1 |
|---|---|---|
| CoTracker3 alone (best literature baseline) | −0.47 [−0.95, −0.11] | −1.34 [−3.29, −0.05] |
| RoMa + PnP | −5.93 [−9.86, −3.13] | −5.92 [−9.68, −2.74] |
| global rc | −2.10 [−3.30, −1.17] | −4.01 [−7.43, −0.59] |
| DispNet v2 + local rc | −0.80 [−1.10, −0.55] | −1.99 [−3.55, −0.59] |
| agree rule | −0.31 [−0.78, +0.15] (tie) | +0.61 [−0.75, +2.62] (tie) |

## Rendering metrics — T2 (all 368 inputs)
| method | err median | input PSNR-ROI ↑ | input LPIPS-ROI ↓ | input IoU ↑ | novel PSNR-ROI ↑ | novel SSIM-ROI ↑ | novel LPIPS-ROI ↓ | novel IoU ↑ |
|---|---|---|---|---|---|---|---|---|
| gt δ (ceiling) | 0.0 | 22.89 | 0.124 | 0.813 | 23.00 | 0.769 | 0.121 | 0.815 |
| no update | 7.3 | 15.93 | 0.245 | 0.361 | 16.01 | 0.442 | 0.244 | 0.358 |
| P1 | 6.6 | 20.20 | 0.172 | 0.553 | 18.39 | 0.543 | 0.206 | 0.448 |
| *iNeRF* | 5.1 | 20.86 | 0.166 | 0.613 | 19.27 | 0.576 | 0.192 | 0.521 |
| *LightGlue + PnP* | 3.5 | 19.57 | 0.186 | 0.570 | 18.92 | 0.574 | 0.195 | 0.543 |
| *RoMa + PnP* | 1.5 | 21.07 | 0.163 | 0.652 | 20.36 | 0.639 | 0.169 | 0.626 |
| *CoTracker3* | 0.3 | 22.36 | 0.131 | 0.774 | 22.23 | 0.722 | 0.129 | 0.771 |
| *ICP* | 5.2 | 18.52 | 0.177 | 0.606 | 17.72 | 0.498 | 0.203 | 0.522 |
| *MASt3R* | 7.7 | 15.98 | 0.246 | 0.362 | 16.03 | 0.442 | 0.246 | 0.353 |
| global rc | 0.6 | 21.86 | 0.139 | 0.734 | 21.37 | 0.677 | 0.142 | 0.720 |
| agree | 0.5 | 22.16 | 0.129 | 0.784 | 21.73 | 0.694 | 0.132 | 0.769 |
| DispNet v2 | 2.8 | 19.47 | 0.171 | 0.639 | 19.55 | 0.574 | 0.166 | 0.642 |
| DispNet v2 + local rc | 0.6 | 22.06 | 0.131 | 0.772 | 21.59 | 0.684 | 0.135 | 0.752 |
| **proposed** | 0.3 | **22.46** | **0.129** | 0.782 | **22.30** | **0.725** | **0.128** | **0.778** |

The rendering metrics rank the methods the same way as the displacement error. The proposed pipeline is closest to
the gt-δ ceiling on the novel views (−0.7 dB PSNR-ROI, against −1.3 dB for agree and −3.4 dB for DispNet alone).
Rendering metrics are noisier than displacement error: the GG model and object selection cap PSNR-ROI at ≈ 23 dB,
and full-image PSNR barely moves (30.4 no update → 31.5 gt) because the object is a small part of the image. T1
tables are in `tables.md`.

## Findings
1. **CoTracker3 is the strongest literature baseline by far** (T2 median 0.3 cm, mean 1.9 cm). Dense tracking through
   a 2-frame "video" plus our support-plane lifting already matches the hand-built agree rule. Its failure mode is
   giving up (33 / 368 T2 inputs, 5 / 37 T1); 24 of the 33 T2 cases are the butter knife, and the moves it gives
   up on are large (median 18 cm vs. 7 cm overall).
2. **The other literature methods are far behind.** PnP methods depend on enough matches on the object: LightGlue gives
   no answer on 41 % of T2 inputs, RoMa+PnP is good when it works (median 1.5 cm) but has a long tail.
   iNeRF-style photometric descent gets stuck in local minima beyond ≈ 10 cm. ICP is limited by monocular depth.
   **MASt3R fails on this task**: its matches on the moved object stay at the object's old pixel position (both with
   reciprocal and with one-way matching), because the model assumes a static scene; it gives no answer on 95 % of
   T2 inputs.
3. **DispNet's value is robustness, not precision.** Alone it always answers and gets 92 % within 10 cm in 25 ms,
   but its median error is 2.8 cm. With local rc it reaches 0.6 cm median.
4. **Proposed final pipeline: CoTracker3, with DispNet v2 + local rc as the fallback when CoTracker3 gives no
   answer.** It is significantly better than every literature baseline and than DispNet + local rc on both test
   sets. It is statistically tied with the agree rule (T2 −0.3 cm, T1 +0.6 cm, both CIs include 0) while needing two
   networks instead of four external models plus a global search, and ≈ 0.6 s per input instead of ≈ 10 s.
   A stricter variant (also fall back when CoTracker3 and DispNet disagree by > 10 cm) is slightly worse (T2 mean
   1.6 cm), so the plain fallback is preferred.
5. **The existing pipelines (P1/P2) are unusable** for metric displacement (48–68 cm mean on T2).
6. **Remaining weakness: thin objects.** On the butter knife (25 T2 inputs) the proposed pipeline has a 13.4 cm
   median error (CoTracker3 gives up, and local rc pulls DispNet's 7.9 cm estimate the wrong way), while agree gets
   4.3 cm and DispNet alone 7.9 cm. On the transparent glass bottle the proposed pipeline is best (0.9 cm vs. 3.2 cm
   for agree). For thin objects, using DispNet without the rc refinement would be the better fallback; we did not
   change the rule after seeing this.

## Caveats
- **The proposed combination was chosen after seeing the T2 baseline results.** Its only parameter (the fallback
  condition "CoTracker3 gives no answer") was not tuned. It is also the best non-agree option on T1 and is
  significantly better than its components on both sets, so the choice is not a T2 artefact, but a fresh test set
  would be the clean confirmation.
- **T1 is small (37 inputs) and was used to design the hand rules**, so its numbers are optimistic for those methods.
- **T2 views are correlated within a move**, so all CIs resample moves rather than views.
- **All data is synthetic** (AI2-THOR renders, GG models trained on them). Real-camera performance is untested.
- **The literature methods are adaptations**, not their original systems; e.g. CoTracker3 and RoMa use our
  support-plane lifting, and the PnP methods use GG depth. They are the strongest adaptations we could build in the
  time available, and P1/P2 use the true class and real BEFORE photo.
- **Runtimes were measured on a shared RTX 3090** and are indicative only.

## Files
- `tables.md`: full tables (all methods, by move size, by object, rc-failure subset, rendering metrics for T1 and T2).
- `per_input.jsonl`: per-input errors for every method.
- `eval/T1|T2/<scene>/`: raw per-input estimates (`eval_cues`, `eval_baselines`, `eval_learned_*`), rendering
  metrics (`eval_render`), local-rc timing (`timing_local_rc`).
- `runs/r18` (v1) and `runs/r18crop` (v2 = final DispNet): `best.pt` checkpoint, training log, predictions.

Code is in `gaussian-grouping/scripts/learn/`: `train.py`, `predict.py`, `eval_methods.py` (our methods),
`eval_baselines.py` (literature baselines), `eval_render.py` (rendering metrics), `time_local_rc.py`,
`summarize_eval.py`, `bootstrap_ci.py`.
