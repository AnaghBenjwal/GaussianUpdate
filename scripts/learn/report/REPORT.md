# Object displacement after a move — all methods vs. the learned model (DispNet)

Task: a GG model of a kitchen exists; one object is moved on its support surface. Given a real BEFORE and AFTER
photo from one camera (plus the GG model), estimate the object's 3D displacement δ. Error = ‖δ̂ − δ‖ in cm, scored
per (move, input view). Identification is fixed to the true class (2-image identification is 100 % on T1) so only
the displacement stage is compared.

## Test sets
| set | moves | inputs | scenes | |δ| | notes |
|---|---|---|---|---|---|
| T1 | 37 | 37 (frame_00000 only) | FloorPlan1, 3 | 15 / 30 cm, fixed directions | original benchmark; the hand-built rules were tuned on it |
| T2 | 67 | 368 (every view that sees the object) | FloorPlan1, 3 | random 3–40 cm, random direction (median 7.3) | generated after all design choices were frozen → the real held-out test |

DispNet never saw FloorPlan1 or FloorPlan3 (training: 10,267 samples from FloorPlan6–30; model selection on
FloorPlan2/4/5). At test time it gets GG-rendered depth and the GG object mask, which is what the real pipeline has.

## Main result (T2, held out) — 95 % CIs from resampling moves

| method | median cm | mean cm | <2 cm | <10 cm | no answer | time / input |
|---|---|---|---|---|---|---|
| no update | 7.3 | 11.0 | 0 % | 60 % | – | – |
| P1 current pipeline (template + constant depth) | 6.6 | 48.4 [38–61] | 15 % | 60 % | – | 0.4 s |
| P2 current pipeline (RAFT + constant depth) | 52.4 | 68.2 [57–81] | 1 % | 8 % | – | 0.4 s |
| Depth Anything V2 cue | 6.7 | 10.8 | 6 % | 64 % | 46 | 0.6 s |
| SAM contact cue | 2.9 | 8.1 | 41 % | 86 % | 97 | 0.7 s |
| RoMa crop + plane lifting | 0.2 | 4.5 | 74 % | 82 % | 90 | 1.0 s |
| RAFT crop + plane lifting | 0.4 | 2.1 | 85 % | 92 % | 29 | 0.3 s |
| global render-and-compare (rc) | 0.6 | 3.5 [2.1–5.1] | 74 % | 87 % [79–94] | 0 | 7.4 s |
| **agree rule** (cue consensus overrides rc) | **0.5** | **1.7 [1.1–2.4]** | **82 %** | **96 % [93–99]** | 0 | ≈ 10 s (all cues + rc) |
| **DispNet v2** (ours, alone) | 2.8 [2.2–3.6] | 4.0 [3.3–4.8] | 38 % | 92 % [87–96] | 0 | **0.025 s** |
| DispNet v2 + local rc | 0.6 | 2.2 [1.5–3.1] | 77 % | 94 % | 0 | 0.025 s + one local rc (not timed) |
| agree + DispNet v2 as 5th cue | 0.5 | 1.6 [1.1–2.3] | 82 % | 97 % [94–99] | 0 | ≈ 10 s |

(DispNet v1 = same network predicting metres directly: median 3.6, mean 4.9, 90 % <10 cm.)

## Findings
1. **The current pipelines (P1/P2) are unusable** for metric displacement: 48–68 cm mean error, because
   constant-depth back-projection of a noisy 2D shift is wrong whenever the object moves in depth. Even with the
   true class and the real BEFORE photo they are worse than not updating at all.
2. **The ablation winner holds up on held-out data.** The agree rule (chosen on T1) gives 0.5 cm median and
   1.7 cm mean on T2, 96 % under 10 cm. The worry that the rule was overfitted to T1 is not borne out.
3. **DispNet alone is fast and robust but not precise.** It always answers (no "no answer" cases), gets 92 %
   within 10 cm, and needs 25 ms (22 ms input building on CPU + 2.4 ms network) against ≈10 s for the full hand
   pipeline (~400× faster). Its median error (2.8 cm) is about 5× worse than the render-and-compare methods. Its
   mean error is statistically tied with global rc (paired difference +0.5 cm, CI [−0.8, +1.7]), with a very
   different error profile: rc is near-perfect or fails completely (≈15–20 cm, i.e. "no motion"), while DispNet
   is never exact but rarely far off.
4. **On the cases where rc fails** (rc error > 10 cm, 47 T2 inputs), DispNet alone gets 74 % within 10 cm and
   agree gets 72 %. Agree is much more precise there (median 2.4 vs 6.2 cm).
5. **Combinations.**
   - DispNet + local rc recovers sub-centimetre precision (median 0.6 cm, mean 2.2 cm) without any of the
     external models (RAFT, RoMa, Depth Anything, SAM).
   - Adding DispNet as a fifth cue to the agree rule changes the T2 mean by −0.07 cm (CI [−0.19, +0.03]), which is
     not significant.
6. **The crop-normalised output (v2) is a real improvement over v1.** Mean −0.86 cm, CI [−1.31, −0.41]. It removes
   most of v1's shrinkage of large moves toward the mean (v1 predicted 0.68 of the true length for 20–45 cm moves).
7. **The simulation-to-GG input gap is small.** DispNet v2 on simulator depth and mask: 2.4 cm median; on GG depth
   and mask: 2.8 cm. Depth and mask augmentation are enough.
8. **Hard cases for every method:** the butter knife (thin; agree 4.3 cm, DispNet 7.9 cm, rc 18 cm) and the glass
   bottle (transparent; ≈3–5 cm for all).

## Recommendation
- **Best accuracy:** use the agree rule (optionally with DispNet as a fifth cue; the difference is not
  significant).
- **Real-time or low compute:** use DispNet + local render-and-compare, which gives nearly the same accuracy
  (0.6 cm median, 94 % <10 cm) with one network and a single local search instead of four external models
  plus global rc (runtime of the local search not measured yet). DispNet alone (25 ms, 2.8 cm median) is enough when
  centimetre-level error is acceptable, e.g. as an initialiser or for tracking.

## Caveats
- **T1 is small and was used to design the rules**, so its numbers are optimistic for the hand methods. On T1,
  agree is 0.8 / 2.5 cm and DispNet v2 is 6.8 / 8.0 cm; T1 moves are larger (median 15 cm).
- **T2 views are correlated within a move**, so all CIs resample moves rather than views.
- **The data is all synthetic** (AI2-THOR renders, GG models trained on them). Real-camera performance is untested.
- **P1/P2 here use the true class and the real BEFORE photo.** This is more favourable than the original pipeline,
  which also misidentifies the object.
- **Runtimes were measured on a shared RTX 3090.** They are indicative only.

## Files
- `tables.md`: full tables (all methods, broken down by move size, by object, and on the rc-failure subset).
- `per_input.jsonl`: per-input errors for every method.
- `eval/`: raw per-input estimates.
- `runs/r18` (v1) and `runs/r18crop` (v2): `best.pt` checkpoint, training log, predictions.

Code is in `gaussian-grouping/scripts/learn/`: `train.py`, `predict.py`, `eval_methods.py`, `summarize_eval.py`,
`bootstrap_ci.py`.
