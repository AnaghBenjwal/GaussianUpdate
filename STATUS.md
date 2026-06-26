# Project Status — Gaussian Grouping Scene Update Pipeline

**Last updated:** 2026-06-26
**Branch:** `dev`  |  **Latest commit:** `1dfd29c` (day7: eval pipeline, batch runner, and sim fixes)

---

## Overview

This project builds a system for **3D scene updating from 2D observations**: given a trained Gaussian Grouping (GG) model of a static scene and a new RGB image showing an object in a different position, the system detects what changed, identifies the responsible Gaussians, estimates the 3D displacement, and updates the model. The evaluation harness uses AI2THOR to generate ground-truth before/after image pairs in synthetic indoor scenes, allowing quantitative measurement of displacement accuracy (pixel error) and render quality (PSNR/SSIM).

---

## Research Methodology — Theoretical Standing

### The Core Problem

3D Gaussian Splatting (3DGS) models a static scene as a set of 3D Gaussians with learned positions, covariances, opacities, and spherical harmonic (SH) colour coefficients. Gaussian Grouping (ECCV'24) extends this by jointly training a per-Gaussian object identity feature, enabling instance-level segmentation of the 3D representation. The fundamental challenge this project addresses is: **when the physical world changes (an object moves), how do we update the 3D model from a single 2D observation without retraining?**

This is a harder version of the novel-view synthesis problem. The Gaussians encode both geometry (position, shape) and appearance (SH coefficients) jointly. Moving Gaussians to a new position preserves geometry but leaves SH coefficients calibrated for the original viewing direction — an unsolved problem in the 3DGS editing literature, noted in GaussianEditor (CVPR'24), GaussianAvatar, and others. Our approach sidesteps full re-optimisation by assuming the scene contains only **rigid planar displacements** (object slides on a surface), which is appropriate for the tabletop/kitchen manipulation domain we target with AI2THOR.

### Pipeline Design

The system has two modes, both sharing the same final update step:

**Mode A — Pixel-diff + Template Matching (current evaluation mode, no neural weights at runtime):**
1. Render the scene at a known camera pose → "before" image
2. Compute per-pixel absolute difference between before and after; threshold and morphologically clean to find the changed region
3. Template-match the changed-region crop from "before" into "after" to recover the 2D centroid shift
4. Back-project the 2D shift to 3D world space using pinhole camera model and the mean depth of the object's Gaussians
5. Apply the 3D displacement to all Gaussians belonging to the detected class (identified via the GG per-Gaussian classifier)
6. Re-render

**Mode B — SAM2 + CoTracker (higher accuracy, requires video clip):**
Same as Mode A but replaces steps 2–3 with: SAM2 segments the object in the first frame of a multi-frame motion clip; CoTracker3 seeds a point grid inside the SAM2 mask and tracks all points forward; the median endpoint displacement becomes the 2D shift estimate. This is more robust to occlusion and lighting variation but requires a video rather than a snapshot pair.

### What Is Theoretically Sound

- **Gaussian identification via classifier:** Applying the GG per-Gaussian feature classifier directly to `gaussians._objects_dc` without rendering is exact and fast. The class boundaries are trained alongside the radiance field and are accurate to within the training distribution.
- **Depth estimation from Gaussian xyz:** Reading object depth from the mean z-coordinate of selected Gaussians in camera space is exact for a pinhole model and avoids depth-rendering complexity. It is valid as long as the object Gaussians are not scattered across very different depths (i.e., the object is compact).
- **Planar-motion back-projection:** Setting the camera-space z-component of the displacement to zero is the correct assumption for objects sliding on a flat surface. It ensures the projected pixel shift in the re-render matches the estimated 2D shift exactly, given accurate depth.
- **Template matching for rigid shifts:** NCC template matching (OpenCV `matchTemplate`) recovers exact sub-pixel shifts for synthetic rigid translations. In real scenes it degrades gracefully with texture richness.

### Known Limitations and Open Problems

**SH appearance artifact:** Moving Gaussians preserves geometry but not appearance. View-dependent colour encoded in SH degree ≥1 coefficients will be incorrect at the new position from novel viewpoints. The PSNR improvement after update is modest (~19 dB) precisely because the appearance is not re-optimised. Fix: zero-out SH degree ≥1 after move (degrade to diffuse), or run a short fine-tuning pass on the new position.

**Change detection fragility:** The pixel-diff approach is sensitive to lighting variation, JPEG compression, and rendering anti-aliasing differences. In the FloorPlan1 eval, 3/5 objects returned zero detected shift because the diff mask was not cleanly isolating the correct object — the threshold was either too high (missing the change) or the object's pixel footprint was too small relative to background variation. This is the primary reason for the high mean localization error (80.4 px).

**Single-view ambiguity:** A 2D shift in one image does not fully constrain the 3D displacement — the depth must be assumed known (taken from the Gaussian positions). For objects that moved in depth (toward/away from camera), the planar assumption breaks down.

**Domain gap:** All current training and evaluation is on AI2THOR synthetic renders. Real RGB images will have motion blur, lighting variation, lens distortion, and compression that the pixel-diff pipeline is not robust to. SAM2+CoTracker mode is designed to address this but has not yet been evaluated on real data.

---

## Day-by-Day Progress

### Day 1 — 2026-06-18 — Baselines (commit `485248c`)
- Gaussian Grouping baseline: rendered 299 frames from pretrained figurines checkpoint
- SAM2: video propagation on 50 GG frames, mask segmentation working
- CoTracker3: point tracking demo on same frames
- Git repo initialized, dev branch created, `.gitignore` committed

### Day 2 — 2026-06-19 — Core Inference Script (commit `cd65539`)
- `scripts/detect_and_update.py` implemented: full pixel-diff → template-match → 3D update pipeline
- Synthetic self-test: ±60px/±30px shift recovered exactly; 5,177 Gaussians updated
- `ALGORITHM.md` written
- Key design decisions: template matching over centroid-shift; depth from Gaussian xyz not depth render

### Day 3 — 2026-06-21 — Presentable OpenCV Example (commit `a8fe226`)
- `scripts/make_final_example.py`: full end-to-end demo using OpenCV inpainting
- GG class 42 (68,044 Gaussians), shift (−300, −100) px recovered exactly
- Documented SH appearance artifact as known limitation
- Output: comparison grid, before/after wipe video

### Day 4 — 2026-06-21 — SAM2 + CoTracker Pipeline (commit `4f9daf6`)
- `scripts/track_and_update.py`: SAM2 + CoTracker variant of detect_and_update
- `scripts/cotracker_track.py`: subprocess-based CoTracker runner in `dev` env
- `scripts/make_cotracker_example.py`: full demo using video input instead of snapshot pair
- Two-env architecture: `dev` (Python 3.10) for SAM2/CoTracker, `gaussian_grouping` (Python 3.8) for GG rendering; cross-env calls via subprocess

### Day 5 — 2026-06-21 — LaMa Inpainting + Batch Runner (commits `fc1ca2c`, `14b3c47`)
- `scripts/lama_inpaint.py`: LaMa deep inpainting as alternative to OpenCV TELEA for cleaner vacated-region fill
- `scripts/run_batch_experiment.py`: automated runner for 5 objects × 5 displacements × 2 methods (pixel-diff vs SAM2+CoTracker) = 25 experiment configurations

### Day 6 — 2026-06-23 — AI2THOR Simulator Integration (commit `570d1e7`)
- `scripts/sim/collect_data.py`: collect 300–500 multi-view RGB frames + COLMAP sparse model + object masks from any AI2THOR scene using CloudRendering (headless GPU)
- `scripts/sim/before_after.py`: generate ground-truth before/after pairs by teleporting objects and rendering from the canonical cam0 — eliminates inpainting entirely (no hallucination artifacts)
- `scripts/sim/colmap_writer.py`: writes COLMAP binary format (cameras.bin, images.bin, points3D.bin) from AI2THOR camera parameters + depth
- `scripts/sim/eval_experiment.py`: end-to-end eval for one experiment: calls detect_and_update, computes PSNR/SSIM vs GT after, generates result panel
- `scripts/sim/run_batch_experiments.sh`: 25-trial batch runner using AI2THOR before/after pairs
- `scripts/sim/summarize_batch.py`: aggregates metrics across all experiments, produces summary table, CSV, JSON, and tiled panel PNG
- **FloorPlan1 data collected:** 484 multi-view frames, COLMAP model, 5 trial objects

### Day 7 — 2026-06-25 — Remote Training + First Eval Results (commit `1dfd29c`)

#### Remote Training (vast.ai RTX 3090)
- Rented RTX 3090 instance on vast.ai; built `diff-gaussian-rasterization` (required `#include <cstdint>` fix for CUDA 13) and `simple-knn`
- Trained GG model on FloorPlan1: 30,000 iterations, ~18 minutes, loss converged from 1.20 → stable
- Model rsynced to local: `output/sim/FloorPlan1/point_cloud/iteration_30000/`

#### First Quantitative Eval — FloorPlan1, 5 Trials
Results stored in `results/FloorPlan1_iter30000/`:

| Object | GT shift (px) | Detected (px) | Loc. Error | PSNR | SSIM |
|---|---|---|---|---|---|
| apple | (−48.9, +3.1) | (0, 0) | 49.0 px | 19.15 dB | 0.874 |
| book | (−73.4, +8.3) | (−190, +47) | 122.9 px | 18.63 dB | 0.856 |
| bowl | (−16.5, +0.5) | (0, 0) | 16.5 px | 19.16 dB | 0.874 |
| bread | (−34.8, +1.7) | (0, 0) | 34.9 px | 19.07 dB | 0.872 |
| coffeemachine | (−22.6, +4.1) | (−38, −174) | 178.8 px | 18.86 dB | 0.865 |
| **MEAN** | | | **80.4 px** | **18.97 dB** | **0.868** |

**Diagnosis:** Render quality (PSNR ~19 dB, SSIM ~0.87) is consistent across all objects — the GG model is well-trained. The localization failure on apple/bowl/bread (detected 0,0) is a change detection issue: the pixel diff between the GG render and the AI2THOR ground truth after-image is dominated by render-vs-real domain discrepancy rather than the object displacement. The before-render (step 01) and the after-image come from different renderers (GG splat vs AI2THOR), so background pixels differ too, swamping the object-displacement signal.

#### Dataset Expansion
- `scripts/sim/collect_scene_and_trials.py` written: collects data + auto-selects 5 visible trial objects per scene
- **FloorPlan2 collected:** 475 images, 5 trials (bowl, bread, cellphone, coffeemachine, dishsponge)
- **FloorPlan3 collected:** 343 images, 5 trials (apple, bowl, bread, coffeemachine, cup)

**Total dataset: 3 scenes × ~430 avg frames = 1,302 training images, 15 displacement trials**

---

## Current State of the Codebase

### Key Scripts
```
scripts/
  detect_and_update.py         Core inference: pixel-diff → template match → GG update
  track_and_update.py          SAM2+CoTracker variant (video input)
  cotracker_track.py           CoTracker3 subprocess runner
  sam2_segment.py              SAM2 image-predictor helper
  lama_inpaint.py              LaMa deep inpainting runner
  make_final_example.py        OpenCV inpainting end-to-end demo
  make_cotracker_example.py    SAM2+CoTracker end-to-end demo
  make_lama_example.py         LaMa inpainting end-to-end demo
  run_batch_experiment.py      25-trial automated batch runner (lerf_mask scenes)
  sim/
    collect_data.py            AI2THOR multi-view data collector
    before_after.py            GT before/after pair generator
    colmap_writer.py           AI2THOR → COLMAP binary writer
    eval_experiment.py         Single-trial eval (PSNR/SSIM)
    run_batch_experiments.sh   25-trial batch runner (AI2THOR scenes)
    run_5_trials.sh            5-trial runner for FloorPlan1 (used for eval)
    collect_scene_and_trials.py  Automated new-scene data collection
    summarize_batch.py         Metrics aggregation → table/CSV/panel
```

### Trained Models
```
output/sim/FloorPlan1/point_cloud/iteration_30000/
  point_cloud.ply      Trained GG Gaussians (FloorPlan1, 30k iters)
  classifier.pth       Per-Gaussian object classifier weights
```

### Dataset
```
dataset/
  FloorPlan1/   484 images + COLMAP + 5 trials (apple, book, bowl, bread, coffeemachine)
  FloorPlan2/   475 images + COLMAP + 5 trials (bowl, bread, cellphone, coffeemachine, dishsponge)
  FloorPlan3/   343 images + COLMAP + 5 trials (apple, bowl, bread, coffeemachine, cup)
```

### Eval Results
```
results/FloorPlan1_iter30000/
  {apple,book,bowl,bread,coffeemachine}/   Per-trial: render stages, metrics.json, result_panel.png
  summary.txt / summary.csv / summary.json
  summary_panel.png                        5-panel tiled comparison image
```

---

## Proposed Future Work

### Priority 1 — Fix Change Detection (blocks all downstream evals)

The core issue in the current eval is that the pixel-diff between a GG splat render and an AI2THOR ground truth image contains too much background noise to isolate the object displacement reliably. Three candidate fixes:

**1a. Domain-adapted thresholding:** Instead of a global diff threshold, compute a per-pixel baseline variance across all training views (which the GG model already saw) and flag pixels that deviate beyond that variance. This separates "render-vs-real style difference" from "genuine scene change."

**1b. Use GG render as the after-image reference too:** Render the GG model at cam0 before and after running detect_and_update. Compare two GG renders instead of comparing GG vs AI2THOR. The before/after AI2THOR images are used only to drive the update (not as the reference for diff); this eliminates the domain gap in the diff step.

**1c. Mask-based detection using GG object IDs:** Instead of diff, explicitly render the GG object segmentation mask for the `force_obj_id` object and compare its centroid in the before-render to the object's location in the after-image using a class-conditional detector (e.g., contour matching on the known object colour from the mask).

### Priority 2 — Train and Evaluate FloorPlan2 and FloorPlan3

Data is collected. Steps:
- Run same vast.ai training pipeline for FloorPlan2 and FloorPlan3 (each ~20 min on RTX 3090)
- Run `run_5_trials.sh` for each scene
- Compare across scenes — do certain object types / sizes work better?
- Aggregate into a 3-scene summary table (15 trials total)

### Priority 3 — Appearance Fix after Gaussian Move

After moving Gaussians, SH degree ≥1 coefficients are stale. Two tractable approaches:
- **Degree-0 fallback:** Set SH coefficients to degree 0 only (mean colour) for moved Gaussians immediately after displacement. Degrades to Lambertian but is always correct. Fast, no retraining.
- **Local re-optimisation:** After applying the displacement, run 500–1000 training iterations with only the moved Gaussians' SH parameters as free variables, holding everything else fixed. Requires the original training images, which we have.

### Priority 4 — Real-Image Evaluation

All current evals use AI2THOR renders as ground truth. The next scientific question is whether the pipeline works on real RGB images. Proposed approach:
- Capture a physical tabletop scene with a phone (or use an existing dataset like ScanNet)
- Run COLMAP on the captures, train a GG model
- Physically move one object, capture a single new image, run the pipeline
- Report PSNR/SSIM vs the real after-photograph

### Priority 5 — Replace Pixel-Diff with Learned Change Detection

The pixel-diff step is the weakest link. A learned 2-class change detection model (e.g., fine-tuned ChangeFormer or a lightweight CNN trained on GG-render vs AI2THOR pairs) would be significantly more robust. Training data: we can generate unlimited synthetic before/after pairs using AI2THOR, giving unlimited supervised training signal for the change detection head.

### Priority 6 — Extend to Object Removal and Insertion

The current pipeline handles displacement only. The GG editing infrastructure already supports removal (zero out Gaussian opacity) and could support insertion (copy Gaussian cluster from another scene and paste at a target location). Combining all three into a unified scene-update API would complete the system.

---

## Environment Notes

| Conda Env | Python | PyTorch | Use |
|---|---|---|---|
| `gaussian_grouping` | 3.8 | 2.4.1+cu118 | GG training, rendering, eval |
| `dev` | 3.10 | 2.5.1+cu121 | SAM2, CoTracker, AI2THOR data collection |

Remote training: vast.ai RTX 3090, `/venv/main` (Python 3.12, PyTorch 2.12+cu130). CUDA 13 required `#include <cstdint>` patch to `diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h`.
