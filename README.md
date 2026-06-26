# GaussianUpdate

A system for updating 3D Gaussian Splatting scene representations from 2D observations. When an object moves in the real world, the pipeline detects the change from a single new RGB image, identifies the responsible Gaussians, estimates the 3D displacement, and updates the model — without retraining.

Built on top of [Gaussian Grouping (ECCV'24)](https://github.com/lkeab/gaussian-grouping) and evaluated on AI2THOR synthetic indoor scenes.

---

## What it does

1. **Train** a Gaussian Grouping model on a scene — learns a 3D splat with per-Gaussian object identity
2. **Detect** what changed between the trained scene and a new image — via pixel diff + template matching or SAM2 + CoTracker
3. **Update** — back-projects the 2D shift to 3D, moves the Gaussian cluster, re-renders

---

## Repository Structure

```
train.py                         GG training script
render.py                        Render trained model
metrics.py                       PSNR/SSIM evaluation
convert.py                       COLMAP format conversion

arguments/                       Training/model argument definitions
gaussian_renderer/               3DGS renderer (diff-gaussian-rasterization)
scene/                           Scene loading, COLMAP reader, GaussianModel
utils/                           Camera, graphics, loss, SH utilities
submodules/                      CUDA extensions (diff-gaussian-rasterization, simple-knn)

config/
  gaussian_dataset/train.json    Standard GG training config
  sim/train_30k.json             AI2THOR scene training config (32 classes, 30k iters)

script/
  train.sh                       Training wrapper: bash script/train.sh <scene> <gpu>

scripts/
  detect_and_update.py           Core inference: pixel-diff → template match → Gaussian update
  track_and_update.py            SAM2 + CoTracker variant (video input)
  cotracker_track.py             CoTracker3 subprocess runner (dev env)
  sam2_segment.py                SAM2 image segmentation helper (dev env)
  lama_inpaint.py                LaMa deep inpainting helper (dev env)
  run_batch_experiment.py        Automated 25-trial batch runner
  make_final_example.py          End-to-end demo using OpenCV inpainting
  make_cotracker_example.py      End-to-end demo using SAM2 + CoTracker
  make_lama_example.py           End-to-end demo using LaMa inpainting
  sim/
    collect_data.py              Collect multi-view images + COLMAP from AI2THOR
    before_after.py              Generate GT before/after pairs via object teleport
    colmap_writer.py             Write AI2THOR camera params to COLMAP binary format
    eval_experiment.py           Single-trial eval: detect + update + PSNR/SSIM
    collect_scene_and_trials.py  Automated full-scene data collection + 5 trials
    run_5_trials.sh              Run eval on 5 pre-captured trials
    run_batch_experiments.sh     25-trial batch runner (AI2THOR scenes)
    summarize_batch.py           Aggregate metrics → table / CSV / panel image

STATUS.md                        Full project status, methodology, results, future work
ALGORITHM.md                     Plain-English pipeline description
```

---

## Dataset

Three AI2THOR kitchen scenes collected and stored in `dataset/`:

| Scene | Training images | Trials |
|---|---|---|
| FloorPlan1 | 484 | apple, book, bowl, bread, coffeemachine |
| FloorPlan2 | 475 | bowl, bread, cellphone, coffeemachine, dishsponge |
| FloorPlan3 | 343 | apple, bowl, bread, coffeemachine, cup |

Each scene contains: RGB frames, per-frame object masks, COLMAP sparse model, and 5 before/after displacement trial pairs with ground-truth pixel shift in `meta.json`.

---

## Trained Models

| Scene | Iterations | Location |
|---|---|---|
| FloorPlan1 | 30,000 | `output/sim/FloorPlan1/point_cloud/iteration_30000/` |

---

## Eval Results — FloorPlan1 (iter 30000)

| Object | GT shift | Loc. Error | PSNR | SSIM |
|---|---|---|---|---|
| apple | (−48.9, +3.1) px | 49.0 px | 19.15 dB | 0.874 |
| book | (−73.4, +8.3) px | 122.9 px | 18.63 dB | 0.856 |
| bowl | (−16.5, +0.5) px | 16.5 px | 19.16 dB | 0.874 |
| bread | (−34.8, +1.7) px | 34.9 px | 19.07 dB | 0.872 |
| coffeemachine | (−22.6, +4.1) px | 178.8 px | 18.86 dB | 0.865 |
| **Mean** | | **80.4 px** | **18.97 dB** | **0.868** |

Render quality is consistent (~19 dB PSNR). Localisation errors on 3/5 objects stem from domain gap between the GG splat render and the AI2THOR ground-truth image. See `STATUS.md` for diagnosis and proposed fixes.

---

## Setup

Two conda environments are required:

**`gaussian_grouping`** — GG training and inference (Python 3.8)
```bash
conda create -n gaussian_grouping python=3.8
pip install torch==2.4.1+cu118 torchvision --index-url https://download.pytorch.org/whl/cu118
pip install plyfile wandb lpips scipy opencv-python scikit-learn
pip install --no-build-isolation submodules/diff-gaussian-rasterization
pip install --no-build-isolation submodules/simple-knn
```

**`dev`** — SAM2, CoTracker, AI2THOR data collection (Python 3.10)
```bash
conda create -n dev python=3.10
pip install torch torchvision
pip install ai2thor
pip install git+https://github.com/facebookresearch/sam2.git
pip install git+https://github.com/facebookresearch/co-tracker
pip install opencv-python-headless
```

Weights required:
- `weights/sam2/sam2.1_hiera_large.pt` (857 MB) — [SAM2 releases](https://github.com/facebookresearch/sam2/releases)
- `weights/cotracker/cotracker3_scaled_online.pth` (97 MB) — [CoTracker releases](https://github.com/facebookresearch/co-tracker/releases)

---

## Quick Start

**Collect data for a new scene:**
```bash
conda run -n dev python scripts/sim/collect_scene_and_trials.py \
    --scene FloorPlan1 --out dataset/
```

**Train:**
```bash
conda run -n gaussian_grouping python train.py \
    -s dataset/FloorPlan1/env \
    -m output/sim/FloorPlan1 \
    --config_file config/sim/train_30k.json
```

**Run 5-trial eval:**
```bash
bash scripts/sim/run_5_trials.sh 30000
```

**Run single inference:**
```bash
conda run -n gaussian_grouping python scripts/detect_and_update.py \
    --model_path output/sim/FloorPlan1 \
    --source_path dataset/FloorPlan1/env \
    --cam_idx 0 \
    --after_image_path /path/to/after.png \
    --output output/result/
```

---

## Acknowledgements

Built on [Gaussian Grouping](https://github.com/lkeab/gaussian-grouping) (Ye et al., ECCV 2024) and [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) (Kerbl et al., SIGGRAPH 2023).
