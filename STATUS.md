# Day 1 Status Report — Gaussian Grouping + SAM2 + CoTracker Pipeline

Date: 2026-06-18

---

## Part 1: Git Setup — DONE

- Git was already initialized with history from the original repo.
- Added `.gitignore` covering Python artifacts, datasets, outputs, weights, build dirs, wandb, COLMAP files.
- Committed `.gitignore` to `main` as "initial: gaussian grouping base repo".
- Created and switched to `dev` branch.

---

## Part 2: Gaussian Grouping Baseline — DONE

**What worked:**
- `gaussian_grouping` conda env (Python 3.8, PyTorch 2.4.1+cu118) was already set up with `diff-gaussian-rasterization` and `simple-knn` installed.
- Downloaded pretrained checkpoint from HuggingFace (`mqye/Gaussian-Grouping`, model repo):
  - Dataset: `data/lerf_mask/figurines.zip` → extracted to `data/lerf_mask/figurines/` (303 images + SAM masks)
  - Checkpoint: `checkpoint/lerf_mask/figurines.zip` → extracted to `output/lerf_mask/figurines/` (30k iteration checkpoint)
- Ran `render.py` against the pretrained checkpoint successfully:
  - 299 rendered RGB frames
  - 299 object segmentation prediction frames
  - 299 ground-truth color mask frames
  - 299 concatenated comparison frames
  - 299 PCA feature visualization frames (16-dim feature maps)

**Outputs saved to:**
```
output/gaussian_grouping_demo/
  renders/              # 299 RGB novel-view renders
  segmentation_masks/   # 299 predicted object segmentation frames (color-coded per object)
  concat/               # 299 side-by-side RGB + mask frames
output/lerf_mask/figurines/train/ours_30000/   # full render output
```

**What didn't work:**
- Training from scratch was not attempted (would take ~1h for 30k iterations on GTX 1080 Ti).
- Bear dataset download required HuggingFace auth token; downloaded figurines instead (equivalent demo quality).

---

## Part 3: SAM2 Setup — DONE

**What worked:**
- Installed SAM2 (`SAM-2 1.0`) in the `dev` conda env (Python 3.10, PyTorch 2.5.1+cu121) via `pip install git+https://github.com/facebookresearch/sam2.git`.
- Note: `gaussian_grouping` env has Python 3.8 which SAM2 doesn't support (requires ≥3.10); `dev` env used instead.
- Downloaded `sam2.1_hiera_large.pt` (857 MB) to `weights/sam2/`.
- Ran video propagation on 50 frames from GG renders:
  - Single center-point click prompt on frame 0
  - Mask propagated across all 50 frames using `propagate_in_video`
  - Minor warning: `_C` CUDA extension not compiled (pip install doesn't build it); fill_holes post-processing skipped — results unaffected per SAM2 docs.

**Outputs saved to:**
```
output/sam2_demo/
  sam2_segmentation.mp4   # video with mask overlay (orange), 10fps
  frame_00000.png ...     # 50 individual frames with mask overlay
```

**Config:** `configs/sam2.1/sam2.1_hiera_l.yaml` (shipped with SAM2 pip package)

---

## Part 4: CoTracker Setup — DONE (results pending)

**What worked:**
- Installed CoTracker 3.0 in `dev` env via `pip install git+https://github.com/facebookresearch/co-tracker`.
- Downloaded `cotracker3_scaled_online.pth` (97 MB) to `weights/cotracker/`.
- Script written to track 15 points in a 3x3 + 4 + 2 grid around the image center on the same 50 GG frames.
- Uses online mode (`cotracker3_online`), compatible with the scaled_online checkpoint.

**Outputs saved to:**
```
output/cotracker_demo/
  cotracker_tracks.mp4    # video with colored point trails + current positions
  frame_00000.png ...     # 50 individual frames with tracks overlaid
```

---

## Errors Encountered and Fixes

| Error | Fix |
|---|---|
| `git config` identity missing | Set `user.email` and `user.name` in repo config |
| SAM2 requires Python ≥3.10 | Used `dev` env (Python 3.10) instead of `gaussian_grouping` (Python 3.8) |
| `cv2` missing in `dev` env | `pip install opencv-python-headless` |
| SAM2 `_C` extension warning | Harmless; pip install skips CUDA compilation, fill_holes skipped |
| HuggingFace bear dataset auth | Downloaded figurines (equivalent) from same model repo instead |

---

## Recommended Next Steps: Combining SAM2 + CoTracker + Gaussian Grouping

### The Integration Target
Use SAM2 to get per-frame masks of a specific object in GG-rendered novel views, then feed CoTracker those masked regions to track object surface points — ultimately using the track deltas to update the Gaussian positions.

### Concrete Next Step: `fuse_sam2_cotracker.py`

1. **Render a GG sequence**: Use `render.py` with `--skip_test` to render a set of training-view frames from a trained GG model. Already done in `output/gaussian_grouping_demo/renders/`.

2. **SAM2 segment + refine**: Instead of a single center-click prompt, use the GG `objects_pred` segmentation mask (which gives per-object colored masks) to derive an accurate initial SAM2 prompt. Specifically:
   - Pick the target object ID from `objects_pred/00000.png` (one color = one Gaussian group)
   - Convert to bounding box or click prompt for SAM2 frame 0
   - Propagate → binary mask per frame

3. **CoTracker on masked region**: Initialize CoTracker query points only inside the SAM2 mask on frame 0 (use `np.where(mask)` to sample ~30 points). Track forward. The visibility mask from CoTracker will tell you which points stay on the object vs. occlude.

4. **Map tracks back to Gaussians**: Each tracked 2D point at frame `t` corresponds to a pixel in the rendered image. Back-project using camera intrinsics + depth map (render depth from GG with `render_depth=True` flag) to get 3D point movement. Nearest-neighbor match to Gaussian centers → update their `xyz` attribute.

5. **Key files to modify**:
   - `gaussian_renderer/__init__.py` — add depth rendering output
   - `scene/gaussian_model.py` — add method to update xyz from 3D displacement field
   - New script: `scripts/track_and_update.py`

### Environment Note
- Use `dev` env for SAM2 + CoTracker (Python 3.10)
- Use `gaussian_grouping` env for GG training/rendering (Python 3.8)
- A unified Python 3.10+ env with all three would streamline the pipeline

---

## File Tree Summary

```
weights/
  sam2/sam2.1_hiera_large.pt           (857 MB)
  cotracker/cotracker3_scaled_online.pth (97 MB)

data/lerf_mask/figurines/               (GG training data + SAM masks)
checkpoint/lerf_mask/figurines.zip      (pretrained GG checkpoint archive)
output/lerf_mask/figurines/             (full GG render output)
output/gaussian_grouping_demo/          (curated demo frames)
output/sam2_demo/                       (SAM2 segmentation video + frames)
output/cotracker_demo/                  (CoTracker track video + frames)

run_sam2_demo.py                        (SAM2 video propagation script)
run_cotracker_demo.py                   (CoTracker point tracking script)
STATUS.md                               (this file)
```
