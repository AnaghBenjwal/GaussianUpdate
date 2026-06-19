# Algorithm: Gaussian Grouping Change Detection and Update

*Plain-English explanation of `scripts/detect_and_update.py` for presentation use.*

---

## What the problem is

We have a static 3D scene represented as a **Gaussian Grouping (GG)** model — a cloud of
millions of tiny 3D blobs ("Gaussians"), each with a position, size, orientation, colour,
and an object-group label.  The model was trained on a set of photographs taken from known
viewpoints.

Now the real world has changed: one object moved.  We get a new photograph from one of the
same viewpoints.  Goal: update the 3D model so that Gaussians belonging to the moved
object reflect its new position.

---

## Step-by-step walkthrough

### Step 1 — Render the "expected" (before) image

We use the *existing* trained GG model to synthesise what the camera should see if nothing
had changed.  The GG rasteriser projects all 3.6 million Gaussians onto the image plane and
alpha-composites them, producing a photo-realistic render.

**Output:** `before_render.png` — a clean RGB image at exactly the training camera resolution.

---

### Step 2 — Compare before vs. after to find the changed region

We compute a per-pixel absolute difference between the before render and the new "after"
photograph.  Pixels that changed more than a threshold (default: 25 intensity units) are
flagged.  The raw binary mask is cleaned up with morphological closing and dilation to fill
small gaps and smooth edges.  We then keep only the **largest connected component** to
avoid reacting to noise or minor lighting changes elsewhere in the scene.

If SAM2 is available (running in the `dev` conda environment), we additionally run
SAM2's image predictor on the after image, using the bounding box of the rough
pixel-diff region as a prompt.  SAM2 produces a clean, semantically aware mask of the
moved object in the after image — this is more robust than pixel diff for real photos
with slight lighting changes.

**Output:** two binary masks — one aligned to the before render, one to the after image.

---

### Step 3 — Identify which Gaussians belong to the changed object

The GG model stores a 16-dimensional **object feature** vector for each Gaussian (learned
during training to cluster Gaussians by object identity).  A small convolutional classifier
(a single 1×1 convolution learned jointly with the 3DGS model) maps these 16 features to
a set of object class IDs.

We apply this classifier to every Gaussian's features directly (not through rendering) to
predict a class ID per Gaussian.  Then we count which class dominates the pixels inside
the changed region in the before render.  All Gaussians assigned to that class are selected
for updating.

**Key invariant:** the same classifier that colours the rendered segmentation masks is used
here, so the per-Gaussian labels are perfectly consistent with what you see in the render.

**Output:** a list of ~5,000–100,000 Gaussian indices belonging to the moved object.

---

### Step 4 — Estimate the 2-D pixel displacement

We need to know where the object moved *on screen* between before and after.

**Method used (primary): Template matching**
We crop the bounding box of the object from the before render, neutralise the non-object
background pixels to grey, and slide this template across the after image using OpenCV's
normalised cross-correlation.  The best-matching position tells us exactly where the object
ended up, giving a pixel shift (Δu, Δv).

**Alternative (when SAM2 is available):** run SAM2 on the after image with the diff-mask
bounding box as a prompt; take the centroid of the SAM2 mask as the new object centre.

**Future improvement:** CoTracker can track a grid of points inside the object across a
short video clip of frames, giving a dense motion field instead of a single centroid shift.

---

### Step 5 — Convert 2-D shift to a 3-D world-space displacement

A 2-D pixel shift is ambiguous without knowing the depth of the object.  We recover
depth by directly reading the camera-space z-coordinate of the selected Gaussians
(no depth rendering needed — we have exact 3D positions for every Gaussian).

With depth `z` known, we use the pinhole camera back-projection formula:

```
X_cam = (u - cx) / fx × z
Y_cam = (v - cy) / fy × z
```

The before and after 3D points in camera space differ by `(ΔX_cam, ΔY_cam, 0)`.  
We then rotate this vector from camera space to world space using the inverse of the
camera's view-rotation matrix:

```
Δ_world = R_cam→world  ×  (ΔX_cam, ΔY_cam, 0)
```

The z=0 assumption means we treat the motion as planar at the object's current depth.
This is exact for lateral motion and a good approximation when the out-of-plane component
is small.

---

### Step 6 — Apply the displacement to the selected Gaussians

We add `Δ_world` to the xyz position of every selected Gaussian:

```python
gaussians._xyz[selected_indices] += delta_world
```

No training or gradient computation is needed.  The model is modified in-place.

---

### Step 7 — Re-render and compare

The updated model is rendered at the same camera pose.  This produces an `updated_render.png`
that should show the object in its new position.

We compute:
- **Before vs. After:** how much the scene changed (ground truth error we want to correct).
- **After vs. Updated:** residual error after our correction.
- **Before vs. Updated:** how much our update moved things (sanity check — should be small
  outside the moved object, large at the object).

---

## Limitations and future work

| Limitation | Status |
|---|---|
| Only handles lateral (in-plane) motion; depth change not estimated | Acceptable for most camera angles |
| Uses a single reference camera; multi-view would improve depth accuracy | Future: use several views |
| Template matching fails for rotation or scale changes | Future: SAM2 + optical flow or CoTracker |
| Gaussian Grouping may merge multiple objects into one class | Depends on training quality |
| Updated model has no Gaussian densification (thin regions may show holes) | Future: trigger GG densification after displacement |

---

## How to swap in real inpainted before/after images

*This is exactly what changes tomorrow:*

1. Place your real "after" frame at `data/after_frames/cam_000_after.png`.
2. Determine which training camera index it corresponds to (check
   `output/lerf_mask/figurines/train/ours_30000/renders/` — the filenames map to
   `scene.getTrainCameras()[idx]`).
3. Run:
   ```bash
   conda run -n gaussian_grouping python scripts/detect_and_update.py \
       --model_path  output/lerf_mask/figurines \
       --source_path data/lerf_mask/figurines \
       --cam_idx     <INDEX>                    \
       --after       data/after_frames/cam_000_after.png \
       --output      output/presentation_real
   ```
4. Check `output/presentation_real/report.json` for the displacement vector and
   `output/presentation_real/06_comparison/comparison.png` for the visual result.

The only difference from the synthetic test is that `--synthetic` is not passed and
`--after` points to your real image.  SAM2 will automatically be invoked for mask
refinement (requires the `dev` conda env to be active and `weights/sam2/` to be present).
