# Learned displacement regressor — design

## Task
Given one camera view of an object BEFORE and AFTER it was moved, predict the object's 3D translation.
It replaces the displacement step of the pipeline (identification and selection stay as they are) and is also
a fifth cue for the agree rule.

## Model input (one 256x256 crop around the object mask ∪ change mask, 11 channels)
| channels | content | at test time | at train time |
|---|---|---|---|
| 3 | BEFORE RGB | real before photo | simulator render |
| 3 | AFTER RGB | real after photo | simulator render |
| 1 | object mask in BEFORE | GG alpha of selected Gaussians (A0 > 0.3) | instance mask (+ erode/dilate noise) |
| 3 | BEFORE point map: depth back-projected to the gravity-aligned camera frame, minus the object anchor (median 3D point of the mask), metres | GG depth | simulator depth (+ GG-like noise and holes) |
| 1 | depth valid | GG alpha > 0.9 | 1 (random holes) |

There is also a 5-d global vector: anchor distance, metres per pixel of the crop, and the anchor ray direction.

## Model output
δ = (dx, dy, dz) in metres in the gravity-aligned camera frame (x = right, y = up, z = forward on the floor),
plus a per-axis log-scale (uncertainty). Because this frame doesn't depend on the camera yaw, δ_world = R_yawᵀ δ.

## Architecture
ResNet-18 (ImageNet weights, first conv widened to 11 channels: the RGB weights are copied to both images and the
other channels start at zero) → global average pool → concat global vector → MLP(512 → 256 → 6). About 11M parameters.
Loss: Laplace NLL on δ. Augmentations:
- crop jitter
- shared colour jitter
- horizontal flip (negates x of both the point map and δ)
- depth and mask noise

## Data (sample_io.py format: context crop at 1600x900 scale, ~0.5 MB/sample)
| split | scenes | content |
|---|---|---|
| train | AI2THOR kitchens FloorPlan6-30 | default and random layouts (InitialRandomSpawn), random materials and lighting; random views 0.5-2.5 m, horizon 0/20/30; random moves, direction uniform, 3-40 cm log-uniform; about 10k samples |
| val | FloorPlan2, 4, 5 | original dataset_1080p_fixed cameras |
| test T1 | FloorPlan1, 3 | the 37 benchmark trials (unchanged); inputs from GG for the main table, simulator depth/mask for the gap study |
| test T2 | FloorPlan1, 3 | new trials (random direction and size, every GG held-out view as input); all methods are run on it |

FloorPlan1 and FloorPlan3 never appear in training or validation.

## Final evaluation
All methods are evaluated on T1 and T2:
- P1 / P2 originals
- diffRC
- each external cue (RAFT-crop, RoMa, DepthAnything, SAM, MASt3R)
- the agree rule
- the learned model
- the learned model + agree rule
- the learned model + local RC refinement

Metrics:
- error (median, mean, <5/<10 cm)
- hard subset
- per object
- by object size and truncation
- novel-view PSNR
- runtime
