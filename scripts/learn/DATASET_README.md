# displacement_learn — object displacement dataset

One sample = one camera view of one object that was moved on its support surface: BEFORE photo, AFTER photo, BEFORE
depth, the object's BEFORE mask, the camera, and the ground-truth 3D move. Rendered with AI2THOR 5.0.0
(CloudRendering): the same build, camera model (vertical FOV 90°, standing agent, 1920x1080 → 1600x900) and
"physically clean move" rules as the original Gaussian-Grouping benchmark. Code: `gaussian-grouping/scripts/learn/`.

## Layout
```
displacement_learn/
  train/<scene>/<sample_id>/      AI2THOR kitchens FloorPlan6-30 (FloorPlan8 has no reachable positions → empty)
  val/<scene>/<sample_id>/        FloorPlan2, 4, 5 seen from the original dataset_1080p_fixed cameras
  test_t1_sim/<scene>/<id>/       benchmark T1 (37 trials, FloorPlan1/3, input frame_00000), simulator depth + mask
  test_t2_sim/<scene>/<id>/       new held-out T2 (FloorPlan1/3, random moves, every visible GG test view as input)
  bench/displacement_bench/       T1 in the raw benchmark format (full frames, all 8-10 views, + before_depth/)
  bench/displacement_bench_t2/    T2 in the raw benchmark format (meta.json has input_views)
  index.jsonl                     one row per sample (split, scene, object, |δ|, pixels, truncation, …)
  STATS.md                        counts and distributions per split
  logs/                           every attempted move with its accept/reject reason
  code/                           the generation / loading / training scripts (snapshot of scripts/learn)
```
Sizes: train 10,267 samples (3,618 moves, 24 scenes, 42 object types), val 450 (171 moves), test_t1_sim 37,
test_t2_sim 368 (67 moves). See STATS.md. Training views are aimed at the object, so none are cut off by the frame
edge (13-49% in val/test); data.py simulates frame-edge truncation in training to cover that.
FloorPlan1 and FloorPlan3 (the scenes with trained GG models, i.e. the test scenes) never appear in train/val.
The `test_*_gg` variants (depth and object mask rendered from the GG model, which is what the real pipeline sees) are
made at evaluation time by `gg_inputs.py` + `export_bench.py --variant gg` from the `bench/` folders.

## Sample files (context crop around the change at 1600x900 scale; crops wider than 640 px are downscaled)
| file | content |
|---|---|
| before.png, after.png | RGB uint8 |
| depth.png | BEFORE depth, uint16 millimetres along the camera axis (0 = invalid) |
| mask_before.png | object mask in BEFORE (model input) |
| mask_after.png | object mask in AFTER (ground truth only, never a model input) |
| meta.json | camera (`cam_pos_unity`, `cam_yaw`, `cam_horizon`, `fov_v`, `K_full`, `K_crop`, `crop_box_full`, `crop_scale`), ground truth (`delta_unity`, `delta_gg`, `delta_grav`, `delta_norm`), object (`object_id`, `object_type`, `obj_pos_unity`, `px_before/after`, `truncated_before/after`), and generation info (`config`, `spawn_seed`, `materials`, `lighting`, `view_source`, `requested_unity`, `rot_change_deg`) |

Frames: `unity` = AI2THOR world (y up, left-handed). `gg` = diag(1,1,-1)·unity, the fixed-dataset / GG world.
`grav` = gravity-aligned camera frame: x = camera right, y = up, z = camera forward projected on the floor. This is
the model's prediction frame: δ_unity = R(yaw)ᵀ δ_grav (`sample_io.grav_rotation`).

## How the moves were made (gen_train.py)
- Scene configs per scene: config 0 is the default layout; configs > 0 use InitialRandomSpawn with a random seed,
  plus RandomizeMaterials (50%) and RandomizeLighting (30%). Physics is settled before anything is rendered.
- Objects: every pickupable or moveable object except furniture-sized ones, up to 12 per config.
- Views: up to 3 per object.
  - Train: reachable agent positions 0.5–2.5 m away, yaw towards the object ±15°, horizon 0/20/30.
  - Val: the original dataset cameras.
  - The object must cover at least 200 px at 1920x1080 and at most 25% of the frame.
- Moves: TeleportObject to the same height and rotation, random direction, |δ| log-uniform 3–40 cm. A move is kept
  only if all of these hold:
  - height change ≤ 1 cm
  - lands within 3 cm of the target
  - rotation ≤ 3°
  - no other object moved > 1 cm
  - at least 120 object px (1600x900) before and after in a view

  The ground truth is the actual (settled) move. The object is then put back and the scene checked to be unchanged.
