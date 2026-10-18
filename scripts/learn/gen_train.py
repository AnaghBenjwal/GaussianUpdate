#!/usr/bin/env python3
"""
gen_train.py — training / validation samples for the learned displacement regressor (AI2THOR, dev env).

Same simulator build, camera model (vertical FOV 90, standing agent, 1920x1080 -> 1600x900), object move and
"physically clean" checks as scripts/bench/gen_trials.py, but with many more scenes, objects, viewpoints and moves:

  * scene configs: k = 0 is the default scene; k > 0 adds InitialRandomSpawn (new object layout) and sometimes
    RandomizeMaterials / RandomizeLighting (appearance diversity). Before and after always share the config.
  * views: reachable agent positions 0.5-2.5 m from the object, yaw towards it (+-15 deg), horizon in {0, 20, 30}
    (dataset uses 0 / 20); or, with --dataset_views, the original dataset cameras (dataset_1080p_fixed true_poses).
  * moves: random direction on the floor plane, |delta| log-uniform in [3, 40] cm, object keeps its rotation.
    Kept only if the object stays on its support (|dy| <= 1 cm), lands within 3 cm of the target, rotates <= 3 deg
    and no other object moves > 1 cm. The ground truth is the object's actual move.
  * every frame is rendered twice and must match bit for bit (local GPU has transient memory errors), and black
    frames (an intermittent CloudRendering failure) restart the controller.

    conda run -n dev python scripts/learn/gen_train.py --split train --scenes 6-30 --worker 0 --workers 2
    conda run -n dev python scripts/learn/gen_train.py --split val --scenes 2,4,5 --dataset_views

Writes <out>/<split>/<scene>/<sample_id>/ (sample_io.py format) and <out>/logs/<split>_<scene>.jsonl (all attempts).
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sample_io as io

FOV = 90.0
HORIZONS = (0.0, 20.0, 30.0)
EXCLUDE_TYPES = {"DiningTable", "ShelvingUnit", "Chair", "Stool", "Sink", "SinkBasin", "CounterTop", "Fridge"}
MIN_PX_SRC = 200                          # object pixels at 1920x1080 for a usable view
MIN_PX_FULL = 120                         # at 1600x900, before and after, for a written sample
MAX_FRAC = 0.25                           # object may cover at most this fraction of the frame


VERIFY = False                            # --verify: render every frame twice (faulty local GPU)


class BlackFrame(RuntimeError):
    pass


class GPUMismatch(RuntimeError):
    pass


def parse_scenes(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return [f"FloorPlan{i}" for i in out]


def obj_mask(seg, colors, oid):
    """Mask of one object from the instance-segmentation frame (ev.instance_masks decodes every object: slow)."""
    c = colors.get(oid)
    if c is None:
        return np.zeros(seg.shape[:2], bool)
    return (seg == np.array(c, seg.dtype)).all(-1)


def pos(o):
    return np.array([o["position"][a] for a in "xyz"], float)


def ang(a, b):
    return max(abs(((b[k] - a[k]) + 180) % 360 - 180) for k in "xyz")


class Sim:
    def __init__(self, scene):
        self.scene = scene
        self.c = None
        self.start()

    def start(self):
        import ai2thor.controller
        if self.c is not None:
            try:
                self.c.stop()
            except Exception:
                pass
        for attempt in range(10):                 # Unity sometimes segfaults in Vulkan init; just retry
            try:
                self.c = ai2thor.controller.Controller(agentMode="default", scene=self.scene, platform="CloudRendering",
                                                       renderInstanceSegmentation=True, renderDepthImage=True,
                                                       width=io.SRC_W, height=io.SRC_H, fieldOfView=FOV, snapToGrid=False)
                f = self.c.last_event.frame
                if f is not None and f.mean() >= 2:
                    return
                self.c.stop()
            except Exception as e:
                print(f"  controller start failed ({attempt}): {str(e)[:80]}", flush=True)
            time.sleep(2)
        raise RuntimeError("AI2THOR would not start")

    def step(self, **kw):
        return self.c.step(**kw)

    def objects(self, ev=None):
        ev = ev or self.c.last_event
        return {o["objectId"]: o for o in ev.metadata["objects"]}

    def settle(self, n=10):
        for _ in range(n):
            ev = self.step(action="Pass", renderImage=False)
        return ev

    def teleport(self, cam):
        ev = self.step(action="TeleportFull", x=cam["agent"][0], y=cam["agent"][1], z=cam["agent"][2],
                       rotation={"x": 0.0, "y": cam["yaw"], "z": 0.0}, horizon=cam["horizon"], standing=True)
        return ev if ev.metadata["lastActionSuccess"] else None

    def shoot(self, cam):
        """Render a view twice; both must agree exactly. Returns (rgb, depth, instance_masks, camera position)."""
        ev = self.teleport(cam)
        if ev is None:
            return None
        frames = []
        for i in range(3):
            if i:
                ev = self.step(action="Pass")
            f = ev.frame
            if f is None or f.mean() < 2 or ev.depth_frame is None or ev.depth_frame.max() <= 0:
                raise BlackFrame(f"shoot i={i} mean={None if f is None else f.mean():.1f} dmax={None if ev.depth_frame is None else ev.depth_frame.max()} cam={cam}")
            frames.append((f.copy(), ev.depth_frame.copy(), (ev.instance_segmentation_frame.copy(), dict(ev.object_id_to_color)),
                           ev.metadata["cameraPosition"]))
            if not VERIFY:
                return frames[0]
            if i == 1 and np.array_equal(frames[0][0], frames[1][0]) and np.array_equal(frames[0][1], frames[1][1]):
                return frames[0]
        same = lambda a, b: np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
        if same(frames[1], frames[2]):
            return frames[1]
        if same(frames[0], frames[2]):
            return frames[0]
        raise GPUMismatch(self.scene)


def init_config(sim, k, seed, log):
    sim.c.reset(scene=sim.scene)
    info = dict(config=k, spawn_seed=None, materials=False, lighting=False)
    if k > 0:
        info["spawn_seed"] = seed
        ev = sim.step(action="InitialRandomSpawn", randomSeed=seed, forceVisible=False, numPlacementAttempts=5,
                      placeStationary=True)
        if not ev.metadata["lastActionSuccess"]:
            log(dict(event="spawn_failed", config=k, err=ev.metadata["errorMessage"][:200]))
        rng = np.random.default_rng(seed)
        if rng.random() < 0.5:
            sim.step(action="RandomizeMaterials"); info["materials"] = True
        if rng.random() < 0.3:
            sim.step(action="RandomizeLighting", synchronized=True); info["lighting"] = True
    sim.settle(30)
    # wait until nothing drifts any more (some objects slide for a while after a reset)
    for _ in range(5):
        a = sim.objects()
        b = sim.objects(sim.settle(10))
        if max(np.linalg.norm(pos(b[i]) - pos(a[i])) for i in a) < 0.001:
            break
    return info


def eligible(objs):
    return [o for o in objs.values() if (o["pickupable"] or o["moveable"]) and o["objectType"] not in EXCLUDE_TYPES]


def pick_views(sim, o, reach, rng, n, dataset_poses=None):
    op = pos(o)
    if dataset_poses is not None:
        cands = [dict(agent=p["spec_eye"], yaw=p["yaw"], horizon=p["spec_horizon"], source="dataset", name=p["name"])
                 for p in dataset_poses]
        rng.shuffle(cands)
    else:
        cands = []
        R = np.array([[p["x"], p["y"], p["z"]] for p in reach])
        d = np.hypot(R[:, 0] - op[0], R[:, 2] - op[2])
        idx = np.nonzero((d > 0.5) & (d < 2.5))[0]
        rng.shuffle(idx)
        for i in idx[:16]:
            a = R[i]
            yaw = np.degrees(np.arctan2(op[0] - a[0], op[2] - a[2])) + rng.uniform(-15, 15)
            down = np.degrees(np.arctan2((a[1] + 0.675) - op[1], d[i]))       # standing camera ~0.675 m above agent y
            hor = min(HORIZONS, key=lambda h: abs(h - down))
            cands.append(dict(agent=a.tolist(), yaw=float(yaw % 360), horizon=float(hor), source="sampled"))
    good = []
    for cam in cands:
        if len(good) >= 3 * n or (dataset_poses is None and len(good) >= n + 2):
            break
        ev = sim.teleport(cam)
        if ev is None:
            continue
        ev = sim.step(action="Pass")
        if ev.frame is None or ev.frame.mean() < 2:
            raise BlackFrame(f"pick_views mean={None if ev.frame is None else ev.frame.mean()} cam={cam}")
        px = int(obj_mask(ev.instance_segmentation_frame, ev.object_id_to_color, o["objectId"]).sum())
        if MIN_PX_SRC <= px <= MAX_FRAC * io.SRC_W * io.SRC_H:
            cam["px"] = px
            good.append(cam)
    # spread the chosen views over different directions
    chosen = []
    for cam in sorted(good, key=lambda c: -c["px"]):
        if len(chosen) >= n:
            break
        if all(abs(((cam["yaw"] - c["yaw"]) + 180) % 360 - 180) > 20 or np.linalg.norm(np.subtract(cam["agent"], c["agent"])) > 0.3
               for c in chosen):
            chosen.append(cam)
    return chosen


def sample_delta(rng):
    r = np.exp(rng.uniform(np.log(0.03), np.log(0.40)))
    t = rng.uniform(0, 2 * np.pi)
    return np.array([r * np.cos(t), 0.0, r * np.sin(t)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--scenes", default="6-30")
    ap.add_argument("--configs", type=int, default=8)
    ap.add_argument("--objs", type=int, default=12, help="objects per config")
    ap.add_argument("--moves", type=int, default=2, help="kept moves per object")
    ap.add_argument("--views", type=int, default=3, help="views per object")
    ap.add_argument("--dataset_views", action="store_true", help="use the dataset_1080p_fixed cameras of the scene")
    ap.add_argument("--data", default=str(Path.home() / "Documents/dataset_1080p_fixed"))
    ap.add_argument("--out", default=str(Path.home() / "Documents/displacement_learn"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--verify", action="store_true", help="render every frame twice and require identical output")
    args = ap.parse_args()
    global VERIFY
    VERIFY = args.verify
    os.environ.setdefault("DISPLAY", ":99")

    tasks = [(sc, k) for sc in parse_scenes(args.scenes) for k in range(args.configs)][args.worker::args.workers]
    out = Path(args.out)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    sim = scene = reach = None
    for task_scene, k in tasks:
        sdir = out / args.split / task_scene
        if (sdir / f"_config{k}_DONE").exists():
            continue
        if task_scene != scene:
            if sim is not None:
                sim.c.stop()
            scene, sim = task_scene, Sim(task_scene)
            reach = None
            for _ in range(5):                    # sometimes returns None right after start-up
                reach = sim.step(action="GetReachablePositions").metadata["actionReturn"]
                if reach:
                    break
                sim.settle(5)
        logf = open(out / "logs" / f"{args.split}_{scene}_c{k}.jsonl", "a")
        log = lambda r: (logf.write(json.dumps(r) + "\n"), logf.flush())
        if not reach:
            log(dict(event="no_reachable_positions")); print(f"{scene}: no reachable positions", flush=True); continue
        poses = json.load(open(f"{args.data}/{scene}/env/scene_meta.json"))["true_poses"] if args.dataset_views else None
        t0 = time.time()
        for attempt in range(3):
            rng = np.random.default_rng([args.seed, int(scene.replace("FloorPlan", "")), k, attempt])
            try:
                n = run_config(sim, scene, k, args, rng, reach, poses, sdir, log)
                sdir.mkdir(parents=True, exist_ok=True)
                (sdir / f"_config{k}_DONE").touch()
                print(f"{scene} config {k}: {n} samples in {time.time() - t0:.0f}s", flush=True)
                break
            except (BlackFrame, GPUMismatch) as e:
                log(dict(event=type(e).__name__, config=k, attempt=attempt, msg=str(e)[:300]))
                print(f"  {scene} config {k}: {type(e).__name__}, restarting controller", flush=True)
                sim.start()
            except Exception as e:
                log(dict(event="error", config=k, err=traceback.format_exc()[-800:]))
                print(f"  {scene} config {k}: error {e!r}", flush=True)
                sim.start()
    if sim is not None:
        sim.c.stop()


def run_config(sim, scene, k, args, rng, reach, poses, sdir, log):
    seed = int(rng.integers(1 << 30))                 # spawn seed of this config
    info = init_config(sim, k, seed, log)
    snap = sim.objects()
    movable_others = [i for i, o in snap.items() if o["pickupable"] or o["moveable"]]
    objs = eligible(snap)
    rng.shuffle(objs)
    n_obj = n_written = 0
    for o in objs:
        if n_obj >= args.objs:
            break
        oid = o["objectId"]
        views = pick_views(sim, o, reach, rng, args.views, poses)
        if not views:
            log(dict(event="no_view", config=k, obj=oid)); continue
        n_obj += 1
        before = None
        kept = tries = 0
        while kept < args.moves and tries < 3 * args.moves + 2:
            tries += 1
            if before is None:
                before = {}
                for vi, cam in enumerate(views):
                    s = sim.shoot(cam)
                    if s is not None:
                        before[vi] = s
            p0, r0 = pos(snap[oid]), snap[oid]["rotation"]
            req = sample_delta(rng)
            move_id = int(rng.integers(1 << 24))
            ev = sim.step(action="TeleportObject", objectId=oid, x=p0[0] + req[0], y=p0[1], z=p0[2] + req[2],
                          rotation=r0, forceAction=True)
            ok = ev.metadata["lastActionSuccess"]
            now = sim.objects(sim.settle(10))
            moved = pos(now[oid]) - p0
            others = max((np.linalg.norm(pos(now[i]) - pos(snap[i])) for i in movable_others if i != oid), default=0.0)
            reason = None
            if not ok: reason = "teleport_failed"
            elif abs(moved[1]) > 0.01: reason = f"height {moved[1]:+.3f}"
            elif np.hypot(*(moved - req)[[0, 2]]) > 0.03: reason = "missed_target"
            elif ang(r0, now[oid]["rotation"]) > 3: reason = "rotated"
            elif others > 0.01: reason = f"pushed {others:.3f}"
            written = []
            if reason is None:
                for vi, cam in enumerate(views):
                    if vi not in before:
                        continue
                    a = sim.shoot(cam)
                    if a is None:
                        continue
                    b_rgb, b_dep, b_inst, cpos = before[vi]
                    mb = io.resize_nearest(obj_mask(*b_inst, oid))
                    ma = io.resize_nearest(obj_mask(*a[2], oid))
                    if mb.sum() < MIN_PX_FULL or ma.sum() < MIN_PX_FULL:
                        continue
                    name = f"{scene}_c{k}_{oid.split('|')[0]}_{move_id:06x}_v{vi}"
                    cam_info = dict(pos_unity=[cpos["x"], cpos["y"], cpos["z"]], yaw=cam["yaw"], horizon=cam["horizon"], fov=FOV)
                    meta = io.make_meta(scene, name, args.split, snap[oid], cam_info, moved, dict(
                        info, requested_unity=req.tolist(), rot_change_deg=ang(r0, now[oid]["rotation"]),
                        view_source=cam["source"], view_name=cam.get("name"), pushed_max_m=float(others)))
                    io.write_sample(sdir / name, io.resize_rgb(b_rgb), io.resize_rgb(a[0]),
                                    io.resize_nearest(b_dep.astype(np.float32)), mb, ma, meta)
                    written.append(name)
                if written:
                    kept += 1; n_written += len(written)
                else:
                    reason = "not_visible_after"
            log(dict(event="move", config=k, obj=oid, req=req.round(3).tolist(), moved=moved.round(3).tolist(),
                     reason=reason, written=written))
            # restore the object; if the scene does not come back exactly, start the config's layout again
            sim.step(action="TeleportObject", objectId=oid, x=p0[0], y=p0[1], z=p0[2], rotation=r0, forceAction=True)
            back = sim.objects(sim.settle(10))
            err = max(np.linalg.norm(pos(back[i]) - pos(snap[i])) for i in movable_others)
            if err > 0.001 or ang(r0, back[oid]["rotation"]) > 0.5:
                if err > 0.005:
                    log(dict(event="restore_failed", config=k, obj=oid, err=float(err)))
                    return n_written + run_rest(sim, scene, k, args, rng, reach, poses, sdir, log, info, seed, n_obj)
                snap = back
                before = None                    # tiny shift: re-render BEFORE for the next move
    return n_written


def run_rest(sim, scene, k, args, rng, reach, poses, sdir, log, info, seed, n_obj):
    """Scene did not restore: rebuild the same config (same spawn seed) and continue with the remaining object quota."""
    sub = argparse.Namespace(**vars(args))
    sub.objs = args.objs - n_obj
    if sub.objs <= 0:
        return 0
    return run_config(sim, scene, k, sub, rng, reach, poses, sdir, log)


if __name__ == "__main__":
    main()
