"""
colmap_writer.py — Write COLMAP sparse binary format from look-at camera definitions.

The look_at function computes a c2w matrix in OpenGL convention (right-hand, camera
looks in -Z). COLMAP uses OpenCV convention (X right, Y down, Z forward). This
module handles the conversion correctly.
"""
from __future__ import annotations
import struct
from pathlib import Path

import numpy as np


PINHOLE = 1


# ── Camera geometry ───────────────────────────────────────────────────────────

def look_at_c2w(eye: np.ndarray, target: np.ndarray,
                up: np.ndarray = np.array([0., 1., 0.])) -> np.ndarray:
    """
    Compute a 4×4 camera-to-world matrix in OpenGL convention.
    Camera looks in -Z, +Y is up, +X is right.

    eye, target, up: (3,) float64 arrays, world coordinates.
    Returns: (4, 4) float64 c2w matrix.
    """
    forward = target - eye
    forward = forward / np.linalg.norm(forward)  # look direction (+Z in scene)

    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        # forward is parallel to up; pick a fallback
        up = np.array([0., 0., 1.])
        right = np.cross(forward, up)
        right_norm = np.linalg.norm(right)
    right /= right_norm

    cam_up = np.cross(right, forward)  # orthogonal up (in camera frame)
    cam_up /= np.linalg.norm(cam_up)

    # OpenGL c2w: columns = [right, cam_up, -forward, position]
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = right
    c2w[:3, 1] = cam_up
    c2w[:3, 2] = -forward   # camera looks in -Z, so +Z is backward
    c2w[:3, 3] = eye
    return c2w


def c2w_to_colmap(c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a 4×4 OpenGL c2w to COLMAP (q, t).

    COLMAP: p_cam = R @ p_world + t  (OpenCV: X right, Y down, Z forward)

    Steps:
      1. Convert from OpenGL to OpenCV by negating Y and Z columns of c2w
      2. Invert → w2c
      3. Extract R, t
      4. R → quaternion [w, x, y, z]
    """
    c2w_cv = c2w.copy()
    c2w_cv[:3, 1:3] *= -1   # OpenGL (Y-up, Z-back) → OpenCV (Y-down, Z-fwd)

    w2c = np.linalg.inv(c2w_cv)
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    q = _rotmat_to_quat(R)
    return q, t


def ai2thor_lookat_params(eye: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """
    Compute AI2THOR (yaw_degrees, horizon_degrees) for a camera at `eye` looking at `target`.

    AI2THOR yaw: 0=facing +Z, 90=facing +X. Horizon: positive = look down.
    """
    forward = target - eye
    forward /= np.linalg.norm(forward)

    yaw_rad = np.arctan2(forward[0], forward[2])   # X/Z components
    yaw_deg = float(np.degrees(yaw_rad))

    # horizon = degrees looking down from horizontal
    horiz_dist = np.sqrt(forward[0] ** 2 + forward[2] ** 2)
    horizon_rad = np.arctan2(-forward[1], horiz_dist)  # +horizon = look down
    horizon_deg = float(np.degrees(horizon_rad))

    return yaw_deg, horizon_deg


# ── Binary writers ────────────────────────────────────────────────────────────

def _write_cameras_bin(path: Path, width: int, height: int,
                       fx: float, fy: float, cx: float, cy: float) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<i", PINHOLE))
        f.write(struct.pack("<QQ", width, height))
        f.write(struct.pack("<4d", fx, fy, cx, cy))


def _write_images_bin(path: Path, image_names: list[str],
                      quaternions: np.ndarray, translations: np.ndarray) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(image_names)))
        for i, name in enumerate(image_names):
            f.write(struct.pack("<I", i + 1))
            f.write(struct.pack("<4d", *quaternions[i]))
            f.write(struct.pack("<3d", *translations[i]))
            f.write(struct.pack("<I", 1))
            f.write(name.encode() + b"\x00")
            f.write(struct.pack("<Q", 0))   # 0 keypoints


def _write_points3d_bin(path: Path,
                        xyzs: np.ndarray, rgbs: np.ndarray) -> None:
    M = len(xyzs)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", M))
        for i in range(M):
            f.write(struct.pack("<Q", i + 1))
            f.write(struct.pack("<3d", *xyzs[i]))
            f.write(struct.pack("<3B", *rgbs[i].astype(np.uint8)))
            f.write(struct.pack("<d", 0.0))
            f.write(struct.pack("<Q", 0))   # empty track


# ── Depth back-projection to sparse 3D points ─────────────────────────────────

def depth_to_world_points(depth: np.ndarray, rgb: np.ndarray,
                          c2w: np.ndarray,
                          fx: float, fy: float, cx: float, cy: float,
                          subsample: int = 4,
                          max_depth: float = 8.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Back-project subsampled depth image pixels to world-space points.
    c2w is in OpenGL convention (camera looks -Z, Y up).
    Returns (xyz_world, rgb) arrays.
    """
    H, W = depth.shape
    ys = np.arange(0, H, subsample)
    xs = np.arange(0, W, subsample)
    xx, yy = np.meshgrid(xs, ys)
    d = depth[yy, xx].astype(np.float64)
    valid = (d > 0.05) & (d < max_depth)
    d = d[valid]; xx = xx[valid]; yy = yy[valid]

    # Camera-space coords (OpenCV: Y down, Z forward)
    xc = (xx - cx) / fx * d
    yc = (yy - cy) / fy * d
    zc = d
    pts_cam_cv = np.stack([xc, yc, zc, np.ones_like(xc)], axis=1)  # (N, 4)

    # Convert cam-cv to cam-opengl: negate Y and Z
    pts_cam_gl = pts_cam_cv.copy()
    pts_cam_gl[:, 1] *= -1
    pts_cam_gl[:, 2] *= -1

    # World coords: p_world = c2w @ p_cam_gl
    pts_world = (c2w @ pts_cam_gl.T).T[:, :3]
    colors = rgb[yy.astype(int), xx.astype(int)]
    return pts_world, colors


# ── Top-level writer ──────────────────────────────────────────────────────────

def write_colmap_sparse(
    out_dir: Path,
    image_names: list[str],
    c2w_matrices: list[np.ndarray],   # list of (4,4) OpenGL c2w
    width: int, height: int,
    fx: float, fy: float, cx: float, cy: float,
    depth_images: list[np.ndarray] | None = None,
    rgb_images: list[np.ndarray] | None = None,
    max_depth: float = 8.0,
) -> None:
    """Write cameras.bin, images.bin, points3D.bin to out_dir/sparse/0/."""
    sparse_dir = out_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    poses = [c2w_to_colmap(c2w) for c2w in c2w_matrices]
    quaternions = np.array([p[0] for p in poses])
    translations = np.array([p[1] for p in poses])

    _write_cameras_bin(sparse_dir / "cameras.bin", width, height, fx, fy, cx, cy)
    _write_images_bin(sparse_dir / "images.bin", image_names, quaternions, translations)

    if depth_images is not None and rgb_images is not None:
        all_xyz, all_rgb = [], []
        for depth, rgb, c2w in zip(depth_images, rgb_images, c2w_matrices):
            xyz, col = depth_to_world_points(
                depth, rgb, c2w, fx, fy, cx, cy,
                subsample=4, max_depth=max_depth
            )
            all_xyz.append(xyz)
            all_rgb.append(col)
        all_xyz = np.concatenate(all_xyz, axis=0)
        all_rgb = np.concatenate(all_rgb, axis=0)
        if len(all_xyz) > 300_000:
            idx = np.random.choice(len(all_xyz), 300_000, replace=False)
            all_xyz = all_xyz[idx]; all_rgb = all_rgb[idx]
        _write_points3d_bin(sparse_dir / "points3D.bin", all_xyz, all_rgb)
        print(f"  points3D: {len(all_xyz):,} back-projected points")
    else:
        _write_points3d_bin(sparse_dir / "points3D.bin",
                            np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8))

    print(f"COLMAP sparse written → {sparse_dir}  ({len(image_names)} images)")


# ── Quaternion helper ─────────────────────────────────────────────────────────

def _rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix → quaternion [w, x, y, z]."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s;                  z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float64)
