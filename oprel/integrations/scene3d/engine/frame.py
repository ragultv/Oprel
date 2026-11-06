"""
Coordinate conventions shared by every part of the engine.

  units      metres
  up         +Y, the ground is y = 0
  camera     looks toward -Z, so right = +X, front = +Z, top = +Y
  origin     every object's origin is the bottom-centre of its mesh
  rotation   degrees [x, y, z] (static X, then Y, then Z), about the shape's centre; the shape
             is then lifted/lowered so its lowest point stays put (it keeps resting where it was)
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from trimesh import transformations as tf

IDENTITY = np.eye(4)


def translation(v: Sequence[float]) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = v
    return m


def rotation_deg(r: Sequence[float] | None) -> np.ndarray:
    if not r or not any(r):
        return np.eye(4)
    rx, ry, rz = np.radians(vec3(r))
    return tf.euler_matrix(rx, ry, rz, axes="sxyz")


def rotation_y_deg(angle: float) -> np.ndarray:
    return tf.rotation_matrix(np.radians(angle), [0, 1, 0])


# Builders that use trimesh's Z-axis helpers (cylinder, cone, torus) turn the result Y-up.
Z_UP_TO_Y_UP = tf.rotation_matrix(-np.pi / 2, [1, 0, 0])


def vec3(v: Sequence[float] | None, default: float = 0.0) -> list[float]:
    """Pad/trim a model-supplied vector to 3 floats ([1, 2] -> [1, 2, 0])."""
    out = [float(x) for x in (v or [])][:3]
    return out + [default] * (3 - len(out))


def transform_points(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ m[:3, :3].T + m[:3, 3]


def aabb(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return pts.min(axis=0), pts.max(axis=0)
