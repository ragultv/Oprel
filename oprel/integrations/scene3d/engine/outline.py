"""
Helpers for shapes built from 2D outlines: point parsing, shapely checks (so a bad outline
comes back as an error the model can act on instead of a broken mesh), Catmull-Rom smoothing,
revolving a profile, and hollow vessel profiles.

Shared by shapes/profiles.py (lathe, extrude) and the ready-made vessels in shapes/objects.py.
It registers no shapes, so importing it doesn't change the order of the shape list.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import trimesh
from shapely.geometry import Polygon
from shapely.validation import explain_validity
from trimesh import transformations as tf

from oprel.integrations.scene3d.engine.frame import Z_UP_TO_Y_UP
from oprel.integrations.scene3d.engine.registry import ToolArgumentError

_POINT = {"type": "array", "items": {"type": "number"}, "maxItems": 2}
POINTS = {"type": "array", "items": _POINT, "maxItems": 200}
# Ground outlines: polygon (x, y) -> (X, Z), extruded downward; the compiler re-anchors the bottom.
FLAT_TO_GROUND = tf.rotation_matrix(np.pi / 2, [1, 0, 0])


def pairs(points: Any, name: str, minimum: int) -> list[tuple[float, float]]:
    try:
        out = [(float(p[0]), float(p[1])) for p in points]
    except (TypeError, ValueError, IndexError) as exc:
        raise ToolArgumentError(f"'{name}' must be a list of [a, b] number pairs") from exc
    # Drop consecutive duplicates: they make degenerate faces.
    out = [p for i, p in enumerate(out) if i == 0 or np.hypot(p[0] - out[i - 1][0], p[1] - out[i - 1][1]) > 1e-9]
    if len(out) < minimum:
        raise ToolArgumentError(f"'{name}' needs at least {minimum} different points")
    return out


def explain(poly: Polygon, what: str, hint: str) -> None:
    if poly.is_valid and poly.area > 1e-10:
        return
    reason = explain_validity(poly) if not poly.is_valid else "it has no area"
    raise ToolArgumentError(f"{what} is not a valid outline ({reason}). {hint}")


def catmull_rom(points: Sequence[tuple[float, float]], samples: int = 8) -> list[tuple[float, float]]:
    """Smooth curve through every point (uniform Catmull-Rom), keeping both ends."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return [tuple(p) for p in pts]
    padded = np.vstack([2 * pts[0] - pts[1], pts, 2 * pts[-1] - pts[-2]])
    out = []
    for i in range(1, len(padded) - 2):
        p0, p1, p2, p3 = padded[i - 1:i + 3]
        for t in np.linspace(0, 1, samples, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(pts[-1])
    # The curve can overshoot between points; keep it within the given heights so the
    # object's height matches its parameters exactly.
    y_lo, y_hi = pts[:, 1].min(), pts[:, 1].max()
    # Same for radii: a dip below the smallest given radius can pinch the shape shut.
    r_lo, r_hi = pts[:, 0].min(), pts[:, 0].max()
    return [(float(min(max(r, r_lo), r_hi)), float(min(max(y, y_lo), y_hi))) for r, y in out]


def revolve_profile(profile: Sequence[tuple[float, float]], segments: int) -> trimesh.Trimesh:
    """
    Spin a (radius, height) outline around the Y axis. The outline is closed along the axis:
    its first and last points are joined to the axis if they aren't on it already.
    """
    pts = [(float(r), float(y)) for r, y in profile]
    if pts[0][0] > 1e-9:
        pts.insert(0, (0.0, pts[0][1]))
    if pts[-1][0] > 1e-9:
        pts.append((0.0, pts[-1][1]))
    explain(Polygon(pts), "The lathe profile",
             "List [radius, height] points from the bottom up along ONE side of the axis; "
             "the profile must not cross itself.")
    mesh = trimesh.creation.revolve(np.asarray(pts), sections=segments)
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    if mesh.volume < 0:  # winding depends on the profile's direction
        mesh.invert()
    return mesh.apply_transform(Z_UP_TO_Y_UP)


def vessel_profile(outer: Sequence[tuple[float, float]], wall: float, floor: float) -> list[tuple[float, float]]:
    """
    Outline of an open-topped vessel: up the outside (`outer`, bottom to top), across the rim,
    and down the inside `wall` metres in, to a floor at height `floor`. Revolving it gives a
    hollow, watertight mesh with no boolean operations.
    """
    outer = list(outer)
    inner = [(r - wall, y) for r, y in reversed(outer) if y >= floor and r - wall > 0.0005]
    if not inner:
        raise ToolArgumentError("The wall is thicker than the vessel is wide")
    return [(0.0, outer[0][1]), *outer, *inner, (0.0, inner[-1][1])]
