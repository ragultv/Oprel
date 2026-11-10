"""
Handles attached to the real surface of any solid: a mug, a vase, a bowl, a box, a cylinder, a
cone, a lathe or extrude profile, a rotated or grouped object.

Everything is read from the target's mesh by casting rays (no bounding boxes, no per-shape
profiles), so one method works for every shape:

  1. ends      a ray from outside at each end finds the surface point; a second hit on the same
               ray gives the wall's thickness, so the tips go into a thin wall but not through it
  2. clearance the surface between the ends is sampled and the grip is pushed out until it clears
               the body everywhere (a vase's belly, a bowl's flare) with room for fingers
  3. shape     round: one smooth cubic Bezier (an ear on an upright handle, a symmetric loop
               otherwise); box: square bar with square corners
  4. errors    an end that misses the object, or a span too short for the bar, comes back with the
               numbers that work

Frames: the handle lies in a plane. `side` is the outward direction d (the way the handle sticks
out); `span` U is the axis its two ends are spread along; K is the plane's normal. All three are
the frame's own axes (+-X, +-Y, +-Z), so a point is u*U + w*d + k*K with u, w, k its coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from oprel.integrations.scene3d.engine import sweep as sw
from oprel.integrations.scene3d.engine.registry import ToolArgumentError

AXES = {"right": np.array([1.0, 0, 0]), "left": np.array([-1.0, 0, 0]), "front": np.array([0, 0, 1.0]),
        "back": np.array([0, 0, -1.0]), "top": np.array([0, 1.0, 0])}
SIDES = tuple(AXES)
ORIENTATIONS = ("vertical", "horizontal")
_SAMPLES = 48


# ---------------------------------------------------------------------------
# rays
# ---------------------------------------------------------------------------

def ray_hits(mesh: trimesh.Trimesh, origins: np.ndarray, direction: np.ndarray, length: float,
             normals: list | None = None) -> list[np.ndarray]:
    """
    Distances along `direction` at which each ray from `origins` crosses the mesh, sorted
    (Moller-Trumbore, vectorised; only triangles near the rays are tested). No scipy/rtree needed.
    `normals`, when given, gets the outward normal of the first surface each ray hits (or None).
    """
    tris = np.asarray(mesh.triangles)
    face_normals = np.asarray(mesh.face_normals)
    seg_lo = np.minimum(origins, origins + direction * length).min(axis=0) - 1e-9
    seg_hi = np.maximum(origins, origins + direction * length).max(axis=0) + 1e-9
    near = np.all((tris.max(axis=1) >= seg_lo) & (tris.min(axis=1) <= seg_hi), axis=1)
    tris, face_normals = tris[near], face_normals[near]
    if not len(tris):
        if normals is not None:
            normals.extend([None] * len(origins))
        return [np.array([]) for _ in origins]
    v0, e1, e2 = tris[:, 0], tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    p = np.cross(direction, e2)
    det = np.einsum("ij,ij->i", e1, p)
    ok = np.abs(det) > 1e-14
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    out = []
    for o in origins:
        s = o - v0
        u = np.einsum("ij,ij->i", s, p) * inv
        q = np.cross(s, e1)
        v = (q @ direction) * inv
        t = np.einsum("ij,ij->i", e2, q) * inv
        hit = ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > 1e-9) & (t <= length)
        if normals is not None:
            idx = np.flatnonzero(hit)
            normals.append(face_normals[idx[np.argmin(t[idx])]] if len(idx) else None)
        ts = np.sort(t[hit])
        # One crossing through a shared edge is found on both triangles: keep it once.
        out.append(ts[np.concatenate([[True], np.diff(ts) > 1e-7])] if len(ts) else ts)
    return out


@dataclass
class Frame:
    d: np.ndarray   # outward: the way the handle sticks out
    U: np.ndarray   # the axis its ends are spread along
    K: np.ndarray   # normal of the handle's plane
    k0: float       # the plane's position along K

    def point(self, u: float, w: float) -> np.ndarray:
        return u * self.U + w * self.d + self.k0 * self.K


def frame_for(side: str, orientation: str, bounds: np.ndarray, cross_offset: float = 0.0) -> Frame:
    d = AXES[side]
    lo, hi = bounds
    if side == "top":
        # Over the top: ends spread along the object's longer horizontal side (a suitcase, a bucket).
        U = np.array([1.0, 0, 0]) if hi[0] - lo[0] >= hi[2] - lo[2] else np.array([0, 0, 1.0])
    elif orientation == "vertical":
        U = np.array([0, 1.0, 0])
    else:
        U = np.abs(np.cross(d, [0, 1.0, 0]))  # sideways along the face (a drawer pull, a pot's lugs)
    K = np.abs(np.cross(d, U))
    k0 = float(((lo + hi) / 2) @ K) + cross_offset
    return Frame(d, U, K, k0)


class Surface:
    """The target's outside as seen from the handle's side, in the handle's plane."""

    def __init__(self, mesh: trimesh.Trimesh, f: Frame):
        self.mesh, self.f = mesh, f
        lo, hi = mesh.bounds
        self.far = float(max(abs(hi @ f.d), abs(lo @ f.d))) + float(np.ptp(mesh.bounds, axis=0).max()) + 1.0
        self.length = 2 * self.far

    def _origins(self, us: np.ndarray) -> np.ndarray:
        return np.array([self.f.point(u, self.far) for u in us])

    def probe(self, us: np.ndarray, normals: list | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        (w of the outside surface, wall thickness along the ray) at each u; nan where nothing is
        hit. `normals` gets the surface's outward normal there in the handle's plane, as (u, w).
        """
        raw: list | None = [] if normals is not None else None
        hits = ray_hits(self.mesh, self._origins(us), -self.f.d, self.length, raw)
        w = np.array([self.far - h[0] if len(h) else np.nan for h in hits])
        wall = np.array([h[1] - h[0] if len(h) > 1 else np.nan for h in hits])
        if normals is not None:
            for n in raw:
                n2 = np.array([n @ self.f.U, n @ self.f.d]) if n is not None else np.array([0.0, 1.0])
                normals.append(_unit2(n2) if np.linalg.norm(n2) > 1e-9 else np.array([0.0, 1.0]))
        return w, wall

    def covered(self) -> tuple[float, float] | None:
        """The range of u where the object has a surface facing this side (for error messages)."""
        lo, hi = self.mesh.bounds
        a, b = float(lo @ self.f.U), float(hi @ self.f.U)
        if a > b:
            a, b = b, a
        us = np.linspace(a, b, 81)
        w, _ = self.probe(us)
        ok = us[~np.isnan(w)]
        return (float(ok.min()), float(ok.max())) if len(ok) else None


# ---------------------------------------------------------------------------
# the handle
# ---------------------------------------------------------------------------

def _bezier(ctrl: np.ndarray, n: int = 64) -> np.ndarray:
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 3 * ctrl[0] + 3 * (1 - t) ** 2 * t * ctrl[1] + 3 * (1 - t) * t ** 2 * ctrl[2] + t ** 3 * ctrl[3]


def _tip(direction: np.ndarray, r: float, wall: float) -> float:
    """
    How far a tip may run on into the object along `direction` ((u, w), pointing in): its deepest
    point (along the tip, plus the bar's end disc tilted with it) stays within 0.8 of the wall.
    """
    dw = max(abs(direction[1]), 1e-6)
    limit = 0.8 * wall if np.isfinite(wall) else 2 * r
    return float(max(0.0, min(2 * r, (limit - r * np.sqrt(max(0.0, 1 - dw * dw))) / dw)))


def build(mesh: trimesh.Trimesh, f: Frame, u_hi: float, u_lo: float, *, reach: float, thickness: float,
          style: str, ear: bool, what: str = "the object") -> trimesh.Trimesh:
    """
    A handle on `mesh` (in the same frame) with its ends at u_hi and u_lo along f.U, sticking out
    along f.d by `reach` past the furthest-out surface between them. Raises ToolArgumentError with
    the numbers that work.
    """
    r = thickness / 2
    if u_hi - u_lo < 2.5 * thickness:
        raise ToolArgumentError(f"its ends are {u_hi - u_lo:.3g} m apart, too close for a {thickness:g} m thick bar; "
                                f"use a height of at least {2.5 * thickness:.3g} m or a thinner bar")
    surf = Surface(mesh, f)
    end_normals: list = []
    (w_hi, w_lo), (wall_hi, wall_lo) = surf.probe(np.array([u_hi, u_lo]), end_normals)
    if np.isnan(w_hi) or np.isnan(w_lo):
        span = surf.covered()
        which = "upper" if np.isnan(w_hi) else "lower"
        at = u_hi if np.isnan(w_hi) else u_lo
        where = (f"it has a surface there from {span[0]:.3g} to {span[1]:.3g}" if span
                 else "nothing of it is in line with the handle")
        raise ToolArgumentError(f"its {which} end (at {at:.3g} m along {_axis(f.U)}) misses {what}: {where}. "
                                f"Use a smaller height or move it (placement vertical / offset)")
    us = np.linspace(u_lo, u_hi, _SAMPLES)
    between, _ = surf.probe(us)
    between = np.where(np.isnan(between), -np.inf, between)
    clear = max(0.25 * thickness, 0.001)
    outermost = float(max(w_hi, w_lo, between.max()))
    span = u_hi - u_lo

    if style == "box":
        out = outermost + reach + r
        line = np.array([[u_hi, w_hi], [u_hi, out], [u_lo, out], [u_lo, w_lo]])
        bend = max(1.01 * r * 2 ** 0.5, min(1.2 * r, (out - max(w_hi, w_lo)) / 2, span / 2))
        tips = [line[0] - np.array([0, _tip(np.array([0, -1.0]), r, wall_hi)]),
                line[-1] - np.array([0, _tip(np.array([0, -1.0]), r, wall_lo)])]
        pts2 = np.vstack([tips[0], line, tips[1]])
        pts3 = sw.round_corners(np.array([f.point(u, w) for u, w in _dedupe(pts2)]), bend)
        return sw.sweep(sw.rect_profile(thickness, thickness), pts3)

    # Round: the middle of a cubic with both inner controls at w = c sits at (w_hi + w_lo)/8 + 0.75c.
    if ear:  # out from the upper end and a little up, one full curve down: a jug's or amphora's ear
        rise_hi, rise_lo = u_lo + 1.35 * span, u_lo + 0.2 * span
    else:    # symmetric loop: a pot's lug, a drawer pull, a suitcase handle
        rise_hi, rise_lo = u_hi + 0.15 * span, u_lo - 0.15 * span
    c = (outermost + reach - (w_hi + w_lo) / 8) / 0.75
    p_hi, p_lo = np.array([u_hi, w_hi]), np.array([u_lo, w_lo])
    for _ in range(16):
        ctrl = np.array([p_hi, _leave(p_hi, np.array([rise_hi, c]), end_normals[0], r, wall_hi, span),
                         _leave(p_lo, np.array([rise_lo, c]), end_normals[1], r, wall_lo, span), p_lo])
        line = _bezier(ctrl)
        # Clearance from the body, away from the two ends where the bar goes into it.
        far_from_ends = (np.hypot(*(line - ctrl[0]).T) > 1.5 * thickness) & (np.hypot(*(line - ctrl[3]).T) > 1.5 * thickness)
        inside = (line[:, 0] > u_lo) & (line[:, 0] < u_hi) & far_from_ends
        if not inside.any():
            break
        need = np.interp(line[inside, 0], us, between) + clear + r - line[inside, 1]
        short = float(need.max())
        if short <= 1e-6:
            break
        c += short / 0.6
    else:
        raise ToolArgumentError(f"it can't clear the body of {what} between its ends; give it a bigger reach "
                                f"or move its ends")
    first = _unit2(ctrl[0] - ctrl[1])   # pointing into the object at the upper end
    last = _unit2(ctrl[3] - ctrl[2])    # and at the lower end
    pts2 = np.vstack([ctrl[0] + first * _tip(first, r, wall_hi), line, ctrl[3] + last * _tip(last, r, wall_lo)])
    pts3 = np.array([f.point(u, w) for u, w in _dedupe(pts2)])
    return sw.sweep(sw.circle_profile(r, None, 20), pts3)


def _leave(p: np.ndarray, aim: np.ndarray, normal: np.ndarray, r: float, wall: float, span: float) -> np.ndarray:
    """
    The Bezier control next to an end: toward `aim` (the shape's own control), but leaving the
    surface no further from square than the wall can hide: a bar meeting a sloped wall at an angle
    shows its end as a wedge. Allowed tilt from the surface normal: the end disc's lift, r*sin,
    stays within 0.8 of the wall (60 degrees on a solid). Kept as far out as `aim` (same w).
    """
    want = _unit2(aim - p)
    limit = np.radians(60) if not np.isfinite(wall) else float(np.arcsin(np.clip(0.8 * wall / r, 0.0, 1.0)))
    cross = normal[0] * want[1] - normal[1] * want[0]
    angle = float(np.arctan2(cross, normal @ want))
    turn = float(np.clip(angle, -limit, limit))
    c, s = np.cos(turn), np.sin(turn)
    direction = np.array([c * normal[0] - s * normal[1], s * normal[0] + c * normal[1]])
    if direction[1] <= 1e-3:
        return aim  # the surface faces away from the handle's side here: keep the shape's own control
    length = min((aim[1] - p[1]) / direction[1], 3 * span)
    return p + direction * max(length, 0.0)


def _unit2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _dedupe(pts: np.ndarray) -> np.ndarray:
    keep = [0] + [i for i in range(1, len(pts)) if np.linalg.norm(pts[i] - pts[i - 1]) > 1e-9]
    return pts[keep]


def _axis(v: np.ndarray) -> str:
    return "xyz"[int(np.argmax(np.abs(v)))]


# ---------------------------------------------------------------------------
# placement words -> ends
# ---------------------------------------------------------------------------

def defaults(span: float | None, thickness: float | None, reach: float | None, extent: float,
             orientation: str, side: str) -> tuple[float, float, float]:
    """Sizes from the object when not given: span half its extent along the handle, bar and reach in proportion."""
    h = span if span else 0.5 * extent
    t = thickness if thickness else float(np.clip(0.09 * h, 0.005, 0.04))
    if reach:
        rch = reach
    elif side == "top":
        rch = max(2.5 * t, 0.35 * h)
    else:
        rch = max(2 * t, (0.3 if orientation == "vertical" else 0.25) * h)
    return h, t, rch


def ends(bounds: np.ndarray, f: Frame, span: float, side: str, orientation: str, vertical: str,
         offset_u: float) -> tuple[float, float, float]:
    """
    (u_hi, u_lo, level) for a handle of `span` on an object with `bounds`. An upright handle sits
    at the middle of the object's height, near its top ("top", a jug's from the rim) or near its
    bottom; a sideways or top handle is centred along its face and set at that height (for a
    sideways one) by the same word. `level` is where its plane is along K when that is the height.
    """
    lo, hi = bounds
    a, b = float(lo @ f.U), float(hi @ f.U)
    if a > b:
        a, b = b, a
    if side != "top" and orientation == "vertical":
        size = b - a
        centre = {"top": b - 0.08 * size - span / 2, "bottom": a + 0.08 * size + span / 2}.get(vertical, (a + b) / 2)
        level = f.k0
    else:
        centre = (a + b) / 2
        level = f.k0
        if side != "top":  # sideways: the plane is a height; vertical picks it
            y0, y1 = float(lo[1]), float(hi[1])
            level = {"top": y1 - 0.15 * (y1 - y0), "bottom": y0 + 0.15 * (y1 - y0)}.get(vertical, (y0 + y1) / 2)
    centre += offset_u
    return centre + span / 2, centre - span / 2, level
