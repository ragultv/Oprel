"""
Sweeping a 2D profile along a 3D path: pipes, bends, springs, handles, beams, rails.

  profile   a shapely Polygon in (u, v): v is the profile's "up", u its side. On a level
            path v stays world-up (a beam's height is vertical); on an upright path it faces +Z.
  path      3D points; corners are rounded with a bend radius (round_corners) so the swept
            solid never folds into itself at a sharp kink.
  frames    rotation-minimising (double reflection): the profile doesn't twist along a helix.

Builders record their open ends as `ports` in mesh.metadata ({name: [point, outward dir]}),
so another part can be joined to an end exactly (compiler: placement side "end").

It registers no shapes (see shapes/paths.py).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import trimesh
from shapely.geometry import Point, Polygon, box
from shapely.geometry.polygon import orient

from oprel.integrations.scene3d.engine.registry import ToolArgumentError

UP = np.array([0.0, 1.0, 0.0])
_ARC_STEP = np.radians(7.5)  # one ring every 7.5 degrees of bend


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


# ---------------------------------------------------------------------------
# profiles
# ---------------------------------------------------------------------------

def circle_profile(radius: float, wall: float | None = None, segments: int = 32) -> Polygon:
    """A round bar, or a tube when `wall` is given."""
    outer = Point(0, 0).buffer(radius, quad_segs=max(2, segments // 4))
    if wall is None:
        return outer
    if wall >= radius:
        raise ToolArgumentError(f"the wall ({wall:g} m) must be thinner than the radius ({radius:g} m); "
                                f"for a tube use a wall of about {round(radius * 0.12, 4):g} m, or leave it out "
                                f"for a solid bar")
    return outer.difference(Point(0, 0).buffer(radius - wall, quad_segs=max(2, segments // 4)))


def rect_profile(width: float, height: float, wall: float | None = None) -> Polygon:
    outer = box(-width / 2, -height / 2, width / 2, height / 2)
    if wall is None:
        return outer
    if 2 * wall >= min(width, height):
        raise ToolArgumentError(f"the wall ({wall:g} m) must be less than half of {min(width, height):g} m")
    return outer.difference(box(-width / 2 + wall, -height / 2 + wall, width / 2 - wall, height / 2 - wall))


def section_profile(kind: str, width: float, height: float, t: float) -> Polygon:
    """Structural sections, `height` along v (up), `width` along u, `t` = plate thickness."""
    if 2 * t >= min(width, height):
        raise ToolArgumentError(f"thickness {t:g} m is too thick for a {width:g} x {height:g} m section; "
                                f"use at most {round(min(width, height) / 4, 4):g} m")
    w, h = width / 2, height / 2
    if kind == "i":
        pts = [(-w, -h), (w, -h), (w, -h + t), (t / 2, -h + t), (t / 2, h - t), (w, h - t), (w, h),
               (-w, h), (-w, h - t), (-t / 2, h - t), (-t / 2, -h + t), (-w, -h + t)]
    elif kind == "channel":  # C, open toward +u
        pts = [(-w, -h), (w, -h), (w, -h + t), (-w + t, -h + t), (-w + t, h - t), (w, h - t), (w, h), (-w, h)]
    elif kind == "angle":  # L, legs along +u and +v
        pts = [(-w, -h), (w, -h), (w, -h + t), (-w + t, -h + t), (-w + t, h), (-w, h)]
    elif kind == "tube":
        return rect_profile(width, height, t)
    else:  # bar
        return rect_profile(width, height)
    return Polygon(pts)


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------

def path_points(raw: object, minimum: int = 2) -> np.ndarray:
    try:
        pts = np.array([[float(c) for c in p] for p in raw], dtype=float)  # type: ignore[union-attr]
    except (TypeError, ValueError) as exc:
        raise ToolArgumentError("'path' must be a list of [x, y, z] points") from exc
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ToolArgumentError("'path' must be a list of [x, y, z] points (3 numbers each)")
    keep = [0] + [i for i in range(1, len(pts)) if np.linalg.norm(pts[i] - pts[i - 1]) > 1e-9]
    pts = pts[keep]
    if len(pts) < minimum:
        raise ToolArgumentError(f"'path' needs at least {minimum} different points")
    return pts


def round_corners(pts: np.ndarray, bend: float) -> np.ndarray:
    """
    Replace every inner corner with an arc of radius `bend`, tangent to both segments.
    Raises with the largest radius that fits when a segment is too short for its arcs.
    """
    if len(pts) < 3 or bend <= 0:
        return pts
    out = [pts[0]]
    used_after = 0.0  # how much of the segment leaving the previous corner its arc took
    for i in range(1, len(pts) - 1):
        a, b, c = pts[i - 1], pts[i], pts[i + 1]
        d1, d2 = _unit(b - a), _unit(c - b)
        turn = float(np.arccos(np.clip(np.dot(d1, d2), -1, 1)))
        if turn < 1e-4:
            continue  # straight on
        if turn > np.pi - 1e-3:
            raise ToolArgumentError(f"the path turns straight back on itself at point {i}; "
                                    f"add a point to the side so it can bend round")
        cut = bend * np.tan(turn / 2)
        len1, len2 = np.linalg.norm(b - a), np.linalg.norm(c - b)
        room1 = len1 - used_after
        room2 = len2 if i == len(pts) - 2 else len2 / 2
        if cut > room1 + 1e-9 or cut > room2 + 1e-9:
            fit = min(room1, room2) / np.tan(turn / 2)
            raise ToolArgumentError(f"bend_radius {bend:g} m doesn't fit at path point {i} (the segments next to it "
                                    f"are too short); use bend_radius {max(round(fit * 0.95, 4), 0.0001):g} or less, "
                                    f"or longer segments")
        p1 = b - d1 * cut
        # Arc centre: from p1 toward the inside of the turn.
        inward = _unit(d2 - d1 * np.dot(d1, d2))
        centre = p1 + inward * bend
        steps = max(2, int(np.ceil(turn / _ARC_STEP)))
        u0 = p1 - centre
        axis = _unit(np.cross(d1, d2))
        for k in range(steps + 1):
            ang = turn * k / steps
            # Rodrigues: rotate u0 about axis by ang (u0 is perpendicular to axis).
            out.append(centre + u0 * np.cos(ang) + np.cross(axis, u0) * np.sin(ang))
        used_after = cut
    out.append(pts[-1])
    res = np.array(out)
    keep = [0] + [i for i in range(1, len(res)) if np.linalg.norm(res[i] - res[i - 1]) > 1e-9]
    return res[keep]


def arc_path(radius: float, angle_deg: float) -> tuple[np.ndarray, Ends]:
    """
    Along +X from the origin, bending up (toward +Y) by `angle_deg` on `radius`. Returns the
    points and the exact tangents at both ends (for sweep / end_ports).
    """
    ang = np.radians(angle_deg)
    steps = max(2, int(np.ceil(ang / _ARC_STEP)))
    t = np.linspace(0, ang, steps + 1)
    arc = np.c_[radius * np.sin(t), radius * (1 - np.cos(t)), np.zeros_like(t)]
    return arc, ((1.0, 0.0, 0.0), (float(np.cos(ang)), float(np.sin(ang)), 0.0))


def helix_outward_sign(pts: np.ndarray) -> float:
    """+1 if the profile's u points away from the helix axis (Y) with up_axis = Y, else -1."""
    tang = _tangents(pts)
    up = np.array([0.0, 1.0, 0.0])
    v = up - tang[0] * (tang[0] @ up)
    u = np.cross(v / np.linalg.norm(v), tang[0])
    return 1.0 if u @ (pts[0] * [1, 0, 1]) > 0 else -1.0


def helix_path(radius: float, height: float, coils: float, per_coil: int = 32) -> np.ndarray:
    n = max(8, int(np.ceil(coils * per_coil)))
    t = np.linspace(0, 2 * np.pi * coils, n + 1)
    return np.c_[radius * np.cos(t), height * t / t[-1], -radius * np.sin(t)]


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------

Ends = "tuple[Sequence[float], Sequence[float]] | None"


def _tangents(pts: np.ndarray, ends: Ends = None) -> np.ndarray:
    """
    Direction of travel at each point. The ends default to their segment's direction; a path that
    starts or stops part-way round a curve (an elbow) passes its exact `ends` tangents instead.
    """
    seg = np.diff(pts, axis=0)
    seg = seg / np.linalg.norm(seg, axis=1, keepdims=True)
    tang = np.empty_like(pts)
    tang[0], tang[-1] = seg[0], seg[-1]
    if len(pts) > 2:
        tang[1:-1] = seg[:-1] + seg[1:]
        tang[1:-1] /= np.maximum(np.linalg.norm(tang[1:-1], axis=1, keepdims=True), 1e-12)
    if ends is not None:
        tang[0], tang[-1] = _unit(np.asarray(ends[0], float)), _unit(np.asarray(ends[1], float))
    return tang


def _frames(pts: np.ndarray, tang: np.ndarray) -> np.ndarray:
    """(n, 3) profile-up vectors along the path (double-reflection rotation-minimising frames)."""
    t0 = tang[0]
    ref = UP if abs(np.dot(t0, UP)) < 0.99 else np.array([0.0, 0.0, 1.0])
    v = _unit(ref - t0 * np.dot(ref, t0))
    ups = [v]
    for i in range(len(pts) - 1):
        v1 = pts[i + 1] - pts[i]
        c1 = np.dot(v1, v1)
        r_l = v - (2 / c1) * np.dot(v1, v) * v1
        t_l = tang[i] - (2 / c1) * np.dot(v1, tang[i]) * v1
        v2 = tang[i + 1] - t_l
        c2 = np.dot(v2, v2)
        v = r_l - (2 / c2) * np.dot(v2, r_l) * v2 if c2 > 1e-18 else r_l
        v = _unit(v - tang[i + 1] * np.dot(v, tang[i + 1]))
        ups.append(v)
    return np.array(ups)


def sweep(profile: Polygon, pts: np.ndarray, ends: Ends = None, up_axis: np.ndarray | None = None) -> trimesh.Trimesh:
    """
    Closed solid: `profile` carried along `pts` (already corner-rounded), capped at both ends.
    `up_axis`: keep the profile's v pointing along this axis everywhere (a helix around it: a
    thread's outline stays square to the screw instead of slowly turning with the frames).
    """
    profile = orient(profile)  # exterior counter-clockwise, holes clockwise
    tang = _tangents(pts, ends)
    if up_axis is None:
        ups = _frames(pts, tang)
    else:
        ups = up_axis - tang * (tang @ up_axis)[:, None]
        ups /= np.linalg.norm(ups, axis=1, keepdims=True)
    sides = np.cross(ups, tang)  # u axis: (u, v, t) right-handed

    def place(uv: np.ndarray, i: int) -> np.ndarray:
        return pts[i] + np.outer(uv[:, 0], sides[i]) + np.outer(uv[:, 1], ups[i])

    rings = [np.asarray(profile.exterior.coords)[:-1]] + [np.asarray(r.coords)[:-1] for r in profile.interiors]
    verts: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    offset = 0
    n = len(pts)
    for ring in rings:
        m = len(ring)
        verts.extend(place(ring, i) for i in range(n))
        idx = np.arange(n * m).reshape(n, m) + offset
        a, b = idx[:-1], np.roll(idx[:-1], -1, axis=1)
        c, d = idx[1:], np.roll(idx[1:], -1, axis=1)
        faces.append(np.c_[a.ravel(), b.ravel(), d.ravel()])
        faces.append(np.c_[a.ravel(), d.ravel(), c.ravel()])
        offset += n * m
    cap_v, cap_f = trimesh.creation.triangulate_polygon(profile, engine="manifold")
    for i, flip in ((0, True), (n - 1, False)):
        faces.append((cap_f[:, ::-1] if flip else cap_f) + offset)
        verts.append(place(np.asarray(cap_v), i))
        offset += len(cap_v)
    mesh = trimesh.Trimesh(np.vstack(verts), np.vstack(faces), process=False)
    mesh.merge_vertices(digits_vertex=9)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    if mesh.volume < 0:
        mesh.invert()
    return mesh


_BOX_WORDS = {"box", "square", "squared", "angular", "rectangular", "flat"}


def handle_style(value: object) -> str:
    """
    "box" or "round" from what a model writes: Qwen3-8B sent "smooth" and was refused three times
    for a vase. Anything not square-ish is the smooth one.
    """
    return "box" if str(value or "").strip().lower() in _BOX_WORDS else "round"


def handle_mesh(height: float, reach: float, thickness: float, style: str = "round") -> trimesh.Trimesh:
    """
    Grip in the XY plane: its two ends at x = 0 (y = thickness / 2 and height - that), reaching out
    to x = reach. Used by the handle shape and the vessels' own handles.
      round  a smooth ear, like a jug's or an amphora's: out from the upper end and a little up,
             then one full curve down and back in (one cubic Bezier, so no kinks); its tips carry on
             along the curve a bar's radius into the wall, so the joins are clean
      box    square bar with square corners (slightly rounded so the sweep stays solid)
    """
    r = thickness / 2
    if style == "box":
        corners = np.array([[0.0, r, 0.0], [reach - r, r, 0.0], [reach - r, height - r, 0.0], [0.0, height - r, 0.0]])
        bend = min(1.2 * r, (reach - r) / 2, (height - 2 * r) / 2)
        return sweep(rect_profile(thickness, thickness), round_corners(corners, max(bend, 1.01 * r * 2 ** 0.5)))
    return ear((0.0, height - r), (0.0, r), reach - r, thickness)


def ear(upper: tuple[float, float], lower: tuple[float, float], reach: float, thickness: float,
        wall: float | None = None) -> trimesh.Trimesh:
    """
    Smooth ear handle (XY plane) from the `upper` attachment point to the `lower` one, reaching
    `reach` past the further-out of the two: out from the upper point and a little up, then one
    full curve down and back in (one cubic Bezier, so no kinks). Its tips carry on along the curve
    a bar's radius into the wall, or, given the `wall` thickness (the points then on its outside),
    only as deep as 0.8 of it, so they don't show inside a thin cup. Built between the two points,
    not tilted to fit them: tilting swung the lower half into a vase's sloping shoulder.
    """
    r = thickness / 2
    (xa, ya), (xb, yb) = upper, lower
    span = ya - yb
    # x of the curve at its middle is (xa + xb) / 8 + 0.75 * out: put the peak `reach` past both ends.
    out = (max(xa, xb) + reach - (xa + xb) / 8) / 0.75
    ctrl = np.array([[xa, ya], [out, yb + 1.35 * span], [out, yb + 0.2 * span], [xb, yb]])
    t = np.linspace(0, 1, 64)[:, None]
    line = ((1 - t) ** 3 * ctrl[0] + 3 * (1 - t) ** 2 * t * ctrl[1] + 3 * (1 - t) * t ** 2 * ctrl[2]
            + t ** 3 * ctrl[3])
    first, last = _unit(ctrl[1] - ctrl[0]), _unit(ctrl[3] - ctrl[2])
    def tip(d: np.ndarray) -> float:
        dx = max(abs(d[0]), 1e-6)
        if wall is None:
            return min(2 * r, r / dx)  # reach x = -r, at most 2r long
        # Deepest point of the bar's end: along the tip, plus its end disc tilted with it.
        return max(0.0, min(2 * r, (0.8 * wall - r * np.sqrt(max(0.0, 1 - dx * dx))) / dx))

    line = np.vstack([ctrl[0] - first * tip(first), line, ctrl[3] + last * tip(last)])
    keep = [0] + [i for i in range(1, len(line)) if np.linalg.norm(line[i] - line[i - 1]) > 1e-9]
    pts = np.c_[line[keep], np.zeros(len(keep))]
    return sweep(circle_profile(r, None, 20), pts)


def end_ports(pts: np.ndarray, ends: Ends = None) -> dict[str, list[list[float]]]:
    """The two open ends of a swept path: where they are and which way they face (outward)."""
    tang = _tangents(pts, ends)
    return {"start": [pts[0].tolist(), (-tang[0]).tolist()], "end": [pts[-1].tolist(), tang[-1].tolist()]}


def with_ports(mesh: trimesh.Trimesh, ports: dict[str, list[list[float]]]) -> trimesh.Trimesh:
    mesh.metadata["ports"] = ports
    return mesh


def transform_ports(ports: dict[str, list[list[float]]], m: np.ndarray) -> dict[str, list[list[float]]]:
    out = {}
    for name, (p, d) in ports.items():
        p2 = m[:3, :3] @ np.asarray(p) + m[:3, 3]
        d2 = _unit(m[:3, :3] @ np.asarray(d))
        out[name] = [p2.tolist(), d2.tolist()]
    return out


def profile_size(profile: Polygon) -> float:
    """Half the profile's largest extent: how far it reaches from the path."""
    lo_u, lo_v, hi_u, hi_v = profile.bounds
    return float(max(abs(lo_u), abs(lo_v), abs(hi_u), abs(hi_v)))


__all__: Sequence[str] = (
    "arc_path", "circle_profile", "end_ports", "handle_mesh", "handle_style", "helix_path", "path_points", "profile_size", "rect_profile",
    "round_corners", "section_profile", "sweep", "transform_ports", "with_ports",
)
