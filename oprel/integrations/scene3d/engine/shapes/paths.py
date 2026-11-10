"""
Row 5: shapes along a path (helpers in engine/sweep.py). Ready-made parts first, so a small
model writes {"shape": "elbow"} instead of path points; `sweep` is there for everything else.

Conventions: pipes, reducers and beams run along X by default (axis "y" = upright, "z" = front
to back), from their "start" end at -X to their "end" at +X. Ends are recorded as ports so the
next part can be joined to them exactly (placement side "end").
"""

from __future__ import annotations

import math
from typing import Any

import manifold3d as m3
import numpy as np
import trimesh
from shapely.affinity import scale
from shapely.geometry import Polygon

from oprel.integrations.scene3d.engine import sweep as sw
from oprel.integrations.scene3d.engine.cuts import to_trimesh
from oprel.integrations.scene3d.engine.outline import POINTS, explain, pairs
from oprel.integrations.scene3d.engine.registry import ToolArgumentError, integer, length, shape

_SEG = 32
_AXIS = {"type": "string", "enum": ["x", "y", "z"], "default": "x"}
# Turn something built along +X so it runs along the given axis.
_ALONG = {"x": np.eye(4),
          "y": trimesh.transformations.rotation_matrix(math.pi / 2, [0, 0, 1]),
          "z": trimesh.transformations.rotation_matrix(-math.pi / 2, [0, 1, 0])}


def _need(cond: bool, message: str) -> None:
    if not cond:
        raise ToolArgumentError(message)


def _wall(p: dict[str, Any], radius: float) -> float:
    """Default wall: about an eighth of the radius (a typical pipe), at least 0.5 mm."""
    return p.get("wall") or max(0.0005, round(0.12 * radius, 4))


def _straight(profile, length_: float, axis: str) -> trimesh.Trimesh:
    pts = np.array([[-length_ / 2, 0.0, 0.0], [length_ / 2, 0.0, 0.0]])
    mesh = sw.sweep(profile, pts)
    mesh.apply_transform(_ALONG[axis])
    return sw.with_ports(mesh, sw.transform_ports(sw.end_ports(pts), _ALONG[axis]))


@shape("pipe", "Along X (axis y = upright).",
       {"length": length("Length"), "radius": length("Outer radius"), "wall": length("Wall thickness"),
        "axis": _AXIS},
       required=("length", "radius"))
def pipe(p: dict[str, Any]) -> trimesh.Trimesh:
    r = p["radius"]
    return _straight(sw.circle_profile(r, _wall(p, r), _SEG), p["length"], p["axis"])


@shape("elbow", "Bend: in along +X, turns up.",
       {"radius": length("Outer radius"), "wall": length("Wall thickness"),
        "angle": {"type": "number", "minimum": 1, "maximum": 180, "default": 90},
        "bend_radius": length("Radius of the bend's centre line (default 3 x radius)")},
       required=("radius",))
def elbow(p: dict[str, Any]) -> trimesh.Trimesh:
    r = p["radius"]
    bend = p.get("bend_radius") or 3 * r
    _need(bend > r, f"elbow 'bend_radius' must be more than its radius {r:g} m (it would fold into itself); "
                    f"use {round(1.5 * 2 * r, 4):g} for a standard bend")
    pts, ends = sw.arc_path(bend, p["angle"])
    return sw.with_ports(sw.sweep(sw.circle_profile(r, _wall(p, r), _SEG), pts, ends), sw.end_ports(pts, ends))


def _cyl_x(r: float, length_: float, x0: float) -> m3.Manifold:
    """Cylinder along +X from x0."""
    return m3.Manifold.cylinder(length_, r, -1, _SEG).rotate([0, 90, 0]).translate([x0, 0, 0])


def _cyl_y(r: float, length_: float, y0: float) -> m3.Manifold:
    """Cylinder along +Y from y0."""
    return m3.Manifold.cylinder(length_, r, -1, _SEG).rotate([-90, 0, 0]).translate([0, y0, 0])


@shape("tee", "Run along X, branch up.",
       {"radius": length("Outer radius"), "wall": length("Wall thickness"), "length": length("Run length"),
        "branch_length": length("Branch end's distance from the run's centre line"),
        "branch_radius": length("Branch outer radius (default = radius)")},
       required=("radius",))
def tee(p: dict[str, Any]) -> trimesh.Trimesh:
    r = p["radius"]
    rb = p.get("branch_radius") or r
    w = _wall(p, min(r, rb))
    run = p.get("length") or 6 * r
    reach = p.get("branch_length") or 3 * r
    _need(rb <= r, f"tee 'branch_radius' ({rb:g}) can't be more than its radius ({r:g}); use a reducer after it")
    _need(reach > r, f"tee 'branch_length' must be more than the radius {r:g} m; use {round(3 * r, 4):g}")
    _need(run > 2 * rb, f"tee 'length' must be more than the branch's width {2 * rb:g} m; use {round(6 * r, 4):g}")
    _need(w < rb, f"the wall ({w:g} m) must be thinner than the branch radius ({rb:g} m)")
    e = 0.001 * r
    outer = _cyl_x(r, run, -run / 2) + _cyl_y(rb, reach, 0)
    bore = _cyl_x(r - w, run + 2 * e, -run / 2 - e) + _cyl_y(rb - w, reach + e, 0)
    mesh = to_trimesh(outer - bore)
    return sw.with_ports(mesh, {"start": [[-run / 2, 0, 0], [-1, 0, 0]], "end": [[run / 2, 0, 0], [1, 0, 0]],
                                "branch": [[0, reach, 0], [0, 1, 0]]})


@shape("reducer", "",
       {"radius": length("Outer radius at the start"), "end_radius": length("Outer radius at the end"),
        "length": length("Length"), "wall": length("Wall thickness"), "axis": _AXIS},
       required=("radius", "end_radius"))
def reducer(p: dict[str, Any]) -> trimesh.Trimesh:
    r1, r2 = p["radius"], p["end_radius"]
    run = p.get("length") or 2.5 * max(r1, r2)
    w = _wall(p, min(r1, r2))
    _need(w < min(r1, r2), f"the wall ({w:g} m) must be thinner than the smaller radius ({min(r1, r2):g} m)")
    a, b = 0.2 * run, 0.8 * run
    ring = [(r1 - w, 0), (r1, 0), (r1, a), (r2, b), (r2, run), (r2 - w, run), (r2 - w, b), (r1 - w, a)]
    # Manifold revolves (x, y) around Y into a solid along Z; turn Z to +X with the start at -X.
    solid = m3.Manifold.revolve(m3.CrossSection([ring]), _SEG).rotate([0, 90, 0]).translate([-run / 2, 0, 0])
    mesh = to_trimesh(solid)
    lo, hi = mesh.bounds
    mesh.apply_translation([-(lo[0] + hi[0]) / 2, 0, 0])
    mesh.apply_transform(_ALONG[p["axis"]])
    ports = {"start": [[-run / 2, 0, 0], [-1, 0, 0]], "end": [[run / 2, 0, 0], [1, 0, 0]]}
    return sw.with_ports(mesh, sw.transform_ports(ports, _ALONG[p["axis"]]))


_SECTIONS = ["i", "channel", "angle", "tube", "bar"]


@shape("beam", "Along X; section i|channel|angle|tube|bar.",
       {"length": length("Length"), "section": {"type": "string", "enum": _SECTIONS, "default": "i"},
        "width": length("Section width (flange)", 0.1), "height": length("Section height", 0.2),
        "thickness": length("Plate thickness (default height / 20)"), "axis": _AXIS},
       required=("length",))
def beam(p: dict[str, Any]) -> trimesh.Trimesh:
    w, h = p["width"], p["height"]
    t = p.get("thickness") or max(0.001, round(min(w, h) / 10 if p["section"] != "i" else h / 20, 4))
    return _straight(sw.section_profile(p["section"], w, h, t), p["length"], p["axis"])


@shape("spring", "Upright.",
       {"radius": length("Coil radius (to the wire's centre)"), "wire_radius": length("Wire radius"),
        "height": length("Height"), "coils": {"type": "number", "minimum": 0.5, "maximum": 200, "default": 8}},
       required=("radius",))
def spring(p: dict[str, Any]) -> trimesh.Trimesh:
    r, n = p["radius"], p["coils"]
    wr = p.get("wire_radius") or max(0.0005, round(0.1 * r, 4))
    h = p.get("height") or 2 * r * max(1.0, n / 4)
    _need(wr < r, f"spring 'wire_radius' ({wr:g}) must be smaller than its radius ({r:g}); use {round(0.1 * r, 4):g}")
    rise = h - 2 * wr
    pitch = rise / n if n else 0
    if pitch <= 2 * wr * 1.05:
        most = max(0.5, math.floor(rise / (2.1 * wr) * 2) / 2)
        raise ToolArgumentError(f"the coils would touch: {n:g} coils of {2 * wr:g} m wire need more than "
                                f"{round(2.1 * wr * n + 2 * wr, 4):g} m height; use coils {most:g} or less, or a "
                                f"bigger height")
    pts = sw.helix_path(r, rise, n)
    pts[:, 1] += wr
    return sw.sweep(sw.circle_profile(wr, None, 16), pts)


@shape("handle", "Built onto the object it is placed beside (side, vertical top|center|bottom), sized to it "
                 "unless given; style round|box; orientation vertical (mug, vase) | horizontal (pot, drawer).",
       {"height": length("Distance between its two ends"), "reach": length("How far it sticks out past the surface"),
        "thickness": length("Bar thickness"),
        "style": {"type": "string", "description": "round (smooth) | box", "default": "round"},
        "orientation": {"type": "string", "description": "vertical | horizontal", "default": "vertical"}})
def handle(p: dict[str, Any]) -> trimesh.Trimesh:
    # Standing alone (not placed beside anything): a mug-sized grip. Placed beside an object, the
    # compiler builds it onto that object's surface instead (engine/attach.py).
    h, reach, t = p.get("height") or 0.07, p.get("reach") or 0.03, p.get("thickness") or 0.012
    p = {**p, "style": sw.handle_style(p["style"])}
    _need(reach > t, f"handle 'reach' must be more than its thickness {t:g} m (there'd be no room for fingers)")
    _need(h > 2 * t, f"handle 'height' must be more than twice its thickness ({2 * t:g} m)")
    if p["style"] == "box":
        _need(reach >= 2 * t, f"a box handle's 'reach' must be at least twice its thickness; use {round(2.5 * t, 4):g}")
    return sw.handle_mesh(h, reach, t, p["style"])


_PROFILES = ["circle", "rectangle", "outline", *_SECTIONS[:4]]


@shape("sweep", "profile along path [[x,y,z],...] or {\"helix\": {radius, pitch, turns}} (threads, coils; "
                "outline u = away from the axis, v = along it); circle: radius, others: width + height; "
                "wall = hollow.",
       # A list of points, or {"helix": {...}}: not typed, the builder reads both.
       {"path": {"description": "[[x, y, z], ...] or {\"helix\": {\"radius\", \"pitch\", \"turns\"}}"},
        # Loose on purpose: models send "wall": true or the outline points as the profile (Qwen3-8B).
        "profile": {"type": ["string", "array"], "default": "circle"},
        "radius": length("circle radius"), "width": length("Profile width"), "height": length("Profile height"),
        "wall": {"type": ["number", "boolean"], "description": "Wall thickness (hollow)"},
        "thickness": length("Rod diameter (circle) or plate thickness (i/channel/angle)"),
        "outline": POINTS, "bend_radius": {"type": "number", "minimum": 0, "maximum": 1000}},
       required=("path",))
def sweep(p: dict[str, Any]) -> trimesh.Trimesh:
    p = dict(p)
    if isinstance(p["profile"], list):
        # Points as the profile: an outline; fewer than 3 (Qwen3-8B sent [[0, 0], [0, 0.01]] with a
        # radius for a round rod) can't be one, so the rod is round.
        if len(p["profile"]) >= 3:
            p["outline"], p["profile"] = p["profile"], "outline"
        else:
            p["profile"] = "circle"
    kind = str(p["profile"]).lower()
    # "a 1 cm thick rod": thickness is its diameter (seen with Qwen3-8B).
    if kind == "circle" and p.get("radius") is None and p.get("thickness"):
        p["radius"] = p["thickness"] / 2
    _need(kind in _PROFILES, f"sweep 'profile' must be one of {', '.join(_PROFILES)}")
    wall = p.get("wall")
    if wall is True:  # hollow, default wall
        size = p.get("radius") or min(p.get("width") or 1, p.get("height") or 1) / 2
        wall = max(0.0005, round(0.12 * size, 4))
    elif wall is False or wall is None:
        wall = None
    else:
        _need(isinstance(wall, (int, float)) and wall > 0, "sweep 'wall' must be a thickness in metres")
    p["wall"] = wall
    if kind == "circle":
        _need(p.get("radius") is not None, "sweep with a circle profile needs 'radius'")
        prof = sw.circle_profile(p["radius"], p.get("wall"), _SEG)
    elif kind == "outline":
        _need(p.get("outline") is not None, "sweep with profile 'outline' needs 'outline': [[u, v], ...]")
        prof = Polygon(pairs(p["outline"], "outline", 3))
        explain(prof, "The sweep outline", "Give the corners in order around the edge, without crossing lines.")
    else:
        _need(p.get("width") is not None and p.get("height") is not None,
              f"sweep with a {kind} profile needs 'width' and 'height'")
        if kind == "rectangle":
            prof = sw.rect_profile(p["width"], p["height"], p.get("wall"))
        else:
            t = p.get("thickness") or p.get("wall") or max(0.001, round(min(p["width"], p["height"]) / 10, 4))
            prof = sw.section_profile(kind, p["width"], p["height"], t)
    reach = sw.profile_size(prof)
    if isinstance(p["path"], dict):
        return _helix_sweep(prof, p["path"])
    pts = sw.path_points(p["path"])
    bend = p.get("bend_radius")
    if bend is None:
        bend = 1.5 * reach
    elif 0 < bend <= reach:
        raise ToolArgumentError(f"bend_radius {bend:g} m is tighter than the profile ({reach:g} m from the path), "
                                f"so the corners would fold into themselves; use {round(1.5 * reach, 4):g} or more")
    elif bend == 0 and len(pts) > 2:
        raise ToolArgumentError(f"corners need a bend_radius above {reach:g} m; use {round(1.5 * reach, 4):g}")
    pts = sw.round_corners(pts, bend)
    return sw.with_ports(sw.sweep(prof, pts), sw.end_ports(pts))


def _helix_sweep(prof: Polygon, spec: dict[str, Any]) -> trimesh.Trimesh:
    """
    A profile around a helix about Y: a screw thread (a triangle outline), a coil of flat wire, a
    spiral rail. The outline keeps u pointing away from the axis and v along it.
    """
    h = spec.get("helix", spec)
    try:
        radius = float(h["radius"])
        turns = float(h.get("turns") or h.get("coils") or 0)
        pitch = float(h.get("pitch") or 0) or (float(h["height"]) / turns if turns and h.get("height") else 0)
        if not turns and pitch and h.get("height"):
            turns = float(h["height"]) / pitch
    except (KeyError, TypeError, ValueError) as exc:
        raise ToolArgumentError('a helix path needs {"helix": {"radius": r, "pitch": p, "turns": n}} '
                                '(or "height" instead of one of pitch / turns)') from exc
    _need(radius > 0 and pitch > 0 and 0 < turns <= 500,
          'a helix path needs a radius, a pitch (rise per turn) and a number of turns, all above 0')
    lo_u, lo_v, hi_u, hi_v = prof.bounds
    _need(hi_v - lo_v < pitch, f"the outline is {hi_v - lo_v:.3g} m tall along the axis but the turns are {pitch:.3g} m "
                               f"apart, so they would overlap; make it shorter than {pitch:.3g} or the pitch bigger")
    _need(radius + lo_u > 0, f"the outline reaches {-lo_u:.3g} m in from the path, past the axis (radius {radius:g})")
    pts = sw.helix_path(radius, pitch * turns, turns, per_coil=48)
    if sw.helix_outward_sign(pts) < 0:
        prof = scale(prof, xfact=-1, yfact=1, origin=(0, 0))
    return sw.sweep(prof, pts, up_axis=np.array([0.0, 1.0, 0.0]))


# Parts with open ends another part can be joined to (placement side "start" / "end" / "branch").
PORTED = frozenset({"pipe", "elbow", "tee", "reducer", "beam", "sweep"})
# Parts placed beside an object are built onto its surface (engine/attach.py).
ATTACHED = frozenset({"handle"})
