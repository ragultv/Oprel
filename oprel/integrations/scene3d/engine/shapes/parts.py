"""
Row 6: standard parts computed from their standard (engine/standards.py). The model names the
part and its size; every other dimension comes from the table or the gear formulas.

Conventions: bolts stand upright, head up, and record a "seat" (the face under the head, or
the top of a countersunk head) so a bolt placed on top of a part sinks into it with its head
resting on it. Gears record their "axis" (centre and direction) so a gear placed beside another
is put at the exact centre distance and turned so the teeth interlock (compiler). Flanges run
along X like pipes and have start / end ports to join pipe runs.
"""

from __future__ import annotations

import math
from typing import Any

import manifold3d as m3
import numpy as np
import trimesh

from oprel.integrations.scene3d.engine import standards
from oprel.integrations.scene3d.engine import sweep as sw
from oprel.integrations.scene3d.engine.cuts import _manifold, to_trimesh
from oprel.integrations.scene3d.engine.registry import ToolArgumentError, integer, length, shape

_MM = 0.001
_Z_TO_Y = [-90, 0, 0]  # manifold builds along Z; parts stand along Y
_Z_TO_X = [0, 90, 0]
_SIZE = {"type": ["string", "number"], "description": "ISO metric size, e.g. \"M10\""}
_TRUE_ROOT = 0.6134  # external thread depth / pitch (ISO 68-1: 17H/24)
_INNER_DEPTH = 0.5413  # internal thread depth / pitch (5H/8)


def _need(cond: bool, message: str) -> None:
    if not cond:
        raise ToolArgumentError(message)


def _mm_or_m(value: float, what: str, notes: list[str], biggest_m: float = 1.0) -> float:
    """A length the model gave in millimetres ("length": 50 for an M10 bolt): read as mm, say so."""
    if value > biggest_m:
        notes.append(f"read {what} {value:g} as {value:g} mm")
        return value * _MM
    return value


# ---------------------------------------------------------------------------
# threads
# ---------------------------------------------------------------------------

def thread_mesh(radius: float, pitch: float, height: float, depth: float, *, tip_chamfer: bool,
                segments: int = 32, per_pitch: int = 8) -> trimesh.Trimesh:
    """
    A threaded rod along Z from 0 to `height`, outer radius `radius`: a cylinder grid whose radius
    follows the ISO profile at each point's phase along the helix (crest flat p/8, root flat p/4,
    straight flanks). Closed and watertight with no booleans; `tip_chamfer` bevels the z=0 end at
    45 degrees from the root, as on a bolt's point.
    """
    rows = max(2, math.ceil(height / pitch * per_pitch - 1e-9))
    z = np.linspace(0.0, height, rows + 1)
    theta = np.linspace(0.0, 2 * math.pi, segments, endpoint=False)
    phase = np.mod(z[:, None] / pitch - theta[None, :] / (2 * math.pi), 1.0)
    # distance from the crest's centre, 0 .. 0.5 of a pitch
    off = np.minimum(phase, 1.0 - phase)
    crest, root = 1 / 16, 0.5 - 1 / 8
    sink = np.clip((off - crest) / (root - crest), 0.0, 1.0)
    r = radius - depth * sink
    if tip_chamfer:
        r = np.minimum(r, radius - depth + z[:, None])
    verts = np.column_stack([(r * np.cos(theta)).ravel(), (r * np.sin(theta)).ravel(),
                             np.repeat(z, segments)])
    n = segments
    i = np.arange(n)
    faces = []
    for j in range(rows):
        a, b = j * n + i, j * n + (i + 1) % n
        c, d = b + n, a + n
        faces.append(np.column_stack([a, b, c]))
        faces.append(np.column_stack([a, c, d]))
    bottom, top = len(verts), len(verts) + 1
    verts = np.vstack([verts, [0, 0, 0], [0, 0, height]])
    faces.append(np.column_stack([np.full(n, bottom), (i + 1) % n, i]))
    faces.append(np.column_stack([np.full(n, top), rows * n + i, rows * n + (i + 1) % n]))
    mesh = trimesh.Trimesh(verts, np.vstack(faces), process=False)
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def _hex(across_flats: float, height: float, z0: float = 0.0) -> m3.Manifold:
    return m3.Manifold.cylinder(height, across_flats / math.sqrt(3), -1, 6).translate([0, 0, z0])


def _chamfered(body: m3.Manifold, across_flats: float, z0: float, z1: float, both: bool) -> m3.Manifold:
    """The 30 degree chamfer that rounds off a hex head's / nut's corners (on top, or both faces)."""
    a = 0.475 * across_flats
    big = across_flats / math.sqrt(3) * 1.05
    drop = (big - a) * math.tan(math.radians(30))
    lo = [(a, z0), (big, z0 + drop)] if both else [(big, z0 - 1e-6)]
    pts = [(0, z0 - 1e-6)] + lo + [(big, z1 - drop), (a, z1), (0, z1)]
    return body ^ m3.Manifold.revolve(m3.CrossSection([pts]), 48)


def _union(parts: list[m3.Manifold]) -> m3.Manifold:
    return m3.Manifold.batch_boolean(parts, m3.OpType.Add)


# ---------------------------------------------------------------------------
# bolt, nut, washer
# ---------------------------------------------------------------------------

_HEADS = ("hex", "socket", "countersunk")


@shape("bolt", "ISO; head hex|socket|countersunk. Placed on top of a part, its head rests on it.",
       {"size": _SIZE, "length": length("Length under the head (countersunk: overall)"),
        "head": {"type": "string", "enum": list(_HEADS), "default": "hex"},
        "thread": {"type": "boolean", "default": True}},
       required=("size", "length"))
def bolt(p: dict[str, Any]) -> trimesh.Trimesh:
    row, pitch = standards.metric(p["size"])
    notes: list[str] = []
    d, P = row.d * _MM, pitch * _MM
    L = _mm_or_m(p["length"], "length", notes, biggest_m=0.6)
    _need(L >= 2 * d if p["head"] != "countersunk" else L >= 2.5 * d,
          f"a {row.name} bolt must be at least {2.5 * row.d if p['head'] == 'countersunk' else 2 * row.d:g} mm long")
    if p["head"] == "hex":
        k = row.k * _MM
        head = _chamfered(_hex(row.s * _MM, k, L), row.s * _MM, L, L + k, both=False)
        seat = L
    elif p["head"] == "socket":
        dk, key, k = row.socket[0] * _MM, row.socket[1] * _MM, d
        head = m3.Manifold.cylinder(k, dk / 2, -1, 48).translate([0, 0, L]) - _hex(key, 0.6 * k + 1e-4, L + 0.4 * k)
        seat = L
    else:  # countersunk: 90 degree head, flush with the surface; L is overall
        dk = 2 * d
        k = (dk - d) / 2
        head = m3.Manifold.cylinder(k, d / 2, dk / 2, 48).translate([0, 0, L - k])
        key = row.socket[1] * _MM * 0.8
        head = head - _hex(key, 0.5 * k, L - 0.5 * k + 1e-4)
        seat = L
    plain_end = L - (k if p["head"] == "countersunk" else 0)
    threaded = min(plain_end, standards.thread_length(row.d, plain_end / _MM) * _MM)
    parts = [head]
    if threaded < plain_end:
        # A hair under the thread's crest and reaching into it: crest and shank points that touch
        # exactly would pinch the fused surface.
        parts.append(m3.Manifold.cylinder(plain_end - threaded + P / 2, 0.995 * d / 2, -1, 48)
                     .translate([0, 0, threaded - P / 2]))
    if p["thread"]:
        parts.append(_manifold(thread_mesh(d / 2, P, threaded, _TRUE_ROOT * P, tip_chamfer=True), "the thread"))
    else:
        tip = m3.Manifold.cylinder(min(P, threaded), d / 2 - _TRUE_ROOT * P, d / 2, 48)
        parts += [tip, m3.Manifold.cylinder(threaded - P, d / 2, -1, 48).translate([0, 0, P])]
    mesh = to_trimesh(_union(parts).rotate(_Z_TO_Y))
    mesh.metadata["ports"] = {"seat": [[0, seat, 0], [0, 1, 0]], "axis": [[0, seat, 0], [0, -1, 0]]}
    mesh.metadata["notes"] = notes
    mesh.metadata["standard"] = {"size": row.name, "pitch_mm": pitch, "clearance_mm": row.clearance}
    return mesh


@shape("nut", "Placed relative_to a bolt: threads onto it, under what it holds.",
       {"size": _SIZE, "thread": {"type": "boolean", "default": True}},
       required=("size",))
def nut(p: dict[str, Any]) -> trimesh.Trimesh:
    row, pitch = standards.metric(p["size"])
    d, P, m, s = row.d * _MM, pitch * _MM, row.m * _MM, row.s * _MM
    body = _chamfered(_hex(s, m), s, 0, m, both=True)
    e = 0.02 * m
    if p["thread"]:
        bore = _manifold(thread_mesh(d / 2, P, m + 2 * e, _INNER_DEPTH * P, tip_chamfer=False), "the thread")
        bore = bore.translate([0, 0, -e])
    else:
        bore = m3.Manifold.cylinder(m + 2 * e, d / 2 - _INNER_DEPTH * P, -1, 48).translate([0, 0, -e])
    mesh = to_trimesh((body - bore).rotate(_Z_TO_Y))
    mesh.metadata["ports"] = {"axis": [[0, m, 0], [0, -1, 0]]}
    return mesh


@shape("washer", "", {"size": _SIZE}, required=("size",))
def washer(p: dict[str, Any]) -> trimesh.Trimesh:
    row, _ = standards.metric(p["size"])
    d1, d2, h = (v * _MM for v in row.washer)
    ring = m3.Manifold.revolve(m3.CrossSection([[(d1 / 2, 0), (d2 / 2, 0), (d2 / 2, h), (d1 / 2, h)]]), 64)
    mesh = to_trimesh(ring.rotate(_Z_TO_Y))
    mesh.metadata["ports"] = {"axis": [[0, h, 0], [0, -1, 0]]}
    return mesh


# ---------------------------------------------------------------------------
# gears
# ---------------------------------------------------------------------------

def _inv(a: float) -> float:
    return math.tan(a) - a


def gear_outline(teeth: int, module: float, pressure_deg: float = 20.0) -> np.ndarray:
    """
    Involute spur gear outline, tooth 0 centred on +X, counter-clockwise. Pitch radius m z / 2,
    addendum m, dedendum 1.25 m, zero backlash; below the base circle the flank runs straight in.
    """
    alpha = math.radians(pressure_deg)
    r = module * teeth / 2
    rb, ra, rf = r * math.cos(alpha), r + module, r - 1.25 * module
    psi = math.pi / (2 * teeth)

    def half(rho: float) -> float:
        return psi + _inv(alpha) - _inv(math.acos(min(1.0, rb / rho)))

    while half(ra) < 0.02 * psi:  # few teeth: the involutes meet before the tip circle
        ra -= 0.02 * module
    start = max(rb, rf)
    flank = np.linspace(start, ra, 10)
    step = 2 * math.pi / teeth
    pts: list[tuple[float, float]] = []

    def at(rho: float, ang: float) -> None:
        pts.append((rho * math.cos(ang), rho * math.sin(ang)))

    for k in range(teeth):
        c = k * step
        h0 = half(start)
        if rf < rb:
            at(rf, c - h0)
        for rho in flank:
            at(rho, c - half(rho))
        tip = half(ra)
        for t in np.linspace(-tip, tip, 5)[1:-1]:
            at(ra, c + t)
        for rho in flank[::-1]:
            at(rho, c + half(rho))
        if rf < rb:
            at(rf, c + h0)
        gap = step - 2 * h0
        for t in np.linspace(0, gap, 6)[1:-1]:
            at(rf, c + h0 + t)
    return np.array(pts)


def gear_module(p: dict[str, Any], notes: list[str] | None = None) -> float:
    teeth = p["teeth"]
    if p.get("module") is not None:
        m = p["module"] * _MM if p["module"] >= 0.05 else p["module"]  # "module": 2 means 2 mm
        if p.get("radius") and notes is not None:
            notes.append(f"used module {m / _MM:g} mm (pitch radius {m * teeth / 2:.4g} m); radius was ignored")
        return m
    if p.get("radius"):
        return 2 * p["radius"] / teeth  # the pitch radius given: module from it
    return 0.002


@shape("gear", "Involute; axis z faces the front, y lies flat. Placed beside a gear, it meshes with it.",
       {"teeth": integer("Number of teeth", 6, 400), "module": {"type": "number", "minimum": 0.0001, "maximum": 100,
        "description": "Tooth size in mm (pitch diameter = module x teeth)"},
        "radius": length("Pitch radius (instead of module)"), "thickness": length("Face width (default 10 x module)"),
        "bore_radius": length("Shaft hole radius"),
        "helix_angle": {"type": "number", "minimum": -45, "maximum": 45, "default": 0},
        "pressure_angle": {"type": "number", "minimum": 14.5, "maximum": 30, "default": 20},
        "axis": {"type": "string", "enum": ["z", "y"], "default": "z"}},
       required=("teeth",))
def gear(p: dict[str, Any]) -> trimesh.Trimesh:
    notes: list[str] = []
    z = p["teeth"]
    m = gear_module(p, notes)
    r = m * z / 2
    rf = r - 1.25 * m
    t = p.get("thickness") or 10 * m
    bore = p.get("bore_radius")
    if bore is None:
        bore = max(0.0005, round(0.3 * rf * 2000) / 2000)
    _need(bore < rf - m, f"gear 'bore_radius' ({bore:g}) must leave material under the teeth: at most {round(rf - m, 4):g}")
    section = m3.CrossSection([gear_outline(z, m, p["pressure_angle"]).tolist()])
    section = section - m3.CrossSection.circle(bore, 48)
    twist = math.degrees(t * math.tan(math.radians(p["helix_angle"])) / r)
    body = m3.Manifold.extrude(section, t, max(0, int(abs(twist) / 2)), twist)
    centre, axis = [0, 0, t / 2], [0, 0, 1]
    if p["axis"] == "y":
        body = body.rotate(_Z_TO_Y)
        centre, axis = [0, t / 2, 0], [0, 1, 0]
    mesh = to_trimesh(body)
    mesh.metadata["ports"] = {"axis": [centre, axis]}
    mesh.metadata["notes"] = notes
    return mesh


# ---------------------------------------------------------------------------
# flange
# ---------------------------------------------------------------------------

@shape("flange", "PN16; join it to a pipe's end.",
       {"size": {"type": ["string", "number"], "description": "Nominal size, e.g. \"DN50\""},
        "thickness": length("Thickness"), "bore_radius": length("Bore radius (default the pipe's inside)"),
        "blind": {"type": "boolean", "default": False}},
       required=("size",))
def flange(p: dict[str, Any]) -> trimesh.Trimesh:
    row = standards.nominal(p["size"])
    D, K, hole, C = row.outer * _MM, row.bolt_circle * _MM, row.hole * _MM, (p.get("thickness") or row.thickness * _MM)
    od = row.pipe_od * _MM
    bore = p.get("bore_radius") or (od / 2 - max(0.0005, round(0.12 * od / 2, 4)))
    _need(bore < K / 2 - hole, f"flange 'bore_radius' must stay inside the bolt holes: at most {round(K / 2 - hole, 4):g}")
    body = m3.Manifold.cylinder(C, D / 2, -1, 96)
    cutters = [m3.Manifold.cylinder(C + 2e-4, hole / 2, -1, 32).translate(
        [K / 2 * math.cos((k + 0.5) * 2 * math.pi / row.holes), K / 2 * math.sin((k + 0.5) * 2 * math.pi / row.holes),
         -1e-4]) for k in range(row.holes)]
    if not p["blind"]:
        cutters.append(m3.Manifold.cylinder(C + 2e-4, bore, -1, 64).translate([0, 0, -1e-4]))
    mesh = to_trimesh((body - _union(cutters)).rotate(_Z_TO_X))
    mesh.metadata["ports"] = {"start": [[0, 0, 0], [-1, 0, 0]], "end": [[C, 0, 0], [1, 0, 0]]}
    mesh.metadata["standard"] = {"size": f"DN{row.dn}", "bolts": f"{row.holes} x {row.bolt}"}
    return sw.with_ports(mesh, mesh.metadata.pop("ports"))
