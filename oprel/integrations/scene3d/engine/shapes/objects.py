"""
Ready-made objects: common things built properly from parts, so a model writes one line
({"shape": "chair"}) instead of assembling (and mis-assembling) seats, legs and backs.

Conventions: sizes in metres, bottom-centre at the origin, the object's FRONT faces +Z
(where you sit or stand). Patterns with face_center / "around" turn the front to the middle.
Each object is one mesh, so it has one colour.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import trimesh

from oprel.integrations.scene3d.engine.frame import Z_UP_TO_Y_UP
from oprel.integrations.scene3d.engine.registry import ToolArgumentError, integer, length, shape
from oprel.integrations.scene3d.engine import attach, sweep
from oprel.integrations.scene3d.engine.outline import catmull_rom, revolve_profile, vessel_profile

_SEG = 24


def _box(w: float, h: float, d: float, x: float = 0, y: float = 0, z: float = 0) -> trimesh.Trimesh:
    """Box with its bottom-centre at (x, y, z)."""
    m = trimesh.creation.box(extents=[w, h, d])
    m.apply_translation([x, y + h / 2, z])
    return m


def _cyl(r: float, h: float, x: float = 0, y: float = 0, z: float = 0, r_top: float | None = None) -> trimesh.Trimesh:
    """Upright cylinder (or frustum with r_top) with its bottom-centre at (x, y, z)."""
    if r_top is None or abs(r_top - r) < 1e-9:
        m = trimesh.creation.cylinder(radius=r, height=h, sections=_SEG)
        m.apply_transform(Z_UP_TO_Y_UP)
        m.apply_translation([x, y + h / 2, z])
        return m
    m = trimesh.creation.revolve([[0, 0], [r, 0], [r_top, h], [0, h]], sections=_SEG)
    m.merge_vertices()
    if m.volume < 0:
        m.invert()
    m.apply_transform(Z_UP_TO_Y_UP)
    m.apply_translation([x, y, z])
    return m


def _join(parts: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    return trimesh.util.concatenate(parts)


def _need(cond: bool, message: str) -> None:
    if not cond:
        raise ToolArgumentError(message)


@shape("table", "top: rectangle|round (round: width = diameter); legs: 1 = pedestal.",
       {"width": length("Size along X (diameter if round)", 1.2), "depth": length("Size along Z", 0.7),
        "height": length("Height of the top surface", 0.75),
        "top": {"type": "string", "enum": ["rectangle", "round"], "default": "rectangle"},
        "legs": integer("Number of legs", 1, 12, 4), "top_thickness": length("Top thickness", 0.03),
        "leg_radius": length("Leg radius", 0.025)})
def table(p: dict[str, Any]) -> trimesh.Trimesh:
    w, d, h, t, lr, n = p["width"], p["depth"], p["height"], p["top_thickness"], p["leg_radius"], p["legs"]
    _need(h > t, "table 'height' must be more than 'top_thickness'")
    leg_h = h - t
    round_top = p["top"] == "round"
    parts = [_cyl(w / 2, t, y=leg_h) if round_top else _box(w, t, d, y=leg_h)]
    half = min(w, w if round_top else d) / 2
    if n == 1:  # pedestal: column and a round foot
        parts += [_cyl(max(lr * 2.5, 0.05), leg_h), _cyl(half * 0.6, 0.03)]
    elif round_top:
        r = half - max(0.05, 0.12 * half) - lr
        parts += [_cyl(lr, leg_h, r * math.sin(2 * math.pi * i / n), 0, r * math.cos(2 * math.pi * i / n))
                  for i in range(n)]
    else:
        inset = max(0.04, 0.06 * min(w, d)) + lr
        x0, z0 = w / 2 - inset, d / 2 - inset
        per_side = max(1, n // 2)
        xs = [0.0] if per_side == 1 else [-x0 + 2 * x0 * i / (per_side - 1) for i in range(per_side)]
        parts += [_cyl(lr, leg_h, x, 0, z) for x in xs for z in (-z0, z0)]
        if n % 2:
            parts.append(_cyl(lr, leg_h))
    return _join(parts)


@shape("chair", "Seat, 4 legs, backrest.",
       {"width": length("Seat width", 0.45), "depth": length("Seat depth", 0.45),
        "seat_height": length("Seat surface height", 0.45), "back_height": length("Backrest height above the seat", 0.45),
        "arms": {"type": "boolean", "default": False}, "leg_radius": length("Leg radius", 0.018)})
def chair(p: dict[str, Any]) -> trimesh.Trimesh:
    w, d, sh, bh, lr = p["width"], p["depth"], p["seat_height"], p["back_height"], p["leg_radius"]
    seat_t, back_t = 0.04, 0.03
    _need(sh > seat_t, "chair 'seat_height' is too low")
    inset = 0.02 + lr
    parts = [_box(w, seat_t, d, y=sh - seat_t)]
    parts += [_cyl(lr, sh - seat_t, x, 0, z) for x in (-w / 2 + inset, w / 2 - inset) for z in (-d / 2 + inset, d / 2 - inset)]
    parts.append(_box(w, bh, back_t, y=sh, z=-d / 2 + back_t / 2))
    if p["arms"]:
        arm_y = sh + min(0.22, bh * 0.5)
        for x in (-w / 2 + 0.02, w / 2 - 0.02):
            parts.append(_box(0.04, 0.03, d * 0.9, x=x, y=arm_y))
            parts.append(_cyl(0.012, arm_y - sh, x, sh, d * 0.35))
    return _join(parts)


@shape("stool", "",
       {"radius": length("Seat radius", 0.18), "height": length("Seat height", 0.65), "legs": integer("Legs", 3, 6, 3)})
def stool(p: dict[str, Any]) -> trimesh.Trimesh:
    r, h, n = p["radius"], p["height"], p["legs"]
    seat_t, lr = 0.04, 0.015
    _need(h > seat_t, "stool 'height' is too low")
    lr_pos = r * 0.75
    parts = [_cyl(r, seat_t, y=h - seat_t)]
    parts += [_cyl(lr, h - seat_t, lr_pos * math.sin(2 * math.pi * i / n), 0, lr_pos * math.cos(2 * math.pi * i / n))
              for i in range(n)]
    return _join(parts)


@shape("bench", "",
       {"width": length("Length along X", 1.2), "depth": length("Seat depth", 0.35),
        "height": length("Seat height", 0.45), "back": {"type": "boolean", "default": False}})
def bench(p: dict[str, Any]) -> trimesh.Trimesh:
    w, d, h = p["width"], p["depth"], p["height"]
    t = 0.04
    _need(h > t, "bench 'height' is too low")
    parts = [_box(w, t, d, y=h - t)]
    parts += [_box(0.05, h - t, d * 0.85, x=x) for x in (-w / 2 + 0.08, w / 2 - 0.08)]
    if p["back"]:
        parts.append(_box(w, 0.4, 0.03, y=h, z=-d / 2 + 0.015))
    return _join(parts)


@shape("bookshelf", "Open.",
       {"width": length("Width", 0.8), "depth": length("Depth", 0.3), "height": length("Height", 1.8),
        "shelves": integer("Inner shelves", 0, 12, 4)})
def bookshelf(p: dict[str, Any]) -> trimesh.Trimesh:
    w, d, h, n = p["width"], p["depth"], p["height"], p["shelves"]
    t = 0.02
    _need(w > 2 * t and h > 2 * t, "bookshelf is too small")
    inner = w - 2 * t
    parts = [_box(t, h, d, x=-w / 2 + t / 2), _box(t, h, d, x=w / 2 - t / 2),
             _box(inner, t, d), _box(inner, t, d, y=h - t), _box(inner, h - 2 * t, 0.01, y=t, z=-d / 2 + 0.005)]
    step = (h - t) / (n + 1)
    parts += [_box(inner, t, d - 0.01, y=step * (i + 1), z=0.005) for i in range(n)]
    return _join(parts)


@shape("bed", "Headboard at the back; length along Z.",
       {"width": length("Width", 1.6), "length": length("Length", 2.0), "height": length("Mattress top height", 0.5),
        "headboard_height": length("Headboard height above the mattress", 0.5)})
def bed(p: dict[str, Any]) -> trimesh.Trimesh:
    w, l, h, hb = p["width"], p["length"], p["height"], p["headboard_height"]
    frame_h = h * 0.55
    parts = [_box(w, frame_h, l), _box(w - 0.06, h - frame_h, l - 0.06, y=frame_h),
             _box(w + 0.04, h + hb, 0.06, z=-l / 2 - 0.03)]
    return _join(parts)


@shape("sofa", "",
       {"width": length("Width", 2.0), "depth": length("Depth", 0.9), "height": length("Back height", 0.8),
        "seat_height": length("Seat height", 0.42), "arms": {"type": "boolean", "default": True}})
def sofa(p: dict[str, Any]) -> trimesh.Trimesh:
    w, d, h, sh = p["width"], p["depth"], p["height"], p["seat_height"]
    _need(h > sh, "sofa 'height' must be more than 'seat_height'")
    back_t, arm_w = 0.2, 0.18
    parts = [_box(w, sh, d), _box(w, h, back_t, z=-d / 2 + back_t / 2)]
    if p["arms"]:
        parts += [_box(arm_w, sh + 0.2, d, x=x) for x in (-w / 2 + arm_w / 2, w / 2 - arm_w / 2)]
    return _join(parts)


# --- vessels: smooth hollow solids of revolution (open at the top) ----------------------

_HANDLES = integer("Handles on its sides: 2 = one each side", 0, 2, 0)
# Not an enum: "smooth", "square", ... are read by sweep.handle_style.
_HANDLE_STYLE = {"type": "string", "description": "round (smooth) | box", "default": "round"}


def _vessel(control: list[tuple[float, float]], radius: float, floor: float, wall: float | None = None,
            segments: int = 48) -> trimesh.Trimesh:
    """Smooth the (radius, height) control points, hollow them out and spin them."""
    narrowest = min(r for r, _ in control)
    # The wall must fit inside the narrowest part, or the inside would poke through it.
    wall = min(wall if wall is not None else max(0.003, 0.06 * radius), 0.45 * narrowest)
    try:
        return revolve_profile(vessel_profile(catmull_rom(control), wall, floor), segments)
    except ToolArgumentError as exc:
        radii = ", ".join(f"{r:g}" for r, _ in control)
        raise ToolArgumentError(f"these radii (bottom to top: {radii} m) jump too much to make a smooth shape; "
                                f"keep neighbouring radii within a few times of each other") from exc


@shape("vase", "Hollow. Radii bottom to top: base_radius, radius (belly, 1/3 up), neck_radius, top_radius "
               "(rim); any can be larger, e.g. a wide top. handles: 2 = one each side.",
       {"height": length("Height", 0.3), "radius": length("Belly radius", 0.08),
        "neck_radius": length("Neck radius", 0.035), "base_radius": length("Base radius"),
        "top_radius": length("Rim radius"), "handles": _HANDLES, "handle_style": _HANDLE_STYLE})
def vase(p: dict[str, Any]) -> trimesh.Trimesh:
    h, r, n = p["height"], p["radius"], p["neck_radius"]
    base = p.get("base_radius") or 0.6 * r
    top = p.get("top_radius") or 1.35 * n
    belly_low = base + 0.75 * (r - base)
    neck_mid = r + 0.5 * (n - r)
    if top <= 1.5 * n:
        neck_y = 0.82
        profile = [(base, 0), (belly_low, 0.12 * h), (r, 0.35 * h), (neck_mid, (0.35 + neck_y) / 2 * h),
                   (n, neck_y * h), (top, h)]
        neck_middle = neck_y
    else:
        # A flared rim: a real neck first, then the flare in the top sixth. Flaring all the way up
        # from the shoulder made a funnel (seen with Qwen3-8B, "an amphora with a flared rim").
        neck_y = 0.66
        profile = [(base, 0), (belly_low, 0.12 * h), (r, 0.35 * h), (neck_mid, 0.52 * h), (n, neck_y * h),
                   (n * 1.05, 0.84 * h), (top, h)]
        neck_middle = 0.75
    body = _vessel(profile, max(base, r, n, top), floor=0.04 * h)
    if not p["handles"]:
        return body
    t = max(0.004, min(0.15 * r, 0.05 * h))
    if p["handles"] == 1:
        # One handle is a jug's: from just under the rim down onto the body.
        return _join([body, *_side_handles(body, h, max(base, r, n, top), 1, 0.4, 0.93, t, p["handle_style"])])
    # Two are an amphora's: from the shoulder (above the belly, where the body turns in) up to the
    # middle of the neck. Lower down, the belly bulges under the end and the bar meets it like a hook.
    return _join([body, *_side_handles(body, h, max(base, r, n, top), 2, 0.5, neck_middle, t,
                                       p["handle_style"])])


@shape("bottle", "",
       {"height": length("Height", 0.3), "radius": length("Body radius", 0.04),
        "neck_radius": length("Neck radius", 0.013)})
def bottle(p: dict[str, Any]) -> trimesh.Trimesh:
    h, r, n = p["height"], p["radius"], p["neck_radius"]
    _need(n < r, "bottle 'neck_radius' must be smaller than 'radius'")
    wall = max(0.002, 0.25 * n)
    return _vessel([(0.94 * r, 0), (r, 0.04 * h), (r, 0.55 * h), (0.75 * r, 0.68 * h), (n, 0.78 * h),
                    (n, 0.97 * h), (n * 1.15, h)], r, floor=0.03 * h, wall=wall)


def _side_handles(body: trimesh.Trimesh, h: float, r: float, count: int, low: float, high: float,
                  thickness: float, style: str = "round") -> list[trimesh.Trimesh]:
    """
    Handles on a vessel's sides, +X first then -X, from `low` to `high` of its height, built onto
    its surface by the same engine as the handle shape (engine/attach.py).
    """
    out = []
    for side in ("right", "left")[:count]:
        f = attach.frame_for(side, "vertical", body.bounds)
        out.append(attach.build(body, f, high * h, low * h, reach=max(0.45 * r, 2 * thickness), thickness=thickness,
                                style=sweep.handle_style(style), ear=True))
    return out


def _lug_handles(body: trimesh.Trimesh, h: float, r: float, count: int, level: float, thickness: float,
                 style: str = "round") -> list[trimesh.Trimesh]:
    """
    Sideways loops near the rim (a pot's or a bowl's lugs): ends side by side at `level` of the
    height, on +X then -X. A low bowl has no room for an upright handle.
    """
    out = []
    span = 0.7 * r
    for side in ("right", "left")[:count]:
        f = attach.frame_for(side, "horizontal", body.bounds)
        f.k0 = level * h
        out.append(attach.build(body, f, span / 2, -span / 2, reach=max(2 * thickness, 0.3 * span),
                                thickness=thickness, style=sweep.handle_style(style), ear=False))
    return out


@shape("cup", "Hollow; handle: true = a mug.",
       {"height": length("Height", 0.1), "radius": length("Top radius", 0.04),
        "handle": {"type": "boolean", "default": False}, "handles": _HANDLES, "handle_style": _HANDLE_STYLE})
def cup(p: dict[str, Any]) -> trimesh.Trimesh:
    h, r = p["height"], p["radius"]
    control = [(0.82 * r, 0), (0.9 * r, 0.4 * h), (r, h)]
    body = _vessel(control, r, floor=0.08 * h)
    count = p["handles"] or (1 if p["handle"] else 0)
    if not count:
        return body
    return _join([body, *_side_handles(body, h, r, count, 0.2, 0.8, max(0.004, 0.2 * r), p["handle_style"])])


@shape("bowl", "handles: 2 = one each side.",
       {"radius": length("Rim radius", 0.08), "height": length("Height", 0.06),
        "handles": _HANDLES, "handle_style": _HANDLE_STYLE})
def bowl(p: dict[str, Any]) -> trimesh.Trimesh:
    h, r = p["height"], p["radius"]
    control = [(0.45 * r, 0), (0.5 * r, 0.08 * h), (0.8 * r, 0.35 * h), (0.96 * r, 0.7 * h), (r, h)]
    body = _vessel(control, r, floor=0.1 * h)
    if not p["handles"]:
        return body
    t = max(0.004, min(0.12 * r, 0.15 * h))
    return _join([body, *_lug_handles(body, h, r, p["handles"], 0.8, t, p["handle_style"])])


@shape("plant_pot", "Also a cooking pot: handles 2 = side lugs.",
       {"height": length("Height", 0.2), "radius": length("Top radius", 0.1), "handles": _HANDLES,
        "handle_style": _HANDLE_STYLE})
def plant_pot(p: dict[str, Any]) -> trimesh.Trimesh:
    h, r = p["height"], p["radius"]
    # Straight segments (no smoothing): a pot has crisp edges.
    profile = [(0.7 * r, 0), (0.92 * r, 0.85 * h), (1.06 * r, 0.86 * h), (1.06 * r, h)]
    body = revolve_profile(vessel_profile(profile, max(0.004, 0.07 * r), 0.08 * h), 48)
    if not p["handles"]:
        return body
    t = max(0.005, min(0.1 * r, 0.12 * h))
    return _join([body, *_lug_handles(body, h, r, p["handles"], 0.75, t, p["handle_style"])])


@shape("wine_glass", "",
       {"height": length("Height", 0.2), "radius": length("Bowl radius", 0.04)})
def wine_glass(p: dict[str, Any]) -> trimesh.Trimesh:
    h, r = p["height"], p["radius"]
    stem_r, stem_top = max(0.003, 0.1 * r), 0.45 * h
    foot = [(0.85 * r, 0), (0.85 * r, 0.015 * h), (stem_r * 1.6, 0.04 * h), (stem_r, 0.08 * h), (stem_r, stem_top)]
    cup_ = catmull_rom([(stem_r, stem_top), (0.6 * r, 0.52 * h), (r, 0.72 * h), (0.85 * r, h)])
    return revolve_profile(vessel_profile(foot + cup_[1:], max(0.0015, 0.04 * r), stem_top + 0.06 * h), 48)


@shape("lamp", "Floor lamp",
       {"height": length("Total height", 1.6), "shade_radius": length("Shade radius", 0.2)})
def lamp(p: dict[str, Any]) -> trimesh.Trimesh:
    h, sr = p["height"], p["shade_radius"]
    shade_h = min(0.3, h * 0.25)
    _need(h > shade_h + 0.05, "lamp 'height' is too low")
    return _join([_cyl(max(sr * 0.75, 0.1), 0.03), _cyl(0.012, h - shade_h, y=0.03),
                  _cyl(sr, shade_h, y=h - shade_h, r_top=sr * 0.6)])


# Things that always stand on something: one floating in mid-air is a mistake (see repair.py).
STANDING_OBJECTS = frozenset({"table", "chair", "stool", "bench", "bookshelf", "bed", "sofa", "vase", "bottle",
                              "cup", "bowl", "plant_pot", "wine_glass", "lamp"})
