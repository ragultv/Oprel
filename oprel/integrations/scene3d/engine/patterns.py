"""
Row 2: repeat an object (and everything parented to it) as a pattern of copies.

  linear  count copies, each `step` [x, y, z] further on; the first copy sits at the object's position
  grid    `count` columns along X by `rows` along Z, `spacing` [sx, sz], centred on the position
  radial  `count` copies on a circle of `radius` around the position (in the ground plane),
          starting at the front (+Z) and going round `arc` degrees; `face_center` turns each copy
          so its front faces the middle
  around  `count` copies set around another object (`relative_to`), each facing it, `gap` metres
          from its edge (negative tucks them under): along the long sides of a rectangular
          footprint (and the ends for 6+ or an odd count), or on a circle for a round one.
          Needs the scene, so the compiler resolves it (see around()).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from oprel.integrations.scene3d.engine.frame import IDENTITY, rotation_y_deg, translation, vec3
from oprel.tools.validation import ToolArgumentError

MAX_COPIES = 500

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": ["linear", "grid", "radial", "around"]},
        "relative_to": {"type": "string"},
        "gap": {"type": "number", "minimum": -10, "maximum": 100},
        "count": {"type": "integer", "minimum": 1, "maximum": MAX_COPIES},
        "rows": {"type": "integer", "minimum": 1, "maximum": MAX_COPIES},
        "step": {"type": "array", "items": {"type": "number"}, "maxItems": 3},
        "spacing": {"type": "array", "items": {"type": "number"}, "maxItems": 2},
        "radius": {"type": "number", "minimum": 0, "maximum": 1000},
        "arc": {"type": "number", "minimum": 1, "maximum": 360},
        "start_angle": {"type": "number"},
        "face_center": {"type": "boolean"},
    },
    "required": ["type", "count"],
}


def copies(pattern: dict[str, Any] | None) -> list[np.ndarray]:
    """Transforms of every copy, relative to the object's own frame. No pattern = [identity]."""
    if not pattern:
        return [IDENTITY]
    kind = pattern["type"]
    count = int(pattern["count"])

    if kind == "linear":
        if not pattern.get("step"):
            raise ToolArgumentError("linear pattern needs 'step' [x, y, z]: the offset between copies")
        step = np.array(vec3(pattern["step"]))
        return [translation(step * i) for i in range(count)]

    if kind == "grid":
        rows = int(pattern.get("rows", 1))
        if count * rows > MAX_COPIES:
            raise ToolArgumentError(f"grid pattern makes {count * rows} copies; the limit is {MAX_COPIES}")
        spacing = [float(v) for v in (pattern.get("spacing") or [])]
        if len(spacing) == 1:
            spacing *= 2
        if len(spacing) != 2:
            raise ToolArgumentError("grid pattern needs 'spacing' [sx, sz]: the distance between columns and rows")
        sx, sz = spacing
        x0 = -(count - 1) * sx / 2
        z0 = -(rows - 1) * sz / 2
        return [translation([x0 + c * sx, 0, z0 + r * sz]) for r in range(rows) for c in range(count)]

    if kind == "radial":
        radius = float(pattern.get("radius", 0))
        arc = float(pattern.get("arc", 360))
        start = float(pattern.get("start_angle", 0))
        # A full circle must not put the last copy on top of the first.
        step = arc / count if arc >= 360 else (arc / (count - 1) if count > 1 else 0)
        out = []
        for i in range(count):
            a = start + step * i
            rad = np.radians(a)
            m = translation([radius * np.sin(rad), 0, radius * np.cos(rad)])
            if pattern.get("face_center"):
                m = m @ rotation_y_deg(a + 180)  # turns the copy's +Z toward the centre
            elif radius == 0:
                m = m @ rotation_y_deg(a)  # copies spin in place (spokes, petals)
            out.append(m)
        return out

    if kind == "around":
        raise ToolArgumentError("'around' patterns are resolved by the compiler")  # needs the reference's size

    raise ToolArgumentError(f"Unknown pattern type '{kind}'")


def around(count: int, ref_lo: np.ndarray, ref_hi: np.ndarray, own_depth: float, gap: float,
           round_ref: bool) -> list[np.ndarray]:
    """
    Copies set around a reference footprint (ref_lo/ref_hi in the same frame as the result),
    each turned so its front (+Z) faces the reference. own_depth is the copy's size along Z.
    """
    cx, cz = (ref_lo[0] + ref_hi[0]) / 2, (ref_lo[2] + ref_hi[2]) / 2
    w, d = ref_hi[0] - ref_lo[0], ref_hi[2] - ref_lo[2]
    out = []
    if round_ref:
        r = max(w, d) / 2 + gap + own_depth / 2
        for i in range(count):
            a = 360 * i / count
            rad = np.radians(a)
            out.append(translation([cx + r * np.sin(rad), 0, cz + r * np.cos(rad)]) @ rotation_y_deg(a + 180))
        return out

    # Rectangle: copies along the two long sides, plus the ends for 6+ or an odd count.
    long_x = w >= d
    ends = 2 if count >= 6 and count % 2 == 0 else (1 if count % 2 else 0)
    if count < 3:
        ends = 0
    per_side = [(count - ends + 1) // 2, (count - ends) // 2]  # front side gets the extra one
    length_, across = (w, d) if long_x else (d, w)
    out_dist = across / 2 + gap + own_depth / 2
    end_dist = length_ / 2 + gap + own_depth / 2
    for side, n in zip((1, -1), per_side):
        for i in range(n):
            t = -length_ / 2 + length_ * (i + 0.5) / n
            if long_x:  # sides at +Z / -Z; a chair at +Z turns 180 to face -Z
                out.append(translation([cx + t, 0, cz + side * out_dist]) @ rotation_y_deg(180 if side > 0 else 0))
            else:       # sides at +X / -X
                out.append(translation([cx + side * out_dist, 0, cz + t]) @ rotation_y_deg(-90 if side > 0 else 90))
    for side in (1, -1)[:ends]:
        if long_x:
            out.append(translation([cx + side * end_dist, 0, cz]) @ rotation_y_deg(-90 if side > 0 else 90))
        else:
            out.append(translation([cx, 0, cz + side * end_dist]) @ rotation_y_deg(180 if side > 0 else 0))
    return out
