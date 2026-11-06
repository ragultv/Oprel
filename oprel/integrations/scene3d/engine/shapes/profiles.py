"""
lathe and extrude: shapes from 2D outlines (helpers and checks in engine/outline.py).
"""

from __future__ import annotations

from typing import Any

import trimesh
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

from oprel.integrations.scene3d.engine.outline import (
    FLAT_TO_GROUND, POINTS, catmull_rom, explain, pairs, revolve_profile,
)
from oprel.integrations.scene3d.engine.registry import ToolArgumentError, integer, length, shape


@shape("lathe", "profile [[radius, height], ...] bottom up, spun around Y; smooth curves it.",
       {"profile": {**POINTS, "description": "[radius, height] points from the bottom up"},
        "smooth": {"type": "boolean", "default": False},
        "segments": integer("Smoothness around", 8, 128, 48)},
       required=("profile",))
def lathe(p: dict[str, Any]) -> trimesh.Trimesh:
    pts = pairs(p["profile"], "profile", 2)
    if any(r < 0 for r, _ in pts):
        raise ToolArgumentError("lathe radii must be >= 0 (the profile is one side of the axis)")
    if p["smooth"]:
        pts = catmull_rom(pts)
    return revolve_profile(pts, p["segments"])


@shape("extrude", "Outline [[x, z], ...] with thickness up; plane front: [x, y] outline, standing.",
       {"points": {**POINTS, "description": "Outline corners in order"},
        "holes": {"type": "array", "items": POINTS, "maxItems": 20},
        "thickness": length("Thickness"),
        "plane": {"type": "string", "enum": ["ground", "front"], "default": "ground"}},
       required=("points", "thickness"))
def extrude(p: dict[str, Any]) -> trimesh.Trimesh:
    shell = pairs(p["points"], "points", 3)
    holes = [pairs(h, f"holes[{i}]", 3) for i, h in enumerate(p.get("holes") or [])]
    poly = Polygon(shell, holes)
    explain(poly, "The extrude outline",
             "Give the corners in order around the edge, without crossing lines; holes must be inside.")
    mesh = trimesh.creation.extrude_polygon(orient(poly), p["thickness"], engine="manifold")
    if p["plane"] == "ground":
        mesh.apply_transform(FLAT_TO_GROUND)
    if mesh.volume < 0:
        mesh.invert()
    return mesh
