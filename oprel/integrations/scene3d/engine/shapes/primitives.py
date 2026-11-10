"""
Row 1: primitives. Builders return a mesh in any position; the compiler moves it so its
bottom-centre sits at the origin, so builders don't need to care about anchoring.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import trimesh

from oprel.integrations.scene3d.engine.frame import Z_UP_TO_Y_UP
from oprel.integrations.scene3d.engine.registry import (
    ToolArgumentError, integer, length, register_meshless, shape,
)

SEGMENTS = integer("Smoothness of round surfaces", 3, 128, 48)
THIN = 0.002  # default thickness of flat shapes, so they are visible and solid


def _y_up(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    return mesh.apply_transform(Z_UP_TO_Y_UP)


def _solid_of_revolution(profile: list[list[float]], segments: int) -> trimesh.Trimesh:
    """Spin a (radius, height) profile around the Y axis."""
    mesh = trimesh.creation.revolve(np.asarray(profile, dtype=float), sections=segments)
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    # Revolve keeps the winding consistent; only the direction depends on the profile order.
    # (fix_normals() would do this too but needs scipy.)
    if mesh.volume < 0:
        mesh.invert()
    return _y_up(mesh)


@shape("box", "Rectangular block: width (X) x height (Y) x depth (Z).",
       {"width": length("Size along X"), "height": length("Size along Y"), "depth": length("Size along Z")},
       required=("width", "height", "depth"))
def box(p: dict[str, Any]) -> trimesh.Trimesh:
    return trimesh.creation.box(extents=[p["width"], p["height"], p["depth"]])


@shape("cylinder", "Upright; radius, or radius_top + radius_bottom = tapered.",
       {"radius": length("Radius"), "radius_top": length("Top radius (tapered)"),
        "radius_bottom": length("Bottom radius (tapered)"), "height": length("Height"), "segments": SEGMENTS},
       required=("height",))
def cylinder(p: dict[str, Any]) -> trimesh.Trimesh:
    top = p.get("radius_top", p.get("radius"))
    bottom = p.get("radius_bottom", p.get("radius"))
    if top is None or bottom is None:
        raise ToolArgumentError("cylinder needs 'radius' (or both 'radius_top' and 'radius_bottom')")
    if abs(top - bottom) < 1e-9:
        return _y_up(trimesh.creation.cylinder(radius=top, height=p["height"], sections=p["segments"]))
    return _solid_of_revolution([[0, 0], [bottom, 0], [top, p["height"]], [0, p["height"]]], p["segments"])


@shape("cone", "Upright, point at the top; radius_top = cut off (funnel, pouring cup: top wider).",
       {"radius": length("Bottom radius"), "radius_top": {"type": "number", "minimum": 0, "maximum": 1000},
        "height": length("Height"), "radius_bottom": length("Bottom radius"), "segments": SEGMENTS},
       required=("height",))
def cone(p: dict[str, Any]) -> trimesh.Trimesh:
    # Qwen3-8B drew "a pouring cup cone from radius 1.5 cm to 5 cm" as a spike: a cone had no top radius.
    bottom = p.get("radius_bottom", p.get("radius"))
    if bottom is None:
        raise ToolArgumentError("cone needs 'radius' (its bottom); add 'radius_top' to cut the point off")
    top = p.get("radius_top") or 0.0
    if top <= 1e-9:
        return _y_up(trimesh.creation.cone(radius=bottom, height=p["height"], sections=p["segments"]))
    return _solid_of_revolution([[0, 0], [bottom, 0], [top, p["height"]], [0, p["height"]]], p["segments"])


@shape("sphere", "",
       {"radius": length("Radius"), "segments": integer("Smoothness", 6, 128, 32)},
       required=("radius",))
def sphere(p: dict[str, Any]) -> trimesh.Trimesh:
    return trimesh.creation.uv_sphere(radius=p["radius"], count=[p["segments"], p["segments"]])


@shape("capsule", "Upright pill; height includes the round ends.",
       {"radius": length("Radius"), "height": length("Total height"), "segments": integer("Smoothness", 6, 128, 32)},
       required=("radius", "height"))
def capsule(p: dict[str, Any]) -> trimesh.Trimesh:
    if p["height"] < 2 * p["radius"]:
        raise ToolArgumentError("capsule 'height' must be at least 2 x radius (it includes both round ends)")
    mesh = trimesh.creation.capsule(height=p["height"] - 2 * p["radius"], radius=p["radius"],
                                    count=[p["segments"], p["segments"]])
    return _y_up(mesh)


@shape("torus", "Ring (donut) lying flat on the ground.",
       {"major_radius": length("Radius from the centre to the middle of the tube"),
        "minor_radius": length("Radius of the tube"), "segments": SEGMENTS},
       required=("major_radius", "minor_radius"))
def torus(p: dict[str, Any]) -> trimesh.Trimesh:
    if p["minor_radius"] >= p["major_radius"]:
        raise ToolArgumentError("torus 'minor_radius' must be smaller than 'major_radius'")
    mesh = trimesh.creation.torus(major_radius=p["major_radius"], minor_radius=p["minor_radius"],
                                  major_sections=p["segments"], minor_sections=max(12, p["segments"] // 2))
    return _y_up(mesh)


@shape("rectangle", "Thin plate lying flat: width (X) x depth (Z).",
       {"width": length("Size along X"), "depth": length("Size along Z"),
        "thickness": length("Thickness", THIN)},
       required=("width", "depth"))
def rectangle(p: dict[str, Any]) -> trimesh.Trimesh:
    return trimesh.creation.box(extents=[p["width"], p["thickness"], p["depth"]])


@shape("circle", "Flat disc lying on the ground.",
       {"radius": length("Radius"), "thickness": length("Thickness", THIN), "segments": SEGMENTS},
       required=("radius",))
def circle(p: dict[str, Any]) -> trimesh.Trimesh:
    return _y_up(trimesh.creation.cylinder(radius=p["radius"], height=p["thickness"], sections=p["segments"]))


register_meshless("group", "Empty container. Give parts `parent: <group id>` so they move together.")
