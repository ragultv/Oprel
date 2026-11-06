"""Compiled scene -> GLB. Nodes are named by object id ('legs#2' for copy 3) so the UI can pick them."""

from __future__ import annotations

import re

import numpy as np
import trimesh
from trimesh.visual.material import PBRMaterial

from oprel.integrations.scene3d.engine.compiler import CompiledScene

DEFAULT_COLOR = (0.72, 0.74, 0.78)
# Used when the model gives no colour (small models usually don't): ready-made objects look like themselves.
SHAPE_COLORS = {
    "table": "#8b5a2b", "chair": "#a0522d", "stool": "#a0522d", "bench": "#8b5a2b", "bookshelf": "#7a4a24",
    "bed": "#d8c8a8", "sofa": "#6b7280", "lamp": "#e8d9b0",
}

_NAMED = {
    "black": "#111111", "white": "#f5f5f5", "gray": "#808080", "grey": "#808080", "silver": "#c0c0c0",
    "red": "#d32f2f", "green": "#388e3c", "blue": "#1976d2", "yellow": "#fbc02d", "orange": "#f57c00",
    "purple": "#7b1fa2", "pink": "#e91e63", "brown": "#795548", "cyan": "#00acc1", "teal": "#00897b",
    "gold": "#d4af37", "beige": "#d8c8a8", "navy": "#1a237e", "wood": "#8b5a2b", "glass": "#a8d8ea",
}
_HEX = re.compile(r"^#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


def parse_color(value: str | None) -> tuple[float, float, float] | None:
    """'#8b5a2b', '8b5a2b', '#abc' or a common colour name -> linear-ish RGB 0..1. None if unknown."""
    if not value:
        return None
    text = _NAMED.get(value.strip().lower(), value.strip())
    m = _HEX.match(text)
    if not m:
        return None
    h = m.group(1)
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]


def export_glb(scene: CompiledScene) -> bytes | None:
    """GLB bytes, or None when nothing in the scene has a mesh."""
    out = trimesh.Scene()
    for oid, co in scene.objects.items():
        if co.built is None:
            continue
        rgb = (parse_color(co.spec.get("color")) or parse_color(SHAPE_COLORS.get(co.spec["shape"]))
               or DEFAULT_COLOR)
        opacity = float(co.spec.get("opacity", 1.0))
        material = PBRMaterial(
            name=f"{oid}_mat",
            baseColorFactor=[*(int(c * 255) for c in rgb), int(opacity * 255)],
            metallicFactor=0.0,
            roughnessFactor=0.65,
            alphaMode="BLEND" if opacity < 1 else "OPAQUE",
            doubleSided=False,
        )
        for i, matrix in enumerate(co.instances):
            mesh = co.built.mesh.copy()
            mesh.visual = trimesh.visual.TextureVisuals(material=material)
            name = oid if len(co.instances) == 1 else f"{oid}#{i}"
            out.add_geometry(mesh, node_name=name, geom_name=name, transform=np.asarray(matrix))
    if not out.geometry:
        return None
    return out.export(file_type="glb")
