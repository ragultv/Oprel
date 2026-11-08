"""
Row 5: merged groups. A group with "merge": true is one solid: every part inside it (with its
pattern copies) is fused by an exact mesh union (manifold3d) into a single watertight surface,
with nothing left buried inside. That is what flow simulation, 3D printing and CAD expect.

The parts stay separate in the recipe, so they can still be edited, placed and patterned; the
union is redone on every compile. The whole scene can also be fused for an STL download.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import manifold3d as m3
import numpy as np
import trimesh

from oprel.integrations.scene3d.engine.cuts import as_solid, to_trimesh


@dataclass(frozen=True)
class MergedSolid:
    mesh: trimesh.Trimesh   # world coordinates
    bodies: int             # separate pieces after the union (1 = one solid part)
    parts: tuple[str, ...]  # ids of the objects fused into it


def _union(items: list[tuple[trimesh.Trimesh, np.ndarray]]) -> tuple[trimesh.Trimesh, int]:
    solids = [as_solid(mesh.copy().apply_transform(np.asarray(matrix)), "A merged part") for mesh, matrix in items]
    fused = solids[0] if len(solids) == 1 else m3.Manifold.batch_boolean(solids, m3.OpType.Add)
    return to_trimesh(fused), len(fused.decompose())


def _descendants(gid: str, specs: dict[str, dict[str, Any]]) -> list[str]:
    out, frontier = [], [gid]
    while frontier:
        node = frontier.pop()
        kids = [oid for oid, spec in specs.items() if spec.get("parent") == node]
        out.extend(kids)
        frontier.extend(kids)
    return out


def merged_roots(specs: dict[str, dict[str, Any]]) -> list[str]:
    """Merged groups not inside another merged group (an outer merge includes the inner one)."""
    def merged_ancestor(oid: str) -> bool:
        node = specs[oid].get("parent")
        while node is not None and node in specs:
            if specs[node].get("merge"):
                return True
            node = specs[node].get("parent")
        return False
    return [oid for oid, spec in specs.items()
            if spec.get("shape") == "group" and spec.get("merge") and not merged_ancestor(oid)]


def merged_into(specs: dict[str, dict[str, Any]]) -> dict[str, str]:
    """part id -> the merged group it is fused into."""
    return {part: gid for gid in merged_roots(specs) for part in _descendants(gid, specs)}


def merge_groups(objects: dict[str, Any]) -> dict[str, MergedSolid]:
    """Fuse every merged group of a compiled scene (`objects`: id -> CompiledObject)."""
    specs = {oid: co.spec for oid, co in objects.items()}
    out: dict[str, MergedSolid] = {}
    for gid in merged_roots(specs):
        parts = [oid for oid in _descendants(gid, specs) if objects[oid].built is not None]
        items = [(objects[oid].built.mesh, m) for oid in parts for m in objects[oid].instances]
        if items:
            mesh, bodies = _union(items)
            out[gid] = MergedSolid(mesh, bodies, tuple(parts))
    return out


def whole_scene(objects: dict[str, Any]) -> MergedSolid | None:
    """Everything in the scene fused into one mesh (for an STL), or None when nothing has a mesh."""
    items = [(co.built.mesh, m) for co in objects.values() if co.built is not None for m in co.instances]
    if not items:
        return None
    mesh, bodies = _union(items)
    return MergedSolid(mesh, bodies, tuple(oid for oid, co in objects.items() if co.built is not None))
