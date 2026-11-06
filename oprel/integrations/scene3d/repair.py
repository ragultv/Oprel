"""
Deterministic fixes for geometry mistakes small models make and then fail to fix in a few
rounds (seen with Qwen3-8B at 4k context):

  fit_supports   repeated legs that stick out past what rests on them -> shrink the pattern
  close_gaps     a part a few cm above the part it should rest on    -> lower it onto it

The engine has the exact numbers; the model does not. Each fix edits the recipe (so it is
saved and visible) and is reported back as a note. Anything ambiguous is left to warnings.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from oprel.integrations.scene3d.engine import patterns
from oprel.integrations.scene3d.engine.compiler import CompiledObject, CompiledScene
from oprel.integrations.scene3d.engine.frame import aabb, transform_points, vec3

TOL = 0.002
MAX_GAP = 0.1        # metres; a bigger gap is probably intended (a lamp hanging above a table)
INSET = 0.04         # keep legs this fraction of the top's half-size away from its edge


def _rotated(oid: str, by_id: dict[str, dict[str, Any]]) -> bool:
    """Does the object or anything containing it have a rotation? Then axis edits aren't simple."""
    node: str | None = oid
    while node is not None and node in by_id:
        if any(vec3(by_id[node].get("rotation"))):
            return True
        node = by_id[node].get("parent")
    return False


def _boxes(co: CompiledObject) -> list[tuple[np.ndarray, np.ndarray]]:
    return [aabb(transform_points(m, co.built.points)) for m in co.instances] if co.built else []


def _overlap_xz(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]) -> bool:
    return bool(a[0][0] <= b[1][0] + TOL and b[0][0] <= a[1][0] + TOL
                and a[0][2] <= b[1][2] + TOL and b[0][2] <= a[1][2] + TOL)


def fit_supports(objects: list[dict[str, Any]], scene: CompiledScene) -> list[str]:
    by_id = {o["id"]: o for o in objects}
    notes: list[str] = []
    for sid, support in scene.objects.items():
        spec = by_id.get(sid)
        pattern = (spec or {}).get("pattern") or {}
        if support.built is None or len(support.instances) < 2 or pattern.get("type") not in ("radial", "grid"):
            continue
        if _rotated(sid, by_id):
            continue
        slo, shi = support.bbox
        for hid, held in scene.objects.items():
            if hid == sid or held.built is None:
                continue
            hlo, hhi = held.bbox
            if abs(hlo[1] - shi[1]) > TOL or not _overlap_xz(held.bbox, support.bbox):
                continue  # not resting on the supports
            half = (hhi - hlo) / 2
            overhang = max(max(hlo[a] - slo[a], shi[a] - hhi[a]) for a in (0, 2))
            if overhang <= 0.02 * max(half[0], half[2]) * 2 + TOL:
                continue
            # Pattern centre = the support's own frame origin (copies are laid out around it).
            first_copy = patterns.copies(pattern)[0]
            centre = (support.instances[0] @ np.linalg.inv(first_copy))[:3, 3]
            hc = (hlo + hhi) / 2
            leg_half = np.abs(support.built.points).max(axis=0)
            inset = INSET * min(half[0], half[2])
            room = [half[a] - abs(centre[a] - hc[a]) - inset - leg_half[a] for a in (0, 2)]
            if pattern["type"] == "radial":
                new_r = round(min(room), 3)
                old_r = float(pattern.get("radius", 0))
                if new_r <= 0 or new_r >= old_r:
                    continue
                pattern["radius"] = new_r
                notes.append(f"Moved the {len(support.instances)} copies of '{sid}' in so they fit under "
                             f"'{hid}' (radius {old_r:g} -> {new_r:g} m).")
            else:
                count, rows = int(pattern["count"]), int(pattern.get("rows", 1))
                sx, sz = (list(pattern.get("spacing") or [0, 0]) + [0, 0])[:2]
                new_sx = min(sx, 2 * room[0] / (count - 1)) if count > 1 else sx
                new_sz = min(sz, 2 * room[1] / (rows - 1)) if rows > 1 else sz
                if new_sx <= 0 or new_sz <= 0 or (new_sx, new_sz) == (sx, sz):
                    continue
                pattern["spacing"] = [round(new_sx, 3), round(new_sz, 3)]
                notes.append(f"Moved the {len(support.instances)} copies of '{sid}' in so they fit under "
                             f"'{hid}' (spacing {[sx, sz]} -> {pattern['spacing']} m).")
            break
    return notes


def close_gaps(objects: list[dict[str, Any]], scene: CompiledScene) -> list[str]:
    by_id = {o["id"]: o for o in objects}
    boxes = {oid: _boxes(co) for oid, co in scene.objects.items() if co.built is not None}
    notes: list[str] = []
    for oid, own in boxes.items():
        spec = by_id.get(oid) or {}
        if not spec.get("parent") or spec.get("placement") or _rotated(oid, by_id):
            continue  # top-level objects may float on purpose; placement already computes contact
        bottom = min(b[0][1] for b in own)
        if bottom <= TOL:
            continue
        others = [b for other, bs in boxes.items() if other != oid for b in bs]
        touching = any(np.all(b[0] <= ob[1] + TOL) and np.all(ob[0] <= b[1] + TOL) for b in own for ob in others)
        if touching:
            continue
        below = [ob[1][1] for ob in others if ob[1][1] <= bottom + TOL and any(_overlap_xz(b, ob) for b in own)]
        if not below:
            continue
        gap = bottom - max(below)
        if not TOL < gap <= MAX_GAP:
            continue
        pos = vec3(spec.get("position"))
        pos[1] = round(pos[1] - gap, 4)
        spec["position"] = pos
        notes.append(f"Lowered '{oid}' {gap:.3f} m so it rests on the part below it.")
    return notes


def repair(objects: list[dict[str, Any]], scene: CompiledScene) -> list[str]:
    """Apply every fix to `objects` in place. Returns notes; empty when nothing changed."""
    return fit_supports(objects, scene) or close_gaps(objects, scene)
