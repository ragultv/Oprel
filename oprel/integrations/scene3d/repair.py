"""
Deterministic fixes for geometry mistakes small models make and then fail to fix in a few
rounds (seen with Qwen3-8B at 4k context):

  fit_supports   repeated legs that stick out past what rests on them -> shrink the pattern
  close_gaps     a part a few cm above the part it should rest on    -> lower it onto it
  spread_on_top  several things put "on top" of the same object      -> space them out
  drop_standing  a chair/vase/... hanging in mid-air                 -> set it down
  join_midpoint  a thin box drawn midway between two objects        -> place it between them
  drill_bolt_holes  a bolt through a part with no hole             -> drill the ISO clearance hole
  lift_trains    flanges / a big gear going into the floor          -> lift the run's first part

The engine has the exact numbers; the model does not. Each fix edits the recipe (so it is
saved and visible) and is reported back as a note. Anything ambiguous is left to warnings.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from oprel.integrations.scene3d.engine import patterns
from oprel.integrations.scene3d.engine.compiler import CompiledObject, CompiledScene, missing_bolt_holes
from oprel.integrations.scene3d.engine.frame import aabb, transform_points, vec3
from oprel.integrations.scene3d.engine.schema import PORT_SIDES
from oprel.integrations.scene3d.engine.shapes.objects import STANDING_OBJECTS

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


def spread_on_top(objects: list[dict[str, Any]], scene: CompiledScene) -> list[str]:
    """
    Several objects put on the same thing with just {"side": "top"} (no align/offset) all land
    in its centre, inside each other (seen with Qwen3-8B: bowl, two cups and a glass on a table).
    Space them evenly along the longer side of the surface instead.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for o in objects:
        pl = o.get("placement") or {}
        if pl.get("side") == "top" and pl.get("relative_to") and not pl.get("align") \
                and not any(vec3(pl.get("offset"))) and not o.get("pattern") and o["id"] in scene.objects:
            groups.setdefault(pl["relative_to"], []).append(o)
    notes: list[str] = []
    for ref, items in groups.items():
        if len(items) < 2 or ref not in scene.objects:
            continue
        lo, hi = scene.objects[ref].bbox
        axis = 0 if hi[0] - lo[0] >= hi[2] - lo[2] else 2
        usable = (hi[axis] - lo[axis]) * 0.8
        for i, o in enumerate(items):
            offset = [0.0, 0.0, 0.0]
            offset[axis] = round(-usable / 2 + usable * (i + 0.5) / len(items), 4)
            o["placement"]["offset"] = offset
        notes.append(f"Spread {', '.join(repr(o['id']) for o in items)} out across the top of '{ref}' "
                     f"(they were all in the same spot).")
    return notes


def drop_standing_objects(objects: list[dict[str, Any]], scene: CompiledScene) -> list[str]:
    """
    A ready-made object (furniture, vessel) floating in mid-air: lower it onto whatever is under
    it, or the ground. Seen with Qwen3-8B, which put a 1.5 m vase's centre at y=0.75.
    """
    boxes = {oid: _boxes(co) for oid, co in scene.objects.items() if co.built is not None}
    notes: list[str] = []
    for o in objects:
        oid = o["id"]
        if (o.get("shape") not in STANDING_OBJECTS or o.get("parent") or o.get("placement")
                or o.get("pattern") or oid not in boxes or _rotated(oid, {x["id"]: x for x in objects})):
            continue
        own = boxes[oid]
        bottom = min(b[0][1] for b in own)
        if bottom <= TOL:
            continue
        below = [ob[1][1] for other, bs in boxes.items() if other != oid for ob in bs
                 if ob[1][1] <= bottom + TOL and any(_overlap_xz(b, ob) for b in own)]
        rest_on = max(below, default=0.0)
        drop = bottom - rest_on
        if drop <= TOL:
            continue
        pos = vec3(o.get("position"))
        pos[1] = round(pos[1] - drop, 4)
        o["position"] = pos
        where = "the object under it" if below else "the ground"
        notes.append(f"Lowered '{oid}' {drop:.3f} m onto {where} (position is the bottom-centre).")
    return notes


def join_midpoint_boxes(objects: list[dict[str, Any]], scene: CompiledScene) -> list[str]:
    """
    A thin box drawn by coordinates exactly midway between two objects is a plate or beam joining
    them: place it `between` them so it reaches both and stays joined when they move. Seen with
    Qwen3-8B: the joining plate was drawn at the midpoint, then left behind when a cylinder moved.
    """
    notes: list[str] = []
    for o in objects:
        co = scene.objects.get(o["id"])
        params = o.get("params") or {}
        if (o.get("shape") != "box" or co is None or co.built is None or len(co.instances) != 1
                or o.get("placement") or o.get("pattern") or any(vec3(o.get("rotation")))
                or "width" not in params or "depth" not in params):
            continue
        if min(params["width"], params["depth"]) > 0.25 * max(params["width"], params["depth"]):
            continue  # not thin: a coffee table between two sofas is not a connector
        lo, hi = co.bbox
        centre = (lo + hi) / 2
        best = None
        siblings = [s for s in scene.objects.values() if s is not co and s.built is not None
                    and len(s.instances) == 1 and s.spec.get("parent") == o.get("parent")]
        for i, a in enumerate(siblings):
            for b in siblings[i + 1:]:
                ca, cb = (a.bbox[0] + a.bbox[1]) / 2, (b.bbox[0] + b.bbox[1]) / 2
                span = float(np.hypot(*(cb - ca)[[0, 2]]))
                off = float(np.hypot(*(centre - (ca + cb) / 2)[[0, 2]]))
                if span > 0.05 and off <= max(0.02, 0.1 * span) and (best is None or off < best[0]):
                    best = (off, a, b)
        if best is None:
            continue
        _, a, b = best
        if abs(lo[1] - min(a.bbox[0][1], b.bbox[0][1])) <= TOL:
            vertical = "bottom"
        elif abs(hi[1] - min(a.bbox[1][1], b.bbox[1][1])) <= TOL:
            vertical = "top"
        else:
            continue
        o.pop("position", None)
        o["placement"] = {"between": [a.id, b.id], **({"vertical": "top"} if vertical == "top" else {})}
        # Its thin side stays the thickness; the engine works out the length and turns it.
        o["params"] = {**{k: v for k, v in params.items() if k != "width"},
                       "depth": min(params["width"], params["depth"])}
        notes.append(f"Placed '{o['id']}' between '{a.id}' and '{b.id}' (it was at their midpoint), so it reaches "
                     f"both and stays joined when they move.")
    return notes


_BESIDE = {"left", "right", "front", "back"}


def onto_the_end(objects: list[dict[str, Any]], scene: CompiledScene) -> list[str]:
    """
    Placed beside an object and lifted (offset y) so it hangs in the air: it was meant to stand on
    top of it at that end. Seen with Qwen3-8B, "a vase on the left end of the counter": side
    "left", offset y 0.45, floating beside the counter. Put it on top, aligned to that side.
    """
    notes: list[str] = []
    for o in objects:
        pl = o.get("placement") or {}
        co, ref = scene.objects.get(o["id"]), scene.objects.get(pl.get("relative_to", ""))
        if (pl.get("side") not in _BESIDE or vec3(pl.get("offset"))[1] <= TOL or co is None or ref is None
                or co.bbox[0][1] <= TOL):
            continue
        under = [b for oid, c in scene.objects.items() if oid != o["id"] and c.built is not None
                 for b in _boxes(c) if b[1][1] <= co.bbox[0][1] + TOL and _overlap_xz(co.bbox, b)]
        if under:
            continue  # something holds it up
        side = pl["side"]
        o["placement"] = {"relative_to": pl["relative_to"], "side": "top", "align": side}
        notes.append(f"Put '{o['id']}' on top of '{ref.id}' at its {side} end (it was beside it, lifted into the air).")
    return notes


def drill_bolt_holes(objects: list[dict[str, Any]], scene: CompiledScene) -> list[str]:
    """
    A bolt through a part that has no hole for it: drill the ISO clearance hole where the bolt is
    (a bolt through a plate means a hole in it; at 4k context the model has no turn left to add it).
    """
    by_id = {o["id"]: o for o in objects}
    notes: list[str] = []
    for gap in missing_bolt_holes(scene):
        host = by_id.get(gap["host"])
        if host is None:
            continue
        host["cuts"] = list(host.get("cuts") or []) + gap["cuts"]
        size = gap["cuts"][0]["bolt"]
        many = f"{gap['copies']} {size} holes" if gap["copies"] > 1 else f"an {size} hole"
        notes.append(f"Drilled {many} in '{gap['host']}' for '{gap['bolt']}' (ISO clearance).")
    return notes


def _train_root(oid: str, by_id: dict[str, dict[str, Any]]) -> str | None:
    """The first part of a pipe run (parts joined end to end) or gear train that `oid` belongs to."""
    seen, node = set(), oid
    while node in by_id and node not in seen:
        seen.add(node)
        o = by_id[node]
        pl = o.get("placement") or {}
        ref = pl.get("relative_to")
        meshing = o.get("shape") == "gear" and by_id.get(ref, {}).get("shape") == "gear" and not pl.get("gap")
        if ref in by_id and (pl.get("side") in PORT_SIDES or meshing):
            node = ref
            continue
        return node if node != oid else None
    return None


def lift_trains(objects: list[dict[str, Any]], scene: CompiledScene) -> list[str]:
    """
    A pipe run or gear train whose wider parts go below the ground (flanges on a pipe lying on the
    floor, a big gear meshing with a small one): lift the run's first part so its lowest part
    stands on the ground. The parts are joined, so they can only move together.
    """
    by_id = {o["id"]: o for o in objects}
    depth: dict[str, float] = {}
    for oid, co in scene.objects.items():
        root = _train_root(oid, by_id) if co.built is not None and not by_id.get(oid, {}).get("parent") else None
        low = min((b[0][1] for b in _boxes(co)), default=0.0)
        if root and low < -TOL and not by_id[root].get("placement") and not by_id[root].get("parent"):
            depth[root] = max(depth.get(root, 0.0), -low)
    notes = []
    for root, d in depth.items():
        pos = list(vec3(by_id[root].get("position")))
        pos[1] = round(pos[1] + d, 4)
        by_id[root]["position"] = pos
        notes.append(f"Lifted '{root}' by {d:.3f} m so the parts joined to it stand on the ground, not in it.")
    return notes


def close_measured_gaps(objects: list[dict[str, Any]], scene: CompiledScene) -> list[str]:
    """
    Two things set side by side by coordinates with a gap exactly one of their widths (or half
    of one): a position measured from the wrong point, not a spacing anyone asked for. Seen with
    Qwen3-8B, "a riser flush at the casting's left end": the riser's centre at x = -0.3 instead of
    -0.2, leaving a gap the size of the riser. Move the later one over to touch.
    """
    by_id = {o["id"]: o for o in objects}
    loose = [o for o in objects if not o.get("placement") and not o.get("parent") and not o.get("pattern")
             and not any(vec3(o.get("rotation"))) and o["id"] in scene.objects
             and scene.objects[o["id"]].built is not None and len(scene.objects[o["id"]].instances) == 1]
    notes: list[str] = []
    for i, later in enumerate(loose):
        b = scene.objects[later["id"]].bbox
        for earlier in loose[:i]:
            a = scene.objects[earlier["id"]].bbox
            if abs(a[0][1] - b[0][1]) > TOL:
                continue  # not standing at the same level
            for axis, other in ((0, 2), (2, 0)):
                if abs((a[0][other] + a[1][other]) - (b[0][other] + b[1][other])) / 2 > TOL:
                    continue  # not lined up across
                gap = max(b[0][axis] - a[1][axis], a[0][axis] - b[1][axis])
                sizes = [a[1][axis] - a[0][axis], b[1][axis] - b[0][axis]]
                right = (b[0][axis] + b[1][axis]) > (a[0][axis] + a[1][axis])
                if gap < -TOL:
                    # Its centre put on the other's end face (Qwen3-8B: an ingate at x = 0.15, the casting's
                    # right end), half of it inside: push it out to touch, if it is the smaller one.
                    if not (sizes[1] < sizes[0] and abs(-gap - sizes[1] / 2) <= TOL):
                        continue
                    shift = -gap if right else gap
                    gap = -gap
                elif gap <= TOL or not any(abs(gap - s * f) <= TOL for s in sizes for f in (1, 0.5)):
                    continue
                else:
                    shift = -gap if right else gap
                pos = list(vec3(by_id[later["id"]].get("position")))
                pos[axis] = round(pos[axis] + shift, 6)
                by_id[later["id"]]["position"] = pos
                notes.append(f"Moved '{later['id']}' {gap:.3g} m to touch '{earlier['id']}' (its position was measured "
                             f"from the wrong point: {'half of it was inside' if shift * (1 if right else -1) > 0 else 'the gap was exactly a width'}).")
                break
    return notes


def repair(objects: list[dict[str, Any]], scene: CompiledScene) -> list[str]:
    """Apply every fix to `objects` in place. Returns notes; empty when nothing changed."""
    return ((drill_bolt_holes(objects, scene) + lift_trains(objects, scene)) or close_measured_gaps(objects, scene)
            or onto_the_end(objects, scene) or join_midpoint_boxes(objects, scene) or spread_on_top(objects, scene)
            or fit_supports(objects, scene) or close_gaps(objects, scene) or drop_standing_objects(objects, scene))
