"""
The scene_* tools. Every change is compiled before it is saved, so a bad call changes nothing
and the model gets a list of what to fix. Results are compact: ids, exact bounds, warnings.
"""

from __future__ import annotations

import asyncio
import copy
import json
import re
from typing import Any

import numpy as np

from oprel.integrations.scene3d import store
from oprel.integrations.scene3d.repair import repair
from oprel.integrations.scene3d.engine import CompiledScene, SceneError, compile_scene
from oprel.integrations.scene3d.engine.compiler import CompiledObject
from oprel.integrations.scene3d.engine.frame import vec3
from oprel.integrations.scene3d.engine.registry import all_shapes
from oprel.integrations.scene3d.engine import intent
from oprel.integrations.scene3d.engine.cuts import CUTTERS, THROUGH
from oprel.integrations.scene3d.engine.schema import PORT_SIDES, lean, object_schema, update_schema
from oprel.integrations.scene3d.engine.shapes.paths import PORTED
from oprel.tools.base import Tool, ToolContext, ToolError

MAX_OBJECTS_PER_CALL = 60
MAX_LISTED = 60  # objects described in one result (keeps it under the executor's size cap)
REMOVABLE = ["placement", "position", "rotation", "pattern", "cuts", "merge", "parent", "color", "opacity", "label"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _conversation(ctx: ToolContext) -> str:
    if not ctx.conversation_id:
        raise ToolError("The 3D scene needs a conversation to store the drawing in.")
    return ctx.conversation_id


def _round(v: np.ndarray) -> list[float]:
    return [round(float(x), 4) + 0.0 for x in v]  # + 0.0 turns -0.0 into 0.0


def _describe(co: CompiledObject, *, with_params: bool = False) -> dict[str, Any]:
    lo, hi = co.bbox
    d: dict[str, Any] = {"id": co.id, "shape": co.spec["shape"], "min": _round(lo), "max": _round(hi),
                         "size": _round(hi - lo)}
    if co.spec.get("parent"):
        d["parent"] = co.spec["parent"]
    if len(co.instances) > 1:
        d["copies"] = len(co.instances)
    if with_params:
        if co.spec.get("params"):
            d["params"] = co.spec["params"]
        d["position"] = _round(co.position)
        for key in ("rotation", "placement", "pattern", "color", "label"):
            if co.spec.get(key):
                d[key] = co.spec[key]
    return d


def _summary(verb: str, items: list[dict[str, Any]]) -> str:
    names = [f"{o['shape']} '{o['id']}'" for o in items[:8]]
    more = f" and {len(items) - 8} more" if len(items) > 8 else ""
    return f"{verb} {', '.join(names)}{more}" if names else f"{verb} nothing"


def _clean_id(raw: Any) -> str:
    text = re.sub(r"\s+", "_", str(raw or "").strip())
    return text.replace("#", "")[:64]


async def _compile(objects: list[dict[str, Any]]) -> CompiledScene:
    try:
        # CPU-bound: keep it off the event loop.
        return await asyncio.to_thread(compile_scene, objects)
    except SceneError as exc:
        raise ToolError("Nothing was changed. Fix these and call again: " + " | ".join(exc.errors)) from exc


async def _compile_and_repair(objects: list[dict[str, Any]]) -> tuple[CompiledScene, list[str]]:
    """Compile, apply the deterministic fixes in repair.py (editing `objects`), recompile."""
    compiled = await _compile(objects)
    notes: list[str] = []
    for _ in range(3):  # fitting legs can expose a gap, which the next round closes
        fixes = repair(objects, compiled)
        if not fixes:
            break
        notes += fixes
        compiled = await _compile(objects)
    return compiled, notes


def _result(version: int, scene: CompiledScene, touched: list[str], verb: str) -> dict[str, Any]:
    described = [_describe(scene.objects[i]) for i in touched if i in scene.objects][:MAX_LISTED]
    return {
        "scene_version": version,
        "summary": _summary(verb, described),
        "objects": described,
        "total_objects": len(scene.objects),
        "warnings": scene.warnings,
    }


def _descendants(ids: set[str], objects: list[dict[str, Any]]) -> set[str]:
    out = set(ids)
    changed = True
    while changed:
        changed = False
        for o in objects:
            if o["id"] not in out and o.get("parent") in out:
                out.add(o["id"])
                changed = True
    return out


# ---------------------------------------------------------------------------
# handlers
# ---------------------------------------------------------------------------

async def add_objects(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    conv = _conversation(ctx)
    new = args.get("objects") or []
    if not new:
        raise ToolError("'objects' is empty: give at least one object to add.")
    if len(new) > MAX_OBJECTS_PER_CALL:
        raise ToolError(f"{len(new)} objects in one call; the limit is {MAX_OBJECTS_PER_CALL}. Use patterns for repeats.")
    scene = store.get_scene(conv)
    existing = copy.deepcopy(scene["objects"])
    by_id = {o["id"]: o for o in existing}
    taken = set(by_id)

    # An object listed again with its existing id is an update, not a copy: models re-list
    # what is there to move it ("put the vase on the table") and then fumble the id error.
    added, updated, unchanged, edit_warnings = [], [], [], []
    word_notes: list[str] = []
    for raw in new:
        obj = copy.deepcopy(raw)
        if isinstance(obj, dict):
            note = intent.removal_to_hole(obj) or intent.normalize_shape(obj)
            if note:
                word_notes.append(note)
        oid = _clean_id(obj.get("id"))
        if oid in by_id:
            if obj.get("shape") and str(obj["shape"]).lower() != by_id[oid].get("shape"):
                raise ToolError(f"id '{oid}' is already a {by_id[oid].get('shape')}. Pick a new id for the "
                                f"{obj['shape']}, or delete '{oid}' first.")
            obj.pop("shape", None)
            _edit(by_id[oid], obj, unchanged, edit_warnings)
            updated.append(oid)
            continue
        twin = _same_spot_twin(obj, existing)
        if twin is not None:
            # Seen with Qwen3-8B: "make the bottom bigger" sent as a new vase on the old one's spot.
            obj.pop("shape", None)
            obj.pop("id", None)
            _edit(twin, obj, unchanged, edit_warnings)
            updated.append(twin["id"])
            continue
        obj["id"] = oid
        if oid:
            taken.add(oid)
        added.append(obj)
    for obj in added:
        obj["_unnamed"] = not obj["id"]
        if not obj["id"]:
            # Models often skip ids and then refer to "the table": name it after its shape.
            base = _clean_id(str(obj.get("shape") or "object").lower()) or "object"
            oid, n = base, 1
            while oid in taken:
                n += 1
                oid = f"{base}{n}"
            obj["id"] = oid
            taken.add(oid)

    wrap_notes = word_notes + await _holes_to_cuts(added, existing, updated)
    for obj in added:
        obj.pop("_hole", None)
    wrap_notes += _handles_from_cuts(added, taken) + _wrap_merged_shapes(added, taken)
    changed = added + [by_id[u] for u in updated]
    notes = wrap_notes + (_chain_references(added) + _join_runs(added, edit_warnings) + _resolve_references(changed, existing + added) + _adopt_orphans(added, existing)
             + _fill_merge_groups(added, existing))
    if updated:
        notes.insert(0, f"{', '.join(repr(u) for u in updated)} {'was' if len(updated) == 1 else 'were'} "
                        f"already there, so {'it was' if len(updated) == 1 else 'they were'} updated "
                        f"instead of adding a copy.")
    added, dropped = await _drop_broken(existing, added)
    objects = existing + added
    try:
        compiled, fixes = await _compile_and_repair(objects)
    except ToolError as exc:
        raise ToolError(f"{exc} (ids in the scene: {', '.join(o['id'] for o in objects)})") from exc
    notes += fixes
    edit_warnings += dropped
    version = store.save_scene(conv, objects)
    result = _result(version, compiled, [o["id"] for o in added] + updated, "Added" if added else "Updated")
    if notes:
        result["notes"] = notes
    result["warnings"] = (_unchanged_warnings(unchanged) + edit_warnings
                          + _hole_like_warnings(added, compiled) + result["warnings"])
    return result


_REF_ID = re.compile(r"^'([^']+)'")
# The size a cut leaves out to go right through along each axis (cuts.THROUGH, per direction).
_THROUGH_AXIS = {"box": {0: "width", 1: "height", 2: "depth"}, "cylinder": {1: "height"}, "extrude": {1: "thickness"}}
# Things that are taken away, not added: an object named like this that overlaps another is a cut.
_OPENING_WORDS = re.compile(r"hole|slot|opening|cut.?out|recess|pocket|groove|bore|sink|window|doorway|vent|notch",
                            re.IGNORECASE)


def _references(obj: dict[str, Any]) -> set[str]:
    pl = obj.get("placement") if isinstance(obj.get("placement"), dict) else {}
    refs = {obj.get("parent"), pl.get("relative_to"), (obj.get("pattern") or {}).get("relative_to")}
    refs |= set(pl.get("between") or [])
    return {r for r in refs if isinstance(r, str)}


async def _drop_broken(existing: list[dict[str, Any]], added: list[dict[str, Any]]) -> tuple[list, list[str]]:
    """
    Draw what can be drawn. One broken object used to reject the whole call; the model then tried
    to update objects that were never saved ("no object with id ...") and lost most of its rounds
    (seen with Qwen3-8B, a kitchen counter with a sink, a vase and a mug). Leave out the new
    objects the engine names in its errors (and those that hang on them), and say so; errors that
    belong to no new object still reject the call.
    """
    left, notes = list(added), []
    for _ in range(len(added)):
        try:
            await asyncio.to_thread(compile_scene, existing + left)
            return left, notes
        except SceneError as exc:
            new_ids = {o["id"] for o in left}
            bad: dict[str, str] = {}
            for err in exc.errors:
                m = _REF_ID.match(err)
                if m and m.group(1) in new_ids:
                    bad.setdefault(m.group(1), err)
            if not bad or len(bad) == len(left) or any(not _REF_ID.match(e) for e in exc.errors):
                return left, notes  # nothing to keep apart: the normal error path reports it all
            gone = set(bad)
            changed = True
            while changed:  # and whatever hangs on them
                changed = False
                for o in left:
                    hangs = _references(o) & gone
                    if o["id"] not in gone and hangs:
                        bad[o["id"]] = f"'{o['id']}' needs '{next(iter(hangs))}', which was not added"
                        gone.add(o["id"])
                        changed = True
            left = [o for o in left if o["id"] not in gone]
            notes += [f"NOT ADDED: {err}. Fix it and add it again; the rest was drawn." for err in bad.values()]
    return left, notes


async def _holes_to_cuts(added: list[dict[str, Any]], existing: list[dict[str, Any]],
                         updated: list[str]) -> list[str]:
    """
    A hole drawn as an object ("sink_hole": a cylinder, sometimes cut away from itself or merged)
    is a cut in the object it goes into. Seen with Qwen3-8B: a counter's sink as its own cylinder
    whose cuts removed all of it, rejected three times. Any shape a cut can be, named like an
    opening (hole, slot, opening, cutout, recess, sink, window...), that overlaps another object:
    it becomes a cut of that object at the same place, right through if it spans it top to bottom.
    """
    holes = [o for o in added if o.get("shape") in CUTTERS and not o.get("pattern")
             and (o.pop("_hole", False) or _OPENING_WORDS.search(f"{o.get('id', '')} {o.get('label', '')}"))]
    if not holes:
        return []
    hole_ids = {o["id"] for o in holes}
    # A hole with no height (no box size) means right through: a stand-in size to lay the scene out.
    open_ended = {o["id"] for o in holes if any(k not in (o.get("params") or {}) for k in THROUGH.get(o["shape"], ()))}

    def stand_in(o: dict[str, Any]) -> dict[str, Any]:
        out = {k: v for k, v in o.items() if k not in ("cuts", "merge")}
        out["params"] = {**{k: 0.01 for k in THROUGH.get(o["shape"], ())}, **(o.get("params") or {})}
        return out

    trial = [stand_in(o) if o["id"] in hole_ids else o for o in existing + added]
    try:
        scene = await asyncio.to_thread(compile_scene, trial)
    except SceneError:
        return []  # can't tell where things are: the errors and warnings will speak about it
    notes: list[str] = []
    every = {o["id"]: o for o in existing + added}
    for hole in holes:
        co = scene.objects.get(hole["id"])
        if co is None or co.built is None or any(vec3(hole.get("rotation"))):
            continue
        hlo, hhi = co.bbox
        best, best_overlap = None, 0.0
        for oid, other in scene.objects.items():
            if oid in hole_ids or other.built is None or len(other.instances) != 1:
                continue
            olo, ohi = other.bbox
            overlap = float(np.prod(np.clip(np.minimum(hhi, ohi) - np.maximum(hlo, olo), 0, None)))
            if overlap > best_overlap:
                best, best_overlap = oid, overlap
        named = _named_host(hole, scene)
        if best is None and named is None:
            continue
        if best is None or (named is not None and (named != best or hole["id"] in open_ended)):
            # Not overlapping what it names (Qwen3-8B put "sink_hole" "left" of the counter): it is a
            # hole in the object it names, at that object's middle (plus any sideways offset).
            best = named
            hlo, hhi = _hole_in(hole, scene.objects[best], hhi - hlo)
        host_co, host = scene.objects[best], every[best]
        olo, ohi = host_co.bbox
        local = np.linalg.inv(host_co.instances[0]) @ np.append((hlo + hhi) / 2, 1.0)
        params = dict(hole.get("params") or {})
        # Spanning the host along an axis (a sink top to bottom, a window front to back): leave that
        # size out, and the cut goes right through that way.
        spans = [a for a in range(3) if hlo[a] <= olo[a] + 1e-3 and hhi[a] >= ohi[a] - 1e-3]
        through = False
        for a in spans:
            size = _THROUGH_AXIS.get(hole["shape"], {}).get(a)
            if size and size in params:
                params.pop(size)
                through = True
        host["cuts"] = list(host.get("cuts") or []) + [
            {"shape": hole["shape"], "params": params, "position": [round(float(x), 4) for x in local[:3]]}]
        added.remove(hole)
        if any(o is host for o in existing) and best not in updated:
            updated.append(best)
        notes.append(f"Made '{hole['id']}' a hole in '{best}'{' (right through)' if through else ''} instead of a "
                     f"separate solid.")
    return notes


def _named_host(hole: dict[str, Any], scene: CompiledScene) -> str | None:
    """The object a hole names as its place (placement.relative_to, or its parent), if it has a mesh."""
    pl = hole.get("placement") if isinstance(hole.get("placement"), dict) else {}
    for ref in (pl.get("relative_to"), hole.get("parent")):
        co = scene.objects.get(ref) if isinstance(ref, str) else None
        if co is not None and co.built is not None and len(co.instances) == 1:
            return ref
    return None


def _hole_in(hole: dict[str, Any], host: CompiledObject, size: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Bounds of a hole of `size` put in `host` where it names it: right through from front to back
    for side front/back, otherwise down through the top; sideways offsets kept, centred otherwise.
    """
    olo, ohi = host.bbox
    pl = hole.get("placement") if isinstance(hole.get("placement"), dict) else {}
    off = np.array(vec3(pl.get("offset")))
    centre = (olo + ohi) / 2
    across = 2 if pl.get("side") in ("front", "back") else 1
    for a in range(3):
        if a != across:
            centre[a] += off[a]
    lo, hi = centre - size / 2, centre + size / 2
    if not (hole.get("params") or {}).get("height") or across == 2:
        lo[across], hi[across] = olo[across], ohi[across]  # right through
    elif pl.get("side") == "top":
        hi[1] = ohi[1] + 1e-4  # a pocket down from the top
        lo[1] = hi[1] - size[1]
    return lo, hi


def _same_spot_twin(obj: dict[str, Any], existing: list[dict[str, Any]]) -> dict[str, Any] | None:
    """
    The one existing object this 'new' one would sit exactly on top of: same shape, same parent,
    same position, neither placed relatively or patterned. Two identical things in the same spot
    are never wanted, so the 'add' is really an edit of that object.
    """
    if obj.get("placement") or obj.get("pattern"):
        return None
    shape_, pos = str(obj.get("shape", "")).lower(), np.array(vec3(obj.get("position")))
    matches = [e for e in existing
               if e.get("shape") == shape_ and e.get("parent") == obj.get("parent")
               and not e.get("placement") and not e.get("pattern")
               and np.allclose(np.array(vec3(e.get("position"))), pos, atol=0.01)]
    return matches[0] if len(matches) == 1 else None


def _chain_references(added: list[dict[str, Any]]) -> list[str]:
    """
    In a run of unnamed parts joined end to end, "relative_to": "elbow" means the elbow just
    before it, not the first one. Seen with Qwen3-8B: pipe, elbow, pipe, elbow, pipe without ids,
    each joined to "pipe" / "elbow", so the second elbow landed on the first pipe's end too.
    """
    notes: list[str] = []
    for i, obj in enumerate(added):
        unnamed = obj.pop("_unnamed", False)
        pl = obj.get("placement")
        if not unnamed or not isinstance(pl, dict) or pl.get("side") not in PORT_SIDES:
            continue
        ref = str(pl.get("relative_to", "")).strip().lower()
        earlier = [o for o in added[:i] if str(o.get("shape", "")).lower() == ref]
        if len(earlier) > 1 and pl["relative_to"] != earlier[-1]["id"]:
            notes.append(f"Joined '{obj['id']}' to '{earlier[-1]['id']}', the {ref} just before it.")
            pl["relative_to"] = earlier[-1]["id"]
    for obj in added:
        obj.pop("_unnamed", None)
    return notes


def _join_runs(added: list[dict[str, Any]], warnings: list[str]) -> list[str]:
    """
    Pipes, elbows, ... listed one after another with no position or placement are a run: join each
    to the end of the one before. Seen with Qwen3-8B: a 5-part pipe run all drawn at the origin.

    Only when the list reads as a run. Two straight pipes in a row with nothing between them means
    the parts were listed by kind (all pipes, then all elbows; seen with Qwen3-8B), and joining them
    in list order made one long straight pipe with the fittings piled at its end: say how to join
    them instead.
    """
    loose = [(prev, obj) for prev, obj in zip(added, added[1:])
             if obj.get("shape") in PORTED and prev.get("shape") in PORTED and not obj.get("placement")
             and not any(vec3(obj.get("position"))) and not obj.get("pattern")
             and obj.get("parent") == prev.get("parent")]
    if any(prev.get("shape") == obj.get("shape") == "pipe" for prev, obj in loose):
        names = ", ".join(repr(o["id"]) for o in added if o.get("shape") in PORTED)
        warnings.append(f"{names} are not joined. In a pipe run every part after the first is joined to the "
                        f"part before it along the run: \"placement\": {{\"relative_to\": \"<previous part>\", "
                        f"\"side\": \"end\"}} (a tee's branch: \"side\": \"branch\"; an elbow's turn: \"toward\").")
        return []
    notes: list[str] = []
    for prev, obj in loose:
        obj.pop("position", None)
        obj["placement"] = {"relative_to": prev["id"], "side": "end"}
        notes.append(f"Joined '{obj['id']}' to the end of '{prev['id']}'.")
    return notes


_WORLD_WORDS = {"scene", "ground", "floor", "world", "origin", "center", "centre"}


def _resolve_references(added: list[dict[str, Any]], objects: list[dict[str, Any]]) -> list[str]:
    """
    Read a reference that isn't an id as "the one object of that shape" (or with that label):
    models write relative_to: "table" for an object they didn't name, or that got id table2.
    """
    ids = {o["id"] for o in objects}

    def resolve(ref: Any) -> str | None:
        if not isinstance(ref, str) or ref in ids:
            return None
        key = ref.strip().lower()
        matches = [o["id"] for o in objects
                   if str(o.get("shape", "")).lower() == key or str(o.get("label", "")).strip().lower() == key]
        return matches[0] if len(matches) == 1 else None

    notes = []
    for obj in added:
        pl = obj.get("placement")
        if isinstance(pl, dict) and str(pl.get("relative_to", "")).strip().lower() in _WORLD_WORDS \
                and pl.get("relative_to") not in ids:
            # "relative to the scene/ground" just means "stand it in the scene": use a position.
            obj.pop("placement")
            obj.setdefault("position", [0, 0, 0])
            notes.append(f"Placed '{obj['id']}' in the scene at {obj['position']} "
                         f"('{pl.get('relative_to')}' is not an object).")
        for holder, key in ((obj, "parent"), (obj.get("placement"), "relative_to"), (obj.get("pattern"), "relative_to")):
            if not isinstance(holder, dict):
                continue
            target = resolve(holder.get(key))
            if target and target != obj["id"]:
                notes.append(f"Read '{holder[key]}' as '{target}' for '{obj['id']}'.")
                holder[key] = target
        between = (obj.get("placement") or {}).get("between")
        if isinstance(between, list):
            for i, ref in enumerate(between):
                target = resolve(ref)
                if target and target != obj["id"]:
                    notes.append(f"Read '{ref}' as '{target}' for '{obj['id']}'.")
                    between[i] = target
    return notes


def merge_scene(objects: list[dict[str, Any]], group: str | None, merge: bool) -> list[str]:
    """
    Turn merging on or off (in place). Used by the canvas's Merge button and by the tools.
      merge on,  group given      that group becomes one solid
      merge on,  no group         everything becomes one solid: the single top-level group if there
                                  is one, otherwise a new merged group "part" wrapping all top-level objects
      merge off, group given      that group's parts are separate again
      merge off, no group         every group's parts are separate again
    Returns notes for the caller. Raises ToolError for a group id that isn't a group.
    """
    by_id = {o["id"]: o for o in objects}
    if group is not None:
        target = by_id.get(group)
        if target is None or target.get("shape") != "group":
            groups = [i for i, o in by_id.items() if o.get("shape") == "group"]
            raise ToolError(f"'{group}' is not a group in the scene; merge works on groups"
                            + (f" ({', '.join(groups)})." if groups else ", or on everything at once."))
        if merge:
            target["merge"] = True
            return [f"'{group}' is now one solid."]
        target.pop("merge", None)
        return [f"'{group}' is separate parts again."]
    if not merge:
        names = [o["id"] for o in objects if o.pop("merge", None)]
        return [f"{', '.join(repr(n) for n in names)} {'is' if len(names) == 1 else 'are'} separate parts again."] \
            if names else []
    tops = [o for o in objects if not o.get("parent") or o["parent"] not in by_id]
    if len(tops) == 1 and tops[0].get("shape") == "group":
        tops[0]["merge"] = True
        return [f"'{tops[0]['id']}' is now one solid."]
    gid, n = "part", 1
    while gid in by_id:
        n += 1
        gid = f"part{n}"
    for o in tops:
        o["parent"] = gid
    objects.insert(0, {"id": gid, "shape": "group", "merge": True})
    return [f"Merged {', '.join(repr(o['id']) for o in tops)} into one solid, group '{gid}'."]


_VESSELS_WITH_HANDLES = {"cup", "bowl", "vase"}


def _handles_from_cuts(added: list[dict[str, Any]], taken: set[str]) -> list[str]:
    """
    Handles listed as cuts (seen with Qwen3-8B, "a curvy vase with smooth handles on both sides"):
    a handle is added, not cut away. On a cup, bowl or vase they become its own "handles"; on
    anything else, handle objects placed on its sides.
    """
    notes: list[str] = []
    for obj in list(added):
        cuts_ = obj.get("cuts")
        if not isinstance(cuts_, list):
            continue
        grips = [c for c in cuts_ if isinstance(c, dict) and str(c.get("shape", "")).lower() == "handle"]
        if not grips:
            continue
        kept = [c for c in cuts_ if c not in grips]
        if kept:
            obj["cuts"] = kept
        else:
            obj.pop("cuts")
        style = next((str((c.get("params") or {}).get("style")) for c in grips
                      if (c.get("params") or {}).get("style") in ("round", "box")), "round")
        count = min(2, len(grips) * max(1, int((grips[0].get("pattern") or {}).get("count", 1))))
        if str(obj.get("shape", "")).lower() in _VESSELS_WITH_HANDLES:
            obj["params"] = {**(obj.get("params") or {}), "handles": count, "handle_style": style}
            notes.append(f"Gave '{obj['id']}' {count} handle{'s' if count > 1 else ''} (handles are added, not cut).")
            continue
        for side in ("right", "left")[:count]:
            hid, n = f"{obj['id']}_handle", 1
            while hid in taken:
                n += 1
                hid = f"{obj['id']}_handle{n}"
            taken.add(hid)
            handle = {"id": hid, "shape": "handle", "params": {"style": style},
                      "placement": {"relative_to": obj["id"], "side": side}}
            if obj.get("parent"):
                handle["parent"] = obj["parent"]
            added.insert(added.index(obj) + 1, handle)
            notes.append(f"Added '{hid}' on the {side} of '{obj['id']}' (handles are added, not cut).")
    return notes


def _wrap_merged_shapes(added: list[dict[str, Any]], taken: set[str]) -> list[str]:
    """
    "merge": true on a shape (not a group) means "this and its parts are one solid". Seen with
    Qwen3-8B: a bowl with merge and two handles parented to it, all rejected. Put the shape in a new
    merged group; its parts come along, since they hang off it.
    """
    notes: list[str] = []
    for obj in list(added):
        if not obj.get("merge") or str(obj.get("shape", "")).lower() == "group":
            continue
        gid, n = f"{obj['id']}_solid", 1
        while gid in taken:
            n += 1
            gid = f"{obj['id']}_solid{n}"
        taken.add(gid)
        group = {"id": gid, "shape": "group", "merge": True}
        if obj.get("parent"):
            group["parent"] = obj["parent"]
        obj.pop("merge")
        obj["parent"] = gid
        added.insert(added.index(obj), group)
        notes.append(f"Put '{obj['id']}' and its parts in merged group '{gid}': one solid.")
    return notes


def _fill_merge_groups(added: list[dict[str, Any]], existing: list[dict[str, Any]]) -> list[str]:
    """
    A new merged group with no parts means "merge what is drawn" ("make it one solid part"): put
    every existing top-level object in it. Saves a small model one update per object.
    """
    notes: list[str] = []
    everything = existing + added
    for g in added:
        if g.get("shape") != "group" or not g.get("merge") or any(o.get("parent") == g["id"] for o in everything):
            continue
        tops = [o for o in existing if not o.get("parent") and o["id"] != g["id"]]
        for o in tops:
            o["parent"] = g["id"]
        if tops:
            notes.append(f"Put {', '.join(repr(o['id']) for o in tops)} in merged group '{g['id']}': one solid now.")
    return notes


def _adopt_orphans(added: list[dict[str, Any]], existing: list[dict[str, Any]]) -> list[str]:
    """
    Small models often create a group and then forget `parent` on every part (seen with
    Qwen3-4B/8B). When a call adds exactly one group and nothing uses it, the call's
    parentless parts are clearly meant to be inside it.
    """
    groups = [o for o in added if str(o.get("shape", "")).lower() == "group"]
    if len(groups) != 1:
        return []
    gid = groups[0]["id"]
    if any(o.get("parent") == gid for o in existing + added):
        return []
    orphans = [o for o in added if o is not groups[0] and not o.get("parent")]
    for o in orphans:
        o["parent"] = gid
    return [f"Put {', '.join(repr(o['id']) for o in orphans)} in group '{gid}'."] if orphans else []


def _normalized(obj: dict[str, Any]) -> dict[str, Any]:
    """For comparing: a missing position or rotation is the same as zeros."""
    out = copy.deepcopy(obj)
    for key in ("position", "rotation"):
        out[key] = vec3(out.get(key))
    return out


def _edit(obj: dict[str, Any], change: dict[str, Any], unchanged: list[str], warnings: list[str]) -> None:
    """Apply a change, noting an object that didn't change and earlier cuts the change threw away."""
    old_cuts = list(obj.get("cuts") or [])
    if not _apply_change(obj, change):
        unchanged.append(obj["id"])
    new_cuts = obj.get("cuts") or []

    def kind(c: dict[str, Any]) -> tuple[Any, Any]:
        return c.get("shape"), (c.get("pattern") or {}).get("type")

    # A new cut of the same kind is the old one moved or resized, not dropped.
    dropped = [c for c in old_cuts if kind(c) not in {kind(n) for n in new_cuts}]
    # Seen with Qwen3-8B: "add 4 screw holes" sent only the 4 new holes, losing the one in the middle.
    # A cleared list ("remove": ["cuts"] or []) is deliberate.
    if dropped and new_cuts:
        shapes = ", ".join(str(c.get("shape")) for c in dropped)
        warnings.append(
            f"'{obj['id']}': the new cuts list replaced {len(dropped)} earlier cut(s) ({shapes}), which are gone now. "
            f"If the user only asked to add cuts, send the whole list again: the earlier cuts plus the new ones."
        )


def _apply_change(obj: dict[str, Any], change: dict[str, Any]) -> bool:
    """Merge a change into an object: only the given fields change; params are merged.
    Returns False when the object ended up exactly as it was."""
    before = _normalized(obj)
    _merge(obj, change)
    return _normalized(obj) != before


def _same_reference(old: Any, new: dict[str, Any]) -> bool:
    """Is a placement change about the same reference (or names none), so it adjusts the old one?"""
    if not isinstance(old, dict):
        return False
    if "between" in new:
        return new["between"] == old.get("between")
    if "between" in old:
        return "relative_to" not in new  # e.g. {"vertical": "top"} for a beam already between two posts
    return new.get("relative_to") in (None, old.get("relative_to"))


def _merge(obj: dict[str, Any], change: dict[str, Any]) -> None:
    for key in change.get("remove") or []:
        obj.pop(key, None)
    same_shape = change.get("shape") in (None, obj.get("shape"))
    for key, value in change.items():
        if key in ("id", "remove"):
            continue
        if key == "params" and isinstance(value, dict) and same_shape:
            obj["params"] = {**(obj.get("params") or {}), **value}
        elif key == "placement" and isinstance(value, dict) and _same_reference(obj.get("placement"), value):
            # "more space" sent as {"gap": 0.4} alone (seen with Qwen3-8B) keeps the side and reference.
            obj["placement"] = {**obj["placement"], **value}
        elif key == "parent" and value == "":
            obj.pop("parent", None)
        else:
            obj[key] = value
    # Moving an object one way replaces the other way of positioning it.
    if "position" in change and "placement" not in change:
        obj.pop("placement", None)
    if "placement" in change and "position" not in change:
        obj.pop("position", None)
    # A box moved "between" two objects should reach both: drop the width it had, so the engine
    # sizes it (seen with Qwen3-8B: the old 14 cm width kept the plate from joining). A width in
    # the same change is kept.
    if (obj.get("shape") == "box" and (change.get("placement") or {}).get("between")
            and "width" not in (change.get("params") or {})):
        (obj.get("params") or {}).pop("width", None)


async def update_objects(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    conv = _conversation(ctx)
    updates = args.get("updates") or []
    if not updates:
        raise ToolError("'updates' is empty.")
    if len(updates) > MAX_OBJECTS_PER_CALL:
        raise ToolError(f"{len(updates)} updates in one call; the limit is {MAX_OBJECTS_PER_CALL}.")
    objects = copy.deepcopy(store.get_scene(conv)["objects"])
    by_id = {o["id"]: o for o in objects}

    touched: list[str] = []
    unchanged: list[str] = []
    edit_warnings: list[str] = []
    merge_notes: list[str] = []
    for i, change in enumerate(updates):
        oid = change.get("id")
        if oid not in by_id:
            known = ", ".join(list(by_id)[:30]) or "none"
            # Seen with Qwen3-8B: after an add was refused, it "fixed" the refused objects by updating them.
            raise ToolError(f"updates[{i}]: no object with id '{oid}' (existing ids: {known}). If it was in a call "
                            f"that was refused, it was never added: add it with scene_add_objects.")
        if "merge" in change and by_id[oid].get("shape") != "group":
            # "merge" on a single object can only mean "make it all one solid".
            change = {k: v for k, v in change.items() if k != "merge"}
            merge_notes += merge_scene(objects, None, bool(updates[i]["merge"]))
            if len(change) == 1:  # nothing else to change on it
                touched.append(oid)
                continue
        change, dropped = _without_example_cuts(change, by_id[oid])
        merge_notes += dropped
        _edit(by_id[oid], change, unchanged, edit_warnings)
        touched.append(oid)

    compiled, notes = await _compile_and_repair(objects)
    notes = merge_notes + notes
    version = store.save_scene(conv, objects)
    result = _result(version, compiled, touched, "Updated")
    if notes:
        result["notes"] = notes
    result["warnings"] = _unchanged_warnings(unchanged) + edit_warnings + result["warnings"]
    return result


# The cuts of the plate example in the system hint (scene3d/__init__.py; a test keeps them in step).
HINT_EXAMPLE_CUTS = [{"shape": "cylinder", "params": {"radius": 0.03}},
                     {"shape": "cylinder", "params": {"radius": 0.004},
                      "pattern": {"type": "grid", "count": 2, "rows": 2, "spacing": [0.24, 0.24]}}]


def _example_like(cut: Any) -> bool:
    """One of the hint's example cuts, give or take a size it added (Qwen3-8B added a height of 0.1)."""
    if not isinstance(cut, dict):
        return False
    for ex in HINT_EXAMPLE_CUTS:
        params = {k: v for k, v in (cut.get("params") or {}).items() if k != "height"}
        if cut.get("shape") == ex["shape"] and params == ex["params"] and cut.get("pattern") == ex.get("pattern"):
            return True
    return False


def _without_example_cuts(change: dict[str, Any], obj: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """
    An edit that brings in the hint's example holes word for word, on an object that had no cuts:
    copied from the instructions, not asked for. Seen with Qwen3-8B: "I don't need a handle for the
    bowl" sent handles 0 plus the example plate's five holes, drilling them into the bowl.
    """
    cuts_ = change.get("cuts")
    if not isinstance(cuts_, list) or obj.get("cuts") or not cuts_ or not all(_example_like(c) for c in cuts_):
        return change, []
    return ({k: v for k, v in change.items() if k != "cuts"},
            [f"Left out the cuts sent for '{obj['id']}': they are the example from the instructions, not "
             f"something the user asked for. Add holes only when asked."])


_HOLE_WORDS = re.compile(r"hole|slot|opening|cut.?out|bore", re.IGNORECASE)
_PANEL_WORDS = re.compile(r"window|door", re.IGNORECASE)


def _hole_like_warnings(added: list[dict[str, Any]], compiled: CompiledScene) -> list[str]:
    """
    Solid boxes standing in for holes. Seen with Qwen3-8B: "add 4 screw holes" drew a box labelled
    "screw_holes" next to the plate; "a wall with a window and a door" drew solid window and door
    boxes inside the wall. A window or door is only flagged inside another object: a door panel
    on its own is a real thing.
    """
    out = []
    for o in added:
        name = f"{o.get('id', '')} {o.get('label', '')}"
        if o.get("shape") not in ("box", "cylinder") or o["id"] not in compiled.objects:
            continue
        lo, hi = compiled.objects[o["id"]].bbox
        mid = (lo + hi) / 2
        host = next((oid for oid, co in compiled.objects.items()
                     if oid != o["id"] and not co.spec.get("parent") and co.built is not None
                     and np.all(co.bbox[0] <= mid) and np.all(mid <= co.bbox[1])), None)
        if _HOLE_WORDS.search(name) and len(compiled.objects) > 1:
            target = f"'{host}'" if host else "the object it goes through"
        elif _PANEL_WORDS.search(name) and host:
            target = f"'{host}'"
        else:
            continue
        out.append(f"'{o['id']}' is a solid {o['shape']} named like an opening. If it is meant as a hole, delete "
                   f"'{o['id']}' and make it a cut of {target} (send its whole cuts list).")
    return out


def _unchanged_warnings(ids: list[str]) -> list[str]:
    # Seen with Qwen3-8B: resent the old values for "make the bottom bigger", then said it was done.
    return [f"'{oid}' did not change: every value you sent is what it already had. Apply what the user "
            f"asked (new numbers) or tell them you didn't change it." for oid in ids]


async def delete_objects(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    conv = _conversation(ctx)
    objects = store.get_scene(conv)["objects"]
    if args.get("all"):
        version = store.save_scene(conv, [])
        return {"scene_version": version, "summary": f"Cleared the scene ({len(objects)} objects)",
                "deleted": [o["id"] for o in objects], "total_objects": 0, "warnings": []}

    ids = set(args.get("ids") or [])
    if not ids:
        raise ToolError("Give 'ids' to delete, or all=true to clear the scene.")
    unknown = ids - {o["id"] for o in objects}
    if unknown:
        raise ToolError(f"No objects with id(s): {', '.join(sorted(unknown))}")
    gone = _descendants(ids, objects)  # parts go with their group

    # Objects placed next to a deleted one keep their current spot.
    current = await _compile(objects)
    kept = []
    for o in objects:
        if o["id"] in gone:
            continue
        o = copy.deepcopy(o)
        pl = o.get("placement") or {}
        if pl.get("relative_to") in gone or set(pl.get("between") or []) & gone:
            co = current.objects[o["id"]]
            o["position"] = _round(co.position)
            if pl.get("between"):
                # Freeze what `between` worked out: its turn and (for a spanning box) its width.
                o["params"] = {**(o.get("params") or {}), **{k: v for k, v in co.spec["params"].items()
                                                              if k == "width"}}
                if not any(o.get("rotation") or []):
                    m = co.local
                    o["rotation"] = [0, round(float(np.degrees(np.arctan2(-m[2, 0], m[0, 0]))), 3), 0]
            o.pop("placement")
        kept.append(o)

    compiled = await _compile(kept)
    version = store.save_scene(conv, kept)
    return {"scene_version": version, "summary": f"Deleted {', '.join(sorted(gone))}",
            "deleted": sorted(gone), "total_objects": len(kept), "warnings": compiled.warnings}


async def get_scene(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    conv = _conversation(ctx)
    scene = store.get_scene(conv)
    if not scene["objects"]:
        return {"scene_version": scene["version"], "total_objects": 0, "objects": [], "warnings": [],
                "summary": "The scene is empty"}
    compiled = await _compile(scene["objects"])
    listed = [_describe(co, with_params=True) for co in list(compiled.objects.values())[:MAX_LISTED]]
    out = {"scene_version": scene["version"], "total_objects": len(compiled.objects), "objects": listed,
           "warnings": compiled.warnings, "summary": f"{len(compiled.objects)} objects"}
    if len(compiled.objects) > MAX_LISTED:
        out["note"] = f"Only the first {MAX_LISTED} objects are listed."
    return out


MAX_IN_SUMMARY = 25


def _placement_words(pl: dict[str, Any] | None) -> str:
    """
    How an object is placed, for the summary: " left of big gap 0.15". Seen with Qwen3-8B: asked
    for "more space between the cylinders", it overwrote a neat gap with guessed coordinates
    because it couldn't see the gap was there to change.
    """
    if not isinstance(pl, dict):
        return ""
    if pl.get("between"):
        out = f" between {' and '.join(map(str, pl['between']))}"
    elif pl.get("relative_to"):
        out = f" {pl.get('side', 'center')} of {pl['relative_to']}"
    else:
        return ""
    if pl.get("gap"):
        out += f" gap {pl['gap']:g}"
    if pl.get("align"):
        out += f" align {pl['align']}"
    if pl.get("toward"):
        out += f" toward {pl['toward']}"
    return out


def scene_summary(conversation_id: str | None) -> str:
    """
    What is already drawn, for the system prompt. Without it a follow-up ("make the bottom
    smaller") can't see the earlier tool calls and draws a second copy instead of editing.
    """
    if not conversation_id:
        return ""
    objects = store.get_scene(conversation_id)["objects"]
    if not objects:
        return "The 3D scene is empty."
    try:
        compiled = compile_scene(objects)
    except SceneError:
        compiled = None
    lines = []
    for o in objects[:MAX_IN_SUMMARY]:
        line = f"- {o['id']}: {o['shape']}"
        if o.get("params"):
            line += " " + json.dumps(o["params"], separators=(",", ":"))
        if o.get("cuts"):
            # Updating cuts replaces the list, so the model needs the current one to add to it.
            line += " cuts " + json.dumps(o["cuts"], separators=(",", ":"))
        if o.get("parent"):
            line += f" in {o['parent']}"
        line += _placement_words(o.get("placement"))
        if o.get("merge"):
            line += " merged (one solid)"
        if compiled and o["id"] in compiled.objects:
            lo, hi = compiled.objects[o["id"]].bbox
            line += f" min {_round2(lo)} max {_round2(hi)}"
        lines.append(line)
    if len(objects) > MAX_IN_SUMMARY:
        lines.append(f"- ... and {len(objects) - MAX_IN_SUMMARY} more (scene_get_scene lists all)")
    return ("Already in the 3D scene. To change one of these, call scene_update_objects with its id "
            "(e.g. new params, a new gap for more or less space, or holes as its cuts); do NOT add it again. "
            "Add only new things:\n" + "\n".join(lines))


def _round2(v: np.ndarray) -> list[float]:
    return [round(float(x), 2) + 0.0 for x in v]


# ---------------------------------------------------------------------------
# definitions
# ---------------------------------------------------------------------------

def shape_catalogue() -> str:
    return "\n".join(f"- {s.signature()}" + (f": {s.description}" if s.description else "") for s in all_shapes())


def build_tools() -> list[Tool]:
    # Descriptions stay short: every token here is resent on each model turn (see engine/schema.py).
    item = lean(object_schema())
    # The description lists every shape with its params; an enum of the names repeats ~100 tokens.
    item["properties"]["shape"] = {"type": "string"}
    return [
        Tool(
            name="scene_add_objects",
            title="Draw in the 3D scene",
            kind="write",
            description=(
                "Add objects to the 3D scene, a whole request in one call. Sizes in metres. "
                "Shapes (? = optional):\n" + shape_catalogue()
            ),
            parameters={
                "type": "object",
                "properties": {
                    "plan": {"type": "string"},
                    "objects": {"type": "array", "items": item},
                },
                "required": ["objects"],
            },
            handler=add_objects,
            compact_history=True,
        ),
        Tool(
            name="scene_update_objects",
            title="Change objects in the 3D scene",
            kind="write",
            description=(
                "Change objects by id; only the given fields change (params are merged). "
                "parent \"\" detaches; remove clears fields."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "updates": {"type": "array", "items": lean(update_schema(REMOVABLE))},
                },
                "required": ["updates"],
            },
            handler=update_objects,
            compact_history=True,
        ),
        Tool(
            name="scene_delete_objects",
            title="Delete from the 3D scene",
            kind="write",
            description="Delete objects by id (with their parts), or all=true to clear the scene.",
            parameters={
                "type": "object",
                "properties": {
                    "ids": {"type": "array", "items": {"type": "string"}, "maxItems": 300},
                    "all": {"type": "boolean"},
                },
            },
            handler=delete_objects,
            compact_history=True,
        ),
        Tool(
            name="scene_get_scene",
            title="Read the 3D scene",
            kind="read",
            description="List the scene's objects with params, position and exact min/max bounds.",
            parameters={"type": "object", "properties": {}},
            handler=get_scene,
            compact_history=True,
        ),
    ]
