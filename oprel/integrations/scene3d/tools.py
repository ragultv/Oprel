"""
The scene_* tools. Every change is compiled before it is saved, so a bad call changes nothing
and the model gets a list of what to fix. Results are compact: ids, exact bounds, warnings.
"""

from __future__ import annotations

import asyncio
import copy
import re
from typing import Any

import numpy as np

from oprel.integrations.scene3d import store
from oprel.integrations.scene3d.repair import repair
from oprel.integrations.scene3d.engine import CompiledScene, SceneError, compile_scene
from oprel.integrations.scene3d.engine.compiler import CompiledObject
from oprel.integrations.scene3d.engine.registry import all_shapes
from oprel.integrations.scene3d.engine.schema import object_schema, update_schema
from oprel.tools.base import Tool, ToolContext, ToolError

MAX_OBJECTS_PER_CALL = 60
MAX_LISTED = 60  # objects described in one result (keeps it under the executor's size cap)
REMOVABLE = ["placement", "position", "rotation", "pattern", "parent", "color", "opacity", "label"]


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
    scene = store.get_scene(conv)
    existing = scene["objects"]
    taken = {o["id"] for o in existing}

    added = [copy.deepcopy(raw) for raw in new]
    for i, obj in enumerate(added):
        oid = _clean_id(obj.get("id"))
        if oid and oid in taken:
            raise ToolError(f"objects[{i}]: id '{oid}' already exists. Use scene_update_objects to change it, "
                            f"or pick a new id.")
        obj["id"] = oid
        if oid:
            taken.add(oid)
    for obj in added:
        if not obj["id"]:
            # Models often skip ids and then refer to "the table": name it after its shape.
            base = _clean_id(str(obj.get("shape") or "object").lower()) or "object"
            oid, n = base, 1
            while oid in taken:
                n += 1
                oid = f"{base}{n}"
            obj["id"] = oid
            taken.add(oid)

    notes = _resolve_references(added, existing + added) + _adopt_orphans(added, existing)
    objects = existing + added
    try:
        compiled, fixes = await _compile_and_repair(objects)
    except ToolError as exc:
        raise ToolError(f"{exc} (ids in the scene: {', '.join(o['id'] for o in objects)})") from exc
    notes += fixes
    version = store.save_scene(conv, objects)
    result = _result(version, compiled, [o["id"] for o in added], "Added")
    if notes:
        result["notes"] = notes
    return result


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
        for holder, key in ((obj, "parent"), (obj.get("placement"), "relative_to"), (obj.get("pattern"), "relative_to")):
            if not isinstance(holder, dict):
                continue
            target = resolve(holder.get(key))
            if target and target != obj["id"]:
                notes.append(f"Read '{holder[key]}' as '{target}' for '{obj['id']}'.")
                holder[key] = target
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


async def update_objects(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    conv = _conversation(ctx)
    updates = args.get("updates") or []
    if not updates:
        raise ToolError("'updates' is empty.")
    objects = copy.deepcopy(store.get_scene(conv)["objects"])
    by_id = {o["id"]: o for o in objects}

    touched: list[str] = []
    for i, change in enumerate(updates):
        oid = change.get("id")
        if oid not in by_id:
            known = ", ".join(list(by_id)[:30]) or "none"
            raise ToolError(f"updates[{i}]: no object with id '{oid}'. Existing ids: {known}")
        obj = by_id[oid]
        for key in change.get("remove") or []:
            obj.pop(key, None)
        for key, value in change.items():
            if key in ("id", "remove"):
                continue
            if key == "params" and isinstance(value, dict) and "shape" not in change:
                obj["params"] = {**(obj.get("params") or {}), **value}
            elif key == "parent" and value == "":
                obj.pop("parent", None)
            else:
                obj[key] = value
        # Moving an object one way replaces the other way of positioning it.
        if "position" in change and "placement" not in change:
            obj.pop("placement", None)
        if "placement" in change and "position" not in change:
            obj.pop("position", None)
        touched.append(oid)

    compiled, notes = await _compile_and_repair(objects)
    version = store.save_scene(conv, objects)
    result = _result(version, compiled, touched, "Updated")
    if notes:
        result["notes"] = notes
    return result


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
        if o.get("placement", {}).get("relative_to") in gone:
            o["position"] = _round(current.objects[o["id"]].position)
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


# ---------------------------------------------------------------------------
# definitions
# ---------------------------------------------------------------------------

def shape_catalogue() -> str:
    return "\n".join(f"- {s.signature()}: {s.description}" for s in all_shapes())


def build_tools() -> list[Tool]:
    # Descriptions stay short: every token here is resent on each model turn (see engine/schema.py).
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
                    "plan": {"type": "string", "maxLength": 1500},
                    "objects": {"type": "array", "items": object_schema(), "maxItems": MAX_OBJECTS_PER_CALL},
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
                    "updates": {"type": "array", "items": update_schema(REMOVABLE), "maxItems": MAX_OBJECTS_PER_CALL},
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
