"""3D scene for a conversation: the recipe (JSON) and the compiled mesh (GLB) for the canvas."""

from __future__ import annotations

import asyncio
import copy
from collections import OrderedDict

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel

router = APIRouter()

# (conversation_id, version, updated_at) -> GLB bytes. A saved version never changes.
_GLB_CACHE: OrderedDict[tuple[str, int, str | None], bytes | None] = OrderedDict()
_GLB_CACHE_SIZE = 16


def _engine():
    try:
        from oprel.integrations.scene3d import store
        from oprel.integrations.scene3d.engine import SceneError, compile_scene, export_glb
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=f"3D scene is unavailable: {exc}") from exc
    return store, SceneError, compile_scene, export_glb


@router.get("/scene3d/{conversation_id}")
async def get_scene(conversation_id: str):
    store, *_ = _engine()
    return store.get_scene(conversation_id)


@router.get("/scene3d/{conversation_id}/scene.glb")
async def get_scene_glb(conversation_id: str):
    store, SceneError, compile_scene, export_glb = _engine()
    scene = store.get_scene(conversation_id)
    # With the save time: a reset database starts versions at 1 again for the same id.
    key = (conversation_id, scene["version"], scene.get("updated_at"))
    if key in _GLB_CACHE:
        glb = _GLB_CACHE[key]
    else:
        def build() -> bytes | None:
            return export_glb(compile_scene(scene["objects"])) if scene["objects"] else None
        try:
            glb = await asyncio.to_thread(build)
        except SceneError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        _GLB_CACHE[key] = glb
        while len(_GLB_CACHE) > _GLB_CACHE_SIZE:
            _GLB_CACHE.popitem(last=False)
    if glb is None:
        return Response(status_code=204, headers={"X-Scene-Version": str(scene["version"])})
    return Response(
        content=glb,
        media_type="model/gltf-binary",
        headers={"X-Scene-Version": str(scene["version"]), "Cache-Control": "no-cache"},
    )


class MergeRequest(BaseModel):
    merge: bool = True
    group: str | None = None  # None: the whole scene


@router.post("/scene3d/{conversation_id}/merge")
async def merge_scene(conversation_id: str, body: MergeRequest):
    """The canvas's Merge button: fuse a group (or everything) into one solid, or split it again."""
    store, SceneError, compile_scene, _ = _engine()
    from oprel.integrations.scene3d.tools import merge_scene as apply_merge
    from oprel.tools.base import ToolError

    objects = copy.deepcopy(store.get_scene(conversation_id)["objects"])
    if not objects:
        raise HTTPException(status_code=400, detail="The scene is empty: there is nothing to merge.")
    try:
        notes = apply_merge(objects, body.group, body.merge)
        compiled = await asyncio.to_thread(compile_scene, objects)
    except ToolError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except SceneError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    version = store.save_scene(conversation_id, objects)
    return {"conversation_id": conversation_id, "version": version, "objects": objects, "notes": notes,
            "warnings": compiled.warnings,
            "merged": {gid: {"bodies": s.bodies, "parts": list(s.parts)} for gid, s in compiled.merged.items()}}


@router.get("/scene3d/{conversation_id}/scene.stl")
async def get_scene_stl(conversation_id: str):
    """
    The whole scene fused into one solid as binary STL (metres), for flow simulation or printing.
    X-Solid-Bodies says how many separate pieces it is (1 = one part); X-Solid-Watertight whether
    it is closed.
    """
    store, SceneError, compile_scene, _ = _engine()
    from oprel.integrations.scene3d.engine import export_stl

    scene = store.get_scene(conversation_id)
    if not scene["objects"]:
        return Response(status_code=204, headers={"X-Scene-Version": str(scene["version"])})
    try:
        result = await asyncio.to_thread(lambda: export_stl(compile_scene(scene["objects"])))
    except SceneError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if result is None:
        return Response(status_code=204, headers={"X-Scene-Version": str(scene["version"])})
    data, bodies, watertight = result
    return Response(
        content=data,
        media_type="model/stl",
        headers={"X-Scene-Version": str(scene["version"]), "X-Solid-Bodies": str(bodies),
                 "X-Solid-Watertight": "true" if watertight else "false", "Cache-Control": "no-cache",
                 "Access-Control-Expose-Headers": "X-Scene-Version, X-Solid-Bodies, X-Solid-Watertight"},
    )


@router.delete("/scene3d/{conversation_id}")
async def clear_scene(conversation_id: str):
    store, *_ = _engine()
    version = store.save_scene(conversation_id, [])
    return {"conversation_id": conversation_id, "version": version, "objects": []}
