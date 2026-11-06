"""3D scene for a conversation: the recipe (JSON) and the compiled mesh (GLB) for the canvas."""

from __future__ import annotations

import asyncio
from collections import OrderedDict

from fastapi import APIRouter, HTTPException, Response

router = APIRouter()

# (conversation_id, version) -> GLB bytes. A version never changes once written.
_GLB_CACHE: OrderedDict[tuple[str, int], bytes | None] = OrderedDict()
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
    key = (conversation_id, scene["version"])
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


@router.delete("/scene3d/{conversation_id}")
async def clear_scene(conversation_id: str):
    store, *_ = _engine()
    version = store.save_scene(conversation_id, [])
    return {"conversation_id": conversation_id, "version": version, "objects": []}
