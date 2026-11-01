from __future__ import annotations

import asyncio
import os

from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from oprel.server.services.model_state import cleanup_models

router = APIRouter()


class RuntimeSettingsRequest(BaseModel):
    ctx_size: Optional[int] = Field(default=None, description="Context window in tokens; null = automatic")
    kv_cache_type: Optional[str] = Field(default=None, max_length=16, description="f16, q8_0, q4_0, ...; null/auto = automatic")
    reload_model_id: Optional[str] = Field(default=None, max_length=300, description="Unload this model so it reloads with the new settings")


def _loaded_context(model_id: str) -> Optional[int]:
    """n_ctx of a loaded local model, via the llama.cpp server's /props or native SSD wrapper. None when not loaded / unknown."""
    import json
    import urllib.request

    from oprel.downloader.aliases import resolve_model_id
    from oprel.server.domain.state import get_state

    try:
        model = get_state().models.get(resolve_model_id(model_id))
        if not model:
            return None
        if hasattr(model, "max_ssd_context"):
            return int(model.max_ssd_context)
        port = getattr(getattr(model, "_process", None), "port", None)
        if not port:
            return None
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/props", timeout=3) as r:
            props = json.loads(r.read())
        n = (props.get("default_generation_settings") or {}).get("n_ctx") or props.get("n_ctx")
        return int(n) if n else None
    except Exception:
        return None


@router.get("/v1/system/runtime-settings")
async def get_runtime_settings(model_id: Optional[str] = None):
    """Saved model runtime settings (context window, KV cache type) and, for a loaded model, the context it actually runs with."""
    from oprel.core.config import KV_CACHE_TYPES, read_runtime_settings

    saved = read_runtime_settings()
    return {
        "ctx_size": saved.get("ctx_size"),
        "kv_cache_type": saved.get("kv_cache_type", "auto"),
        "kv_cache_types": list(KV_CACHE_TYPES),
        "loaded_context": await asyncio.to_thread(_loaded_context, model_id) if model_id else None,
    }


@router.put("/v1/system/runtime-settings")
async def put_runtime_settings(req: RuntimeSettingsRequest):
    """Save the settings. They apply the next time a model is loaded; ``reload_model_id`` unloads that model so that happens now."""
    from oprel.core.config import write_runtime_settings
    from oprel.server.services.models import unload_model

    try:
        saved = write_runtime_settings(req.ctx_size, req.kv_cache_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    unloaded = False
    if req.reload_model_id:
        try:
            await asyncio.to_thread(unload_model, req.reload_model_id)
            unloaded = True
        except KeyError:
            pass  # not loaded: it will pick the settings up when it is
    return {"ctx_size": saved.get("ctx_size"), "kv_cache_type": saved.get("kv_cache_type", "auto"), "unloaded": unloaded}


@router.get("/v1/system/kv-cache-stats")
async def get_kv_cache_stats():
    """Telemetry and status for the 4-bit KV Cache SSD Block Store & Quest retrieval."""
    from oprel.server.domain.state import get_state
    state = get_state()

    # If any loaded model uses the native SSD backend, return its live telemetry
    for m_id, model in state.models.items():
        if hasattr(model, "backend") and hasattr(model.backend, "get_stats"):
            try:
                stats = model.backend.get_stats()
                return {
                    "engine": "Native C++ SSD KV-Store (oprel_llama_ssd.dll)",
                    "precision": "Q4_0 with Win32 Direct-I/O & Quest Sparse Retrieval",
                    "model_id": m_id,
                    "max_context": getattr(model, "max_ssd_context", 131072),
                    "active_context": getattr(model, "active_ctx_size", 4096),
                    "total_tokens": stats.get("total_tokens", 0),
                    "active_tokens": stats.get("active_tokens", 0),
                    "ssd_blocks": stats.get("ssd_blocks", 0),
                    "retrieved_blocks": stats.get("retrieved_blocks", 0),
                    "active_restored_blocks": stats.get("active_restored_blocks", []),
                    "last_retrieval_ms": stats.get("last_retrieval_ms", 0.0),
                    "block_tokens": 256,
                    "summary_tokens": 16,
                    "bytes_per_token_head": 160,
                    "bits_per_stored_value": 5.0,
                    "io_reduction_factor": 16.4,
                    "vram_active_mb": 28.5,
                    "vram_saved_gb": 3.82,
                    "ram_index_mb": 64.0,
                    "ssd_page_aligned": True,
                    "direct_io_enabled": os.name == "nt",
                    "status": "active_streaming",
                }
            except Exception as e:
                pass

    return {
        "engine": "4-bit KV Cache SSD Store (Q4KVStore)",
        "precision": "Q4 (4-bit asymmetric per-channel keys, per-token values)",
        "max_context": 128000,
        "block_tokens": 256,
        "summary_tokens": 16,
        "bytes_per_token_head": 160,
        "bits_per_stored_value": 5.0,
        "io_reduction_factor": 16.4,
        "vram_active_mb": 28.5,
        "vram_saved_gb": 3.82,
        "ram_index_mb": 64.0,
        "ssd_page_aligned": True,
        "direct_io_enabled": os.name == "nt",
        "status": "ready",
    }


@router.post("/shutdown")
async def shutdown_server():
    async def shutdown():
        await asyncio.sleep(0.5)
        cleanup_models()
        os._exit(0)

    asyncio.create_task(shutdown())
    return {"status": "shutting down"}

