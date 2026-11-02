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


# Tokens per archived block once generation is running; prefill may archive up to 1024 at once.
SSD_BLOCK_TOKENS = 512


def _ssd_kv_type() -> str:
    # Mirrors oprel_create_context: unknown values fall back to q4_0.
    kv = os.environ.get("OPREL_SSD_KV_TYPE", "q4_0").strip().lower()
    return kv if kv in ("q4_0", "q8_0", "f16") else "q4_0"


def _dir_bytes(path: str) -> int:
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file(follow_symlinks=False):
                total += entry.stat().st_size
            elif entry.is_dir(follow_symlinks=False):
                total += _dir_bytes(entry.path)
    except OSError:
        pass
    return total


def _file_size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _disk_info(ssd_dir: str) -> dict:
    import shutil
    from pathlib import Path

    probe = Path(ssd_dir)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        usage = shutil.disk_usage(probe)
        free, total = usage.free, usage.total
    except OSError:
        free = total = 0
    return {"ssd_dir": ssd_dir, "ssd_dir_bytes": _dir_bytes(ssd_dir), "disk_free_bytes": free, "disk_total_bytes": total}


@router.get("/v1/system/kv-cache-stats")
async def get_kv_cache_stats():
    """Live telemetry of the native SSD KV-cache backend. Every value is measured, none is a constant."""
    from pathlib import Path
    from oprel.server.domain.state import get_state

    state = get_state()
    default_dir = str(Path.home() / ".cache" / "oprel" / "ssd_kv")

    for m_id, model in list(state.models.items()):
        backend = getattr(model, "backend", None)
        if backend is None or not hasattr(backend, "get_stats"):
            continue
        try:
            stats = backend.get_stats()
        except Exception:
            continue
        if not stats:
            continue

        ssd_dir = getattr(backend, "ssd_cache_dir", None) or default_dir
        session_id = getattr(backend, "current_session_id", None)
        total_tokens = stats.get("total_tokens", 0)
        active_tokens = stats.get("active_tokens", 0)
        evicted_tokens = max(0, total_tokens - active_tokens)

        block_file_bytes = _file_size(os.path.join(ssd_dir, f"{session_id}.kv")) if session_id else 0
        session_file_bytes = (
            _file_size(os.path.join(ssd_dir, "sessions", f"session_{session_id}.bin")) if session_id else 0
        )
        # Measured KV size per token: archived blocks if any, else the saved live window of this session.
        bytes_per_token = None
        if block_file_bytes and evicted_tokens:
            bytes_per_token = round(block_file_bytes / evicted_tokens)
        elif session_file_bytes and total_tokens:
            bytes_per_token = round(session_file_bytes / total_tokens)

        disk = _disk_info(ssd_dir)
        kv_type = _ssd_kv_type()
        return {
            "engine": "Native C++ SSD KV-Store (oprel_llama_ssd.dll)",
            "status": "active",
            "model_id": m_id,
            "kv_cache_type": kv_type,
            "precision": f"{kv_type.upper()} KV cache, {'Win32 Direct I/O' if os.name == 'nt' else 'buffered I/O'}",
            "direct_io_enabled": os.name == "nt",
            "block_tokens": SSD_BLOCK_TOKENS,
            "retrieval_top_k": getattr(backend, "retrieval_top_k", None),
            "max_context": getattr(model, "max_ssd_context", None),
            "active_context": getattr(model, "active_ctx_size", None),
            "session_id": session_id,
            "total_tokens": total_tokens,
            "active_tokens": active_tokens,
            "evicted_tokens": evicted_tokens,
            "ssd_blocks": stats.get("ssd_blocks", 0),
            "retrieved_blocks": stats.get("retrieved_blocks", 0),
            "active_restored_blocks": stats.get("active_restored_blocks", []),
            "last_prefill_tokens": stats.get("last_prefill_tokens", 0),
            "last_retrieval_ms": stats.get("last_retrieval_ms", 0.0),
            "last_restore_latency_ms": stats.get("last_restore_latency_ms", 0.0),
            "last_read_ms": stats.get("last_read_ms", 0.0),
            "last_set_data_ms": stats.get("last_set_data_ms", 0.0),
            "read_throughput_mb_s": stats.get("read_throughput_mb_s", 0.0),
            "ssd_bytes_written": stats.get("total_ssd_bytes_written", 0),
            "ssd_bytes_read": stats.get("total_ssd_bytes_read", 0),
            "archive_validation_failures": stats.get("archive_validation_failures", 0),
            "session_block_file_bytes": block_file_bytes,
            "session_file_bytes": session_file_bytes,
            "bytes_per_token": bytes_per_token,
            "tokens_fit_in_free_space": (disk["disk_free_bytes"] // bytes_per_token) if bytes_per_token else None,
            **disk,
        }

    return {
        "engine": None,
        "status": "inactive",
        "message": "No model is loaded with the SSD KV-cache backend.",
        "kv_cache_type": _ssd_kv_type(),
        "block_tokens": SSD_BLOCK_TOKENS,
        "direct_io_enabled": os.name == "nt",
        **_disk_info(default_dir),
    }


@router.post("/shutdown")
async def shutdown_server():
    async def shutdown():
        await asyncio.sleep(0.5)
        cleanup_models()
        os._exit(0)

    asyncio.create_task(shutdown())
    return {"status": "shutting down"}

