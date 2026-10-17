from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from oprel.server.services import preflight as preflight_service

router = APIRouter()


@router.get("/system/preflight")
async def preflight(
    model_id: str = Query(..., description="Model alias, repo id or .gguf filename"),
    quantization: str | None = Query(None),
    context_length: int | None = Query(None, ge=256, le=262144),
    kv_cache_type: str = Query("auto"),
):
    """Will this downloaded model fit in the free VRAM/RAM? Returns the verdict and suggested settings."""
    try:
        return preflight_service.run_preflight(model_id, quantization, context_length, kv_cache_type)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/system/onboarding")
async def onboarding():
    """Hardware summary and one recommended first model, for the first-run screen."""
    import psutil

    from oprel.runtime.tiers import detect_tier, recommend_starter
    from oprel.telemetry.hardware import get_hardware_info

    hw = get_hardware_info()
    vram = hw.get("vram_total_gb") or None
    ram = hw.get("ram_total_gb", 0.0)
    return {
        "first_run": _is_first_run(),
        "hardware": {
            "os": hw.get("os"),
            "cpu_threads": hw.get("cpu_threads"),
            "ram_total_gb": ram,
            "ram_available_gb": round(psutil.virtual_memory().available / 1024**3, 1),
            "gpu_name": hw.get("gpu_name"),
            "vram_total_gb": vram,
        },
        "tier": detect_tier(vram),
        "recommended": recommend_starter(vram, ram),
    }


def _is_first_run() -> bool:
    """True only for a genuinely new install: no chats, downloaded models, providers or HF account."""
    from oprel.downloader.cache import list_cached_models
    from oprel.server import db

    try:
        used = (
            bool(db.list_conversations())
            or bool(db.list_providers())
            or bool(db.get_hf_auth())
            or bool(list_cached_models())
        )
    except Exception:
        used = True  # when unsure, never push an existing user into the welcome screen
    return not used
