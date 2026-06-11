from __future__ import annotations

import asyncio
import base64
import json
import re
import threading
import time as time_module
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator

from oprel.core.config import Config
from oprel.server.domain.state import get_state
from oprel.runtime.image_generation import ImageGenerationParams, generate_image_file, generate_image_file_with_progress
from oprel.server.services.context import CONFIG, logger, synchronized_backend_operation
from oprel.server.services.model_state import force_unload_model
from oprel.downloader.image_hub import resolve_image_model_assets


def _parse_size(size: str | None) -> tuple[int, int]:
    if not size:
        return (1024, 1024)
    try:
        width, height = size.lower().split("x", 1)
        return (int(width), int(height))
    except Exception as exc:
        raise ValueError(f"Invalid image size '{size}'. Expected format like 1024x1024.") from exc


def _format_image_data(image_b64: str, response_format: str | None) -> dict[str, str]:
    if response_format == "b64_json":
        return {"b64_json": image_b64}
    return {"url": f"data:image/png;base64,{image_b64}"}


@dataclass
class ImageGenerationJob:
    id: str
    status: str
    progress: float
    message: str
    created: int
    result: dict[str, Any] | None = None
    error: str | None = None


_jobs: dict[str, ImageGenerationJob] = {}
_jobs_lock = threading.Lock()
_prepared_image_models: dict[str, str] = {}
_prepared_image_models_lock = threading.Lock()

_QUANT_PRIORITY = [
    "F32",
    "F16",
    "Q8_0",
    "Q6_K",
    "Q5_K_M",
    "Q5_K_S",
    "Q5_1",
    "Q5_0",
    "Q4_K_M",
    "Q4_K_S",
    "Q4_1",
    "Q4_0",
    "Q3_K_L",
    "Q3_K_M",
    "Q3_K_S",
    "Q2_K_S",
    "Q2_K",
]


def _extract_quant_from_filename(filename: str) -> str | None:
    upper_name = filename.upper()
    for quant in _QUANT_PRIORITY:
        if quant in upper_name:
            return quant

    match = re.search(r"\b(Q\d(?:_[A-Z0-9]+)*)\b", upper_name)
    if match:
        return match.group(1)
    return None


def _quant_sort_key(quant: str | None) -> int:
    if not quant:
        return len(_QUANT_PRIORITY) + 1
    try:
        return _QUANT_PRIORITY.index(quant)
    except ValueError:
        return len(_QUANT_PRIORITY)


def _list_local_image_variants(repo_id: str) -> list[dict[str, Any]]:
    from oprel.downloader.metadata import load_model_metadata
    from oprel.downloader.image_hub import validate_image_gguf_compatibility

    metadata_dir = CONFIG.cache_dir / ".metadata"
    variants: list[dict[str, Any]] = []
    seen_paths: set[str] = set()

    if not metadata_dir.exists():
        return variants

    for metadata_file in metadata_dir.glob("*.json"):
        try:
            filename = metadata_file.stem
            metadata = load_model_metadata(CONFIG.cache_dir, filename) or {}
            if metadata.get("repo_id") != repo_id:
                continue

            local_path = Path(metadata.get("file_path") or "")
            if not local_path.exists() or local_path.suffix.lower() != ".gguf":
                continue

            is_valid, reason = validate_image_gguf_compatibility(local_path)
            if not is_valid:
                logger.info("Skipping incompatible image GGUF '%s': %s", local_path.name, reason)
                continue

            path_str = str(local_path)
            if path_str in seen_paths:
                continue

            seen_paths.add(path_str)
            quant = metadata.get("quantization") or _extract_quant_from_filename(local_path.name)
            supported, reason = validate_image_gguf_compatibility(local_path)
            variants.append(
                {
                    "local_path": path_str,
                    "quantization": quant,
                    "filename": local_path.name,
                    "supported": supported,
                    "compatibility_reason": reason,
                }
            )
        except Exception as exc:
            logger.debug("Skipping invalid image metadata '%s': %s", metadata_file, exc)

    variants.sort(key=lambda item: (_quant_sort_key(item.get("quantization")), item.get("filename") or ""))
    return variants


def list_image_models() -> dict[str, Any]:
    from oprel.downloader.aliases import OFFICIAL_REPOS
    from oprel.downloader.hub import _find_cached_model_for_repo
    from oprel.downloader.image_hub import validate_image_gguf_compatibility

    items: list[dict[str, Any]] = []
    models = OFFICIAL_REPOS.get("text-to-image", {})
    for alias, repo_id in models.items():
        if alias == "vae":
            continue

        local_variants = _list_local_image_variants(repo_id)
        if local_variants:
            for variant in local_variants:
                items.append(
                    {
                        "id": alias,
                        "repo_id": repo_id,
                        "backend": "stable-diffusion.cpp",
                        "downloaded": True,
                        "local_path": variant["local_path"],
                        "quantization": variant.get("quantization"),
                        "supported": variant.get("supported", True),
                        "compatibility_reason": variant.get("compatibility_reason"),
                    }
                )
            continue

        cached = _find_cached_model_for_repo(repo_id, CONFIG.cache_dir, quantization=None)
        if cached is not None:
            is_valid, reason = validate_image_gguf_compatibility(cached)
            if not is_valid:
                logger.info("Cached image model '%s' is incompatible but will still be listed: %s", cached.name, reason)

            items.append(
                {
                    "id": alias,
                    "repo_id": repo_id,
                    "backend": "stable-diffusion.cpp",
                    "downloaded": True,
                    "local_path": str(cached),
                    "quantization": _extract_quant_from_filename(cached.name),
                    "supported": is_valid,
                    "compatibility_reason": reason,
                }
            )
            continue

        items.append(
            {
                "id": alias,
                "repo_id": repo_id,
                "backend": "stable-diffusion.cpp",
                "downloaded": False,
                "local_path": None,
                "quantization": None,
            }
        )

    return {"data": items}


def _create_job() -> ImageGenerationJob:
    job = ImageGenerationJob(
        id=uuid.uuid4().hex,
        status="queued",
        progress=0.0,
        message="Queued",
        created=int(time_module.time()),
    )
    with _jobs_lock:
        _jobs[job.id] = job
    return job


def _update_job(job_id: str, **fields: Any) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return
        for key, value in fields.items():
            setattr(job, key, value)


def _serialize_job(job: ImageGenerationJob) -> dict[str, Any]:
    return {
        "id": job.id,
        "status": job.status,
        "progress": round(job.progress, 2),
        "message": job.message,
        "created": job.created,
        "error": job.error,
        "result": job.result,
    }


def _get_job(job_id: str) -> ImageGenerationJob:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise KeyError("Image generation job not found")
        return job


def _prepare_image_model(model: str) -> str:
    assets = resolve_image_model_assets(model, cache_dir=CONFIG.cache_dir)
    prepared_model = str(assets.model_path or assets.primary_path)

    with _prepared_image_models_lock:
        _prepared_image_models[prepared_model] = model

    logger.info("Prepared image model for generation: %s -> %s", model, prepared_model)
    return prepared_model


def _ensure_exclusive_backend() -> None:
    state = get_state()
    for model_id, model in list(state.models.items()):
        logger.info("Unloading active model before image generation: %s", model_id)
        force_unload_model(model_id, model)
        state.models.pop(model_id, None)
        state.model_configs.pop(model_id, None)
        state.model_last_used.pop(model_id, None)


@synchronized_backend_operation
def _run_image_generation_job(
    job_id: str,
    params: ImageGenerationParams,
    prompt: str,
    response_format: str | None,
) -> None:
    _ensure_exclusive_backend()
    _update_job(job_id, status="running", progress=1.0, message="Starting backend")

    def on_progress(progress: float, message: str) -> None:
        _update_job(job_id, progress=progress, message=message)

    try:
        output_path = generate_image_file_with_progress(
            params,
            progress_callback=on_progress,
            config=Config(**CONFIG.model_dump()),
        )
        image_bytes = Path(output_path).read_bytes()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        result = {
            "created": int(time_module.time()),
            "data": [
                {
                    **_format_image_data(image_b64, response_format),
                    "revised_prompt": prompt,
                }
            ],
        }

        _update_job(job_id, status="completed", progress=100.0, message="Completed", result=result)
    except Exception as exc:
        _update_job(job_id, status="error", error=str(exc), message="Failed")


def _release_image_model(prepared_model: str) -> None:
    with _prepared_image_models_lock:
        original_model = _prepared_image_models.pop(prepared_model, None)

    if original_model is not None:
        logger.info("Released image model after generation: %s", prepared_model)


@synchronized_backend_operation
def start_image_generation(
    prompt: str,
    model: str | None,
    response_format: str | None,
    size: str | None,
    negative_prompt: str | None = None,
    steps: int | None = None,
    cfg_scale: float | None = None,
    seed: int | None = None,
    sampler: str | None = None,
) -> dict[str, Any]:
    width, height = _parse_size(size)
    if not model:
        raise ValueError("Image generation requires a GGUF model path or GGUF repo ID.")

    _ensure_exclusive_backend()
    prepared_model = _prepare_image_model(model)

    params = ImageGenerationParams(
        model=prepared_model,
        prompt=prompt,
        negative_prompt=negative_prompt or "",
        width=width,
        height=height,
        steps=steps or 20,
        cfg_scale=cfg_scale or 7.0,
        seed=seed if seed is not None else -1,
        sampler=sampler,
    )

    job = _create_job()

    def run_job() -> None:
        try:
            _run_image_generation_job(job.id, params, prompt, response_format)
        finally:
            _release_image_model(prepared_model)

    threading.Thread(target=run_job, daemon=True).start()

    return {
        "id": job.id,
        "status": job.status,
        "progress": job.progress,
        "message": job.message,
        "created": job.created,
    }


def get_image_generation_job(job_id: str) -> dict[str, Any]:
    return _serialize_job(_get_job(job_id))


async def stream_image_generation_progress(job_id: str) -> AsyncIterator[str]:
    try:
        while True:
            job = _get_job(job_id)
            yield f"data: {json.dumps(_serialize_job(job))}\n\n"

            if job.status in {"completed", "error"}:
                break
            await asyncio.sleep(0.35)
    except Exception as exc:
        yield f"data: {json.dumps({'error': str(exc)})}\n\n"


@synchronized_backend_operation
async def generate_image(
    prompt: str,
    model: str | None,
    response_format: str | None,
    size: str | None,
    negative_prompt: str | None = None,
    steps: int | None = None,
    cfg_scale: float | None = None,
    seed: int | None = None,
    sampler: str | None = None,
) -> dict:
    width, height = _parse_size(size)
    if not model:
        raise ValueError("Image generation requires a GGUF model path or GGUF repo ID.")

    _ensure_exclusive_backend()
    selected_model = _prepare_image_model(model)

    params = ImageGenerationParams(
        model=selected_model,
        prompt=prompt,
        negative_prompt=negative_prompt or "",
        width=width,
        height=height,
        steps=steps or 20,
        cfg_scale=cfg_scale or 7.0,
        seed=seed if seed is not None else -1,
        sampler=sampler,
    )

    logger.info("Generating image with stable-diffusion.cpp using model '%s'", selected_model)

    try:
        output_path = await asyncio.to_thread(generate_image_file, params, Config(**CONFIG.model_dump()))
        image_bytes = Path(output_path).read_bytes()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        return {
            "created": int(time_module.time()),
            "data": [
                {
                    **_format_image_data(image_b64, response_format),
                    "revised_prompt": prompt,
                }
            ],
        }
    finally:
        _release_image_model(selected_model)
