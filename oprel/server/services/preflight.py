from __future__ import annotations

from pathlib import Path

from oprel.server.services.context import CONFIG, logger


def _resolve_gguf(model_id: str, quantization: str | None) -> Path:
    from oprel.downloader.aliases import resolve_model_id
    from oprel.downloader.metadata import get_repo_id_from_filename, infer_repo_id_from_cache

    if model_id.endswith(".gguf"):
        repo_id = get_repo_id_from_filename(CONFIG.cache_dir, model_id) or infer_repo_id_from_cache(
            CONFIG.cache_dir, model_id
        )
    else:
        repo_id = resolve_model_id(model_id)
    if not repo_id:
        raise FileNotFoundError(f"Model '{model_id}' is not downloaded")

    snapshots = CONFIG.cache_dir / ("models--" + repo_id.replace("/", "--")) / "snapshots"
    candidates = (
        [f for f in snapshots.glob("*/*.gguf") if "mmproj" not in f.name.lower()] if snapshots.exists() else []
    )
    if quantization:
        candidates = [f for f in candidates if quantization.upper() in f.name.upper()]
    if model_id.endswith(".gguf"):
        candidates = [f for f in candidates if f.name == model_id] or candidates
    if not candidates:
        raise FileNotFoundError(f"No downloaded GGUF found for '{model_id}'")
    return sorted(candidates, key=lambda f: f.stat().st_size)[0]


def run_preflight(model_id: str, quantization: str | None, context_length: int | None, kv_cache_type: str) -> dict:
    import psutil

    from oprel.models.gguf_parser import parse_gguf_fast
    from oprel.runtime.preflight import check_fit
    from oprel.telemetry.hardware import get_vram_info

    path = _resolve_gguf(model_id, quantization)
    meta = parse_gguf_fast(str(path))

    vram = get_vram_info()
    free_vram = vram["vram_free_gb"] if vram else None
    free_ram = psutil.virtual_memory().available / 1024**3

    mmproj = next(iter(path.parent.glob("*mmproj*.gguf")), None)
    mmproj_gb = mmproj.stat().st_size / 1024**3 if mmproj else 0.0

    from oprel.runtime.tiers import default_context

    if context_length is None:
        context_length = default_context(free_vram)
    if kv_cache_type in ("auto", "turboquant"):
        kv_cache_type = "q8_0" if kv_cache_type == "auto" else "q4_0"

    result = check_fit(meta, free_vram, free_ram, context_length, kv_cache_type, mmproj_gb).to_dict()
    result.update(
        model_file=path.name,
        model_size_gb=round(meta.file_size_bytes / 1024**3, 2),
        free_vram_gb=round(free_vram, 2) if free_vram is not None else None,
        free_ram_gb=round(free_ram, 2),
    )
    logger.debug(f"Preflight {model_id}: {result['verdict']}")
    return result
