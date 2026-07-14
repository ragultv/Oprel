from __future__ import annotations

import asyncio
import time as time_module
from typing import Any

from oprel.server.domain.models import ModelInfoData
from oprel.server.domain.state import get_state
from oprel.server.services.context import CONFIG, logger, remove_daemon_pid, untrack_backend_pid

IDLE_TIMEOUT_SECONDS = 15 * 60


def cleanup_models() -> None:
    state = get_state()
    for model_id, model in list(state.models.items()):
        try:
            force_unload_model(model_id, model)
            print(f"Unloaded model: {model_id}")
        except Exception as exc:
            print(f"Error unloading {model_id}: {exc}")
    state.models.clear()
    state.model_configs.clear()
    state.model_last_used.clear()
    remove_daemon_pid()
    try:
        from oprel.server.services.context import BACKEND_PIDS_FILE
        BACKEND_PIDS_FILE.unlink(missing_ok=True)
    except Exception:
        pass


def force_unload_model(model_id: str, model: Any | None = None) -> None:
    state = get_state()
    if model is None:
        model = state.models.get(model_id)
    if model is None:
        return

    backend_pid = None

    try:
        if hasattr(model, "_process") and model._process:
            if model._process.process:
                backend_pid = model._process.process.pid

            logger.info(f"Stopping backend process for {model_id} (PID: {backend_pid})")
            model._process.stop(force=False)

            if backend_pid:
                try:
                    import psutil

                    proc = psutil.Process(backend_pid)
                    if proc.is_running():
                        logger.warning(f"Backend PID {backend_pid} still alive after stop, force killing")
                        proc.kill()
                        proc.wait(timeout=3)
                except Exception:
                    pass
                untrack_backend_pid(backend_pid)

            model._process = None

        if hasattr(model, "_monitor") and model._monitor:
            model._monitor.stop()
            model._monitor = None

        model._loaded = False

        import gc

        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except (ImportError, Exception):
            pass

        logger.info(f"Model {model_id} fully unloaded")

    except Exception as exc:
        logger.error(f"Error force-unloading model {model_id}: {exc}")
        import traceback

        logger.error(traceback.format_exc())


def unload_idle_model(model_id: str) -> None:
    state = get_state()
    if model_id not in state.models:
        return

    try:
        model = state.models[model_id]
        logger.info(f"Unloading idle model: {model_id} (forcing backend termination)")

        force_unload_model(model_id, model)

        state.models.pop(model_id, None)
        state.model_configs.pop(model_id, None)
        state.model_last_used.pop(model_id, None)

        logger.info(f"Idle model {model_id} unloaded, GPU memory freed")
    except Exception as exc:
        logger.error(f"Error unloading idle model {model_id}: {exc}")
        import traceback

        logger.error(traceback.format_exc())


async def monitor_idle_models() -> None:
    state = get_state()
    while True:
        try:
            await asyncio.sleep(60)

            current_time = time_module.time()
            models_to_unload: list[str] = []

            for model_id in list(state.models.keys()):
                last_used = state.model_last_used.get(model_id, 0)
                idle_time = current_time - last_used

                if idle_time > IDLE_TIMEOUT_SECONDS:
                    models_to_unload.append(model_id)
                    logger.info(
                        f"Model {model_id} has been idle for {idle_time / 60:.1f} minutes"
                    )

            for model_id in models_to_unload:
                unload_idle_model(model_id)

        except asyncio.CancelledError:
            logger.info("Idle model monitoring task cancelled")
            break
        except Exception as exc:
            logger.error(f"Error in idle model monitor: {exc}")


def mark_model_used(model_id: str) -> None:
    state = get_state()
    state.model_last_used[model_id] = time_module.time()


def scan_cached_models() -> list[ModelInfoData]:
    state = get_state()
    available: list[ModelInfoData] = []
    seen_files: set[tuple[str, str]] = set()

    quant_patterns = [
        "Q3_K_XL", "Q3_K_L", "Q3_K_M", "Q3_K_S", "Q3_K",
        "Q4_K_XL", "Q4_K_L", "Q4_K_M", "Q4_K_S", "Q4_K",
        "Q5_K_XL", "Q5_K_L", "Q5_K_M", "Q5_K_S", "Q5_K",
        "Q2_K_XL", "Q2_K_L", "Q2_K_S", "Q2_K",
        "Q6_K", "Q8_0", "Q4_0", "Q4_1", "Q5_0", "Q5_1",
        "IQ1_M", "IQ1_S", "IQ2_M", "IQ2_S", "IQ2_XS", "IQ2_XXS",
        "IQ3_M", "IQ3_S", "IQ3_XS", "IQ4_NL", "IQ4_XS",
        "F32", "F16", "BF16",
    ]

    for model_id, config in state.model_configs.items():
        from oprel.downloader.aliases import get_category_info, get_model_category, get_best_alias_for_repo

        cat = get_model_category(model_id)
        backend = get_category_info(cat).get("backend", config.get("backend", "llama.cpp")) if cat else config.get("backend", "llama.cpp")

        model_id_lower = model_id.lower()
        is_unwanted = any(
            kw in model_id_lower
            for kw in ["embed", "embedding", "nomic-embed", "bge-m3"]
        )
        if is_unwanted or cat in ["embeddings", "text-to-video"]:
            continue

        quant = config.get("quantization") or "Unknown"
        if quant == "Unknown":
            continue

        file_key = (model_id, quant)
        if file_key in seen_files:
            continue
        seen_files.add(file_key)

        best_alias = get_best_alias_for_repo(model_id)
        display_name = best_alias or (model_id.split("/")[-1] if "/" in model_id else model_id)

        available.append(
            ModelInfoData(
                model_id=model_id,
                quantization=quant,
                backend=backend,
                loaded=True,
                name=display_name,
                category=cat,
            )
        )

    cache_dir = CONFIG.cache_dir
    if not cache_dir.exists():
        return available

    try:
        from oprel.downloader.metadata import get_repo_id_from_filename, infer_repo_id_from_cache
        from oprel.downloader.aliases import get_category_info, get_model_category, get_best_alias_for_repo

        for gguf_file in cache_dir.rglob("*.gguf"):
            if not gguf_file.is_file() or gguf_file.stat().st_size == 0:
                continue

            filename = gguf_file.name
            filename_lower = filename.lower()

            if "mmproj" in filename_lower or filename_lower.startswith("vision-") or filename_lower.startswith("clip-"):
                continue

            repo_id = (
                get_repo_id_from_filename(cache_dir, filename)
                or infer_repo_id_from_cache(cache_dir, filename)
                or filename
            )
            # Determine category for the repo_id (works even if repo_id == filename)
            cat = get_model_category(repo_id)
            repo_id_lower = repo_id.lower()
            is_unwanted = any(
                kw in repo_id_lower
                for kw in [
                    "embed", "embedding", "nomic-embed", "bge-m3",
                ]
            )
            if is_unwanted or cat in ["embeddings", "text-to-video"]:
                continue

            file_key = (repo_id, filename)
            if file_key in seen_files:
                continue
            seen_files.add(file_key)

            name_upper = filename.upper()
            quant = next((q for q in quant_patterns if q in name_upper), "Unknown")

            is_loaded = False
            for loaded_id, cfg in state.model_configs.items():
                ids_match = loaded_id in (repo_id, filename.replace(".gguf", ""))
                if not ids_match:
                    continue
                stored_filename = cfg.get("filename")
                loaded_quant = (cfg.get("quantization") or "").upper()
                if stored_filename:
                    is_loaded = filename == stored_filename
                elif loaded_quant:
                    is_loaded = loaded_quant == quant
                if is_loaded:
                    break

            size_gb = gguf_file.stat().st_size / (1024 ** 3)
            backend = get_category_info(cat).get("backend", "llama.cpp") if cat else "llama.cpp"

            best_alias = get_best_alias_for_repo(repo_id)
            if best_alias:
                display_name = best_alias
            elif repo_id != filename and "/" in repo_id:
                display_name = repo_id.split("/")[-1]
            else:
                display_name = filename.replace(".gguf", "")

            already_loaded = any(
                cfg.get("quantization") == quant
                and (model_id == repo_id or cfg.get("filename") == filename)
                for model_id, cfg in state.model_configs.items()
            )
            if already_loaded:
                continue

            available.append(
                ModelInfoData(
                    model_id=repo_id,
                    quantization=quant if quant != "Unknown" else None,
                    backend=backend,
                    loaded=is_loaded,
                    size_gb=round(size_gb, 2),
                    name=display_name,
                    category=cat,
                )
            )

    except Exception as exc:
        logger.error(f"Error scanning cache: {exc}")

    return available
