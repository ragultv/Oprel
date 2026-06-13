from __future__ import annotations

from typing import Any

from oprel.server.domain.models import ModelInfoData
from oprel.server.domain.state import get_state
from oprel.server.services.context import CONFIG, logger, synchronized_backend_operation, track_backend_pid
from oprel.server.services.model_state import force_unload_model, mark_model_used, scan_cached_models


def list_models() -> list[ModelInfoData]:
    return scan_cached_models()


def list_registry_models() -> dict[str, Any]:
    from oprel.downloader.aliases import OFFICIAL_REPOS, CATEGORY_INFO

    return {"categories": CATEGORY_INFO, "models": OFFICIAL_REPOS}


def get_model_detailed_info(model_id: str) -> dict[str, Any]:
    from oprel.downloader.aliases import resolve_model_id
    from oprel.utils.model_info import get_model_info
    from oprel.downloader.metadata import get_repo_id_from_filename, infer_repo_id_from_cache

    try:
        repo_id = model_id

        if model_id.endswith(".gguf"):
            resolved_repo_id = get_repo_id_from_filename(CONFIG.cache_dir, model_id)
            if not resolved_repo_id:
                resolved_repo_id = infer_repo_id_from_cache(CONFIG.cache_dir, model_id)

            if resolved_repo_id:
                logger.info(f"Resolved local filename '{model_id}' -> '{resolved_repo_id}'")
                repo_id = resolved_repo_id
            else:
                logger.warning(f"Could not resolve repo_id for local file: {model_id}")
        else:
            repo_id = resolve_model_id(model_id)

        return get_model_info(repo_id, alias=model_id)
    except Exception as exc:
        logger.error(f"Failed to get model info for {model_id}: {exc}")
        raise


def get_local_quantizations(model_id: str) -> dict[str, Any]:
    from oprel.downloader.aliases import resolve_model_id
    from oprel.downloader.metadata import get_repo_id_from_filename, infer_repo_id_from_cache

    try:
        repo_id = model_id

        if model_id.endswith(".gguf"):
            resolved_repo_id = get_repo_id_from_filename(CONFIG.cache_dir, model_id)
            if not resolved_repo_id:
                resolved_repo_id = infer_repo_id_from_cache(CONFIG.cache_dir, model_id)

            if resolved_repo_id:
                logger.debug(f"Resolved local filename '{model_id}' -> '{resolved_repo_id}'")
                repo_id = resolved_repo_id
            else:
                logger.warning(f"Could not resolve repo_id for local file: {model_id}")
        else:
            repo_id = resolve_model_id(model_id)

        cache_name = "models--" + repo_id.replace("/", "--")
        cache_dir = CONFIG.cache_dir / cache_name

        local_quants: list[str] = []

        if cache_dir.exists():
            snapshots_dir = cache_dir / "snapshots"
            if snapshots_dir.exists():
                for snapshot in snapshots_dir.iterdir():
                    if snapshot.is_dir():
                        for file in snapshot.glob("*.gguf"):
                            name_upper = file.name.upper()
                            for quant in ["Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16"]:
                                if quant in name_upper and quant not in local_quants:
                                    local_quants.append(quant)
                                    break

        return {
            "model_id": model_id,
            "repo_id": repo_id,
            "local_quantizations": local_quants,
            "has_local": len(local_quants) > 0,
        }
    except Exception as exc:
        logger.error(f"Failed to get local quantizations for {model_id}: {exc}")
        raise


@synchronized_backend_operation
def load_model(
    model_id: str,
    quantization: str | None = None,
    max_memory_mb: int | None = None,
    backend: str = "llama.cpp",
) -> dict[str, Any]:
    state = get_state()
    from oprel.downloader.aliases import resolve_model_id, get_model_category
    from oprel.downloader.metadata import get_repo_id_from_filename, infer_repo_id_from_cache
    from oprel.server import db

    original_id = model_id

    if model_id.endswith(".gguf"):
        repo_id = get_repo_id_from_filename(CONFIG.cache_dir, model_id)
        if not repo_id:
            repo_id = infer_repo_id_from_cache(CONFIG.cache_dir, model_id)
        if repo_id:
            logger.info(f"Resolved local filename '{model_id}' -> '{repo_id}'")
            model_id = repo_id
        else:
            logger.warning(f"Could not resolve repo_id for local file: {model_id}")
    else:
        model_id = resolve_model_id(model_id)

    if model_id in state.models:
        model = state.models[model_id]
        process_alive = False
        if hasattr(model, "_process") and model._process is not None:
            process_alive = model._process.is_running()

        if not process_alive:
            logger.warning(f"Backend process for {model_id} died, reloading...")
            force_unload_model(model_id, model)
            state.models.pop(model_id, None)
            state.model_configs.pop(model_id, None)
            state.model_last_used.pop(model_id, None)
        else:
            loaded_quant = (state.model_configs.get(model_id, {}).get("quantization") or "").upper()
            requested_quant = (quantization or "").upper()

            if not requested_quant or loaded_quant == requested_quant:
                logger.info(f"Model {model_id} [{loaded_quant or 'auto'}] already loaded")
                return {"success": True, "model_id": model_id, "message": "Model already loaded"}

            logger.info(f"Switching quant for {model_id}: {loaded_quant} -> {requested_quant}")
            force_unload_model(model_id, model)
            state.models.pop(model_id, None)
            state.model_configs.pop(model_id, None)
            state.model_last_used.pop(model_id, None)

    if model_id in state.models:
        mark_model_used(model_id)
        return {"success": True, "model_id": model_id, "message": "Model already loaded"}

    def is_embedding_model(m_id: str) -> bool:
        if get_model_category(m_id) == "embeddings":
            return True
        return "embed" in m_id.lower() or "bge-" in m_id.lower()

    is_embedding = is_embedding_model(original_id) or is_embedding_model(model_id)

    if not is_embedding:
        for old_model_id in list(state.models.keys()):
            if is_embedding_model(old_model_id):
                continue
            logger.info(f"Unloading previous LLM '{old_model_id}' before loading '{model_id}'")
            try:
                old_model = state.models[old_model_id]
                force_unload_model(old_model_id, old_model)
                state.models.pop(old_model_id, None)
                state.model_configs.pop(old_model_id, None)
                state.model_last_used.pop(old_model_id, None)
            except Exception as exc:
                logger.warning(f"Error unloading previous model {old_model_id}: {exc}")

    p_id = model_id
    if "::" in p_id:
        p_id = p_id.split("::", 1)[0]
    elif ":" in p_id:
        p_id = p_id.split(":", 1)[0]

    provider = db.get_provider(p_id)
    if provider:
        logger.info(f"Model ID '{model_id}' matches external provider '{provider['name']}'")
        return {
            "success": True,
            "model_id": model_id,
            "message": f"Model is provided by external provider: {provider['name']}",
        }

    try:
        from oprel.core.model import Model

        logger.info(f"Loading model: {model_id} (quant={quantization}, backend={backend})")

        model = Model(
            model_id=model_id,
            quantization=quantization,
            max_memory_mb=max_memory_mb,
            backend=backend,
            use_server=False,
        )

        model.load()

        if hasattr(model, "_process") and model._process and model._process.process:
            track_backend_pid(model._process.process.pid)

        state.models[model_id] = model

        actual_quant = quantization
        actual_filename = None
        if not actual_quant:
            if hasattr(model, "_model_path") and model._model_path:
                p = str(model._model_path)
                actual_filename = p.split("\\")[-1].split("/")[-1]
            elif hasattr(model, "model_path") and model.model_path:
                p = str(model.model_path)
                actual_filename = p.split("\\")[-1].split("/")[-1]
            elif hasattr(model, "_process") and model._process:
                if hasattr(model._process, "model_path") and model._process.model_path:
                    p = str(model._process.model_path)
                    actual_filename = p.split("\\")[-1].split("/")[-1]

            if actual_filename:
                quant_patterns = [
                    "Q3_K_XL", "Q3_K_L", "Q3_K_M", "Q3_K_S", "Q3_K",
                    "Q4_K_XL", "Q4_K_L", "Q4_K_M", "Q4_K_S", "Q4_K",
                    "Q5_K_XL", "Q5_K_L", "Q5_K_M", "Q5_K_S", "Q5_K",
                    "Q2_K_XL", "Q2_K_L", "Q2_K_S", "Q2_K",
                    "Q6_K", "Q8_0", "Q4_0", "Q4_1", "F32", "F16", "BF16",
                ]
                fn_upper = actual_filename.upper()
                actual_quant = next((q for q in quant_patterns if q in fn_upper), None)

        state.model_configs[model_id] = {
            "quantization": actual_quant,
            "filename": actual_filename,
            "max_memory_mb": max_memory_mb,
            "backend": backend,
        }

        mark_model_used(model_id)

        logger.info(f"Model loaded successfully: {model_id}")

        return {"success": True, "model_id": model_id, "message": "Model loaded successfully"}

    except Exception as exc:
        logger.error(f"Failed to load model {model_id}: {exc}", exc_info=True)
        raise


@synchronized_backend_operation
def unload_model(model_id: str) -> dict[str, Any]:
    state = get_state()
    from oprel.downloader.aliases import resolve_model_id

    model_id = resolve_model_id(model_id)

    if model_id not in state.models:
        raise KeyError(f"Model '{model_id}' not loaded")

    try:
        model = state.models[model_id]
        force_unload_model(model_id, model)
        state.models.pop(model_id, None)
        state.model_configs.pop(model_id, None)
        state.model_last_used.pop(model_id, None)
        return {"success": True, "model_id": model_id, "message": "Unloaded"}
    except Exception as exc:
        logger.error(f"Failed to unload {model_id}: {exc}")
        raise


@synchronized_backend_operation
def delete_model_quant(model_id: str, quantization: str) -> dict[str, Any]:
    from oprel.downloader.aliases import resolve_model_id
    from oprel.downloader.metadata import get_repo_id_from_filename, infer_repo_id_from_cache

    state = get_state()

    repo_id = resolve_model_id(model_id)
    deleted_files: list[str] = []
    quant_upper = quantization.upper()

    try:
        for gguf_file in CONFIG.cache_dir.rglob("*.gguf"):
            if not gguf_file.is_file():
                continue

            filename = gguf_file.name
            fname_lower = filename.lower()

            if "mmproj" in fname_lower or fname_lower.startswith("vision-") or fname_lower.startswith("clip-"):
                continue

            if quant_upper not in filename.upper():
                continue

            file_repo_id = get_repo_id_from_filename(CONFIG.cache_dir, filename)
            if not file_repo_id:
                file_repo_id = infer_repo_id_from_cache(CONFIG.cache_dir, filename)
            if file_repo_id and file_repo_id != repo_id:
                continue

            gguf_file.unlink(missing_ok=True)
            deleted_files.append(filename)
            logger.info(f"Deleted model file: {gguf_file}")

            meta_file = CONFIG.cache_dir / ".metadata" / f"{filename}.json"
            if meta_file.exists():
                meta_file.unlink()
                logger.info(f"Deleted metadata: {meta_file}")

        if not deleted_files:
            raise FileNotFoundError(f"No downloaded file found for {model_id} / {quantization}")

        for loaded_id in list(state.models.keys()):
            cfg = state.model_configs.get(loaded_id, {})
            if loaded_id in (model_id, repo_id) and (cfg.get("quantization", "").upper() == quant_upper):
                logger.info(f"Unloading active model {loaded_id} after file deletion")
                force_unload_model(loaded_id)
                state.models.pop(loaded_id, None)
                state.model_configs.pop(loaded_id, None)
                state.model_last_used.pop(loaded_id, None)

        return {
            "success": True,
            "deleted": deleted_files,
            "model_id": model_id,
            "quantization": quantization,
        }

    except Exception as exc:
        logger.error(f"Error deleting model {model_id} / {quantization}: {exc}")
        raise
