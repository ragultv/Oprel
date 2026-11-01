from __future__ import annotations

import os
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

        if "kokoro" in repo_id.lower() or "kokoro" in model_id.lower():
            return {
                "repo_id": "hexgrad/Kokoro-82M",
                "alias": "kokoro-82m",
                "parameters": "82M",
                "quantizations": ["ONNX"],
                "sizes": {"ONNX": 0.35},
                "default_quantization": "ONNX",
            }

        return get_model_info(repo_id, alias=model_id)
    except Exception as exc:
        logger.error(f"Failed to get model info for {model_id}: {exc}")
        raise


def get_local_quantizations(model_id: str) -> dict[str, Any]:
    from oprel.downloader.aliases import resolve_model_id
    from oprel.downloader.metadata import get_repo_id_from_filename, infer_repo_id_from_cache

    try:
        repo_id = model_id

        if "kokoro" in repo_id.lower() or "kokoro" in model_id.lower():
            from oprel.server.services.speech import check_kokoro_installed
            has_kokoro = check_kokoro_installed()
            return {
                "model_id": model_id,
                "repo_id": "hexgrad/Kokoro-82M",
                "local_quantizations": ["ONNX"] if has_kokoro else [],
                "has_local": has_kokoro,
            }

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

    cat = get_model_category(model_id) or get_model_category(original_id)
    if cat == "speech":
        raise ValueError(
            f"Model '{model_id}' is a speech synthesis model and cannot be loaded as a text model. Please use the Audio Studio."
        )

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

    from oprel.core.config import read_runtime_settings
    saved_rt = read_runtime_settings()
    user_ctx = saved_rt.get("ctx_size") or (CONFIG.ctx_size if "ctx_size" in CONFIG.model_fields_set else None)
    user_kv = saved_rt.get("kv_cache_type") or CONFIG.kv_cache_type

    if not is_embedding:
        # Refuse early, with a plain-language reason, when the model cannot fit even
        # with CPU offload.  Any failure of the check itself must not block loading.
        try:
            from oprel.server.services.preflight import run_preflight

            fit = run_preflight(
                model_id,
                quantization,
                user_ctx,
                user_kv,
            )

            use_ssd_mode = (
                backend in ("native-ssd", "ssd", "llama-ssd")
                or bool(saved_rt.get("use_ssd_wrapper"))
                or os.environ.get("OPREL_BACKEND") == "native-ssd"
            )

            if not use_ssd_mode and fit["verdict"] == "insufficient":
                suggested = fit.get("suggested_context")
                if suggested and suggested < (user_ctx or 131072):
                    logger.warning(
                        f"Requested context {user_ctx} exceeds free RAM ({fit.get('free_ram_gb')} GB free, "
                        f"{fit.get('ram_needed_gb')} GB needed). Automatically clamping to {suggested} tokens."
                    )
                    from oprel.core.config import write_runtime_settings
                    write_runtime_settings(ctx_size=suggested, kv_cache_type=user_kv)
                else:
                    raise RuntimeError(" ".join(fit["messages"]) or "Not enough memory to load this model.")
        except FileNotFoundError:
            pass  # not downloaded yet; the loader will download it
        except RuntimeError:
            raise
        except Exception as exc:
            logger.debug(f"Preflight skipped for {model_id}: {exc}")

    try:
        from oprel.core.config import read_runtime_settings
        saved_rt = read_runtime_settings()
        use_ssd = (
            backend in ("native-ssd", "ssd", "llama-ssd")
            or bool(saved_rt.get("use_ssd_wrapper"))
            or os.environ.get("OPREL_BACKEND") == "native-ssd"
        )

        if use_ssd:
            from oprel.downloader.hub import download_model
            from oprel.runtime.backends.llama_ssd_native import LlamaSSDNativeBackend

            logger.info(f"Loading model with native C++ SSD KV-cache wrapper: {model_id}")
            model_path = download_model(
                model_id,
                quantization=quantization,
                cache_dir=CONFIG.cache_dir,
            )
            # Calculate optimal GPU layers dynamically based on hardware profile
            n_gpu_layers = 16  # fallback default
            free_vram_mb = 0.0
            try:
                from oprel.telemetry.hardware import get_vram_usage
                vram_info = get_vram_usage()
                if vram_info and vram_info.get("free_mb"):
                    free_vram_mb = vram_info["free_mb"]
                    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
                    reserved_overhead_mb = 600.0  # headroom for active KV cache and compute buffer
                    budget_mb = free_vram_mb - reserved_overhead_mb
                    # If entire model + overhead fits in free VRAM, offload 100% of layers (99) to GPU!
                    if free_vram_mb >= file_size_mb + reserved_overhead_mb:
                        n_gpu_layers = 99
                    elif free_vram_mb >= 4500:
                        n_gpu_layers = 99
                    elif free_vram_mb >= 3100:
                        # Near-fit on 4GB systems: offload maximum safe layers
                        n_gpu_layers = 33
                    elif budget_mb > 100:
                        layer_size_mb = file_size_mb / 35.0
                        n_gpu_layers = max(1, min(99, int((free_vram_mb - 500) / layer_size_mb)))
                    else:
                        n_gpu_layers = 0
                    logger.info(
                        f"[VRAM Allocator] Free: {free_vram_mb:.0f}MB, Model: {file_size_mb:.0f}MB "
                        f"-> Dynamically offloading {n_gpu_layers} layers to GPU"
                    )
            except Exception as vram_err:
                logger.warning(f"[VRAM Allocator] Could not dynamically calculate VRAM layers: {vram_err}")

            # Enable MMQ for non-tensor-core GPUs (GTX 1650/1060/1080) for 3x faster prefill throughput
            os.environ["GGML_CUDA_FORCE_MMQ"] = "1"
            native_backend = LlamaSSDNativeBackend()
            native_backend.load_model(str(model_path), n_gpu_layers=n_gpu_layers)

            # Dynamic Context Sizing:
            # max_ssd_ctx represents the total sequence length supported across NVMe SSD offload
            max_ssd_ctx = user_ctx or 131072

            # active_ctx is the active working window kept in GPU VRAM
            # It must be sized strictly to what fits safely in free VRAM without triggering CUDA OOM
            if free_vram_mb >= 10000:
                active_ctx = 8192
            elif free_vram_mb >= 5000:
                active_ctx = 4096
            else:
                active_ctx = 2560

            if user_ctx and user_ctx < active_ctx:
                active_ctx = user_ctx

            # Multi-block retrieval top_k setting (default 2 = 1024 tokens)
            retrieval_top_k = int(saved_rt.get("retrieval_top_k", 2)) if saved_rt else 2

            native_backend.create_context(
                active_ctx_size=active_ctx,
                max_ssd_context=max_ssd_ctx,
                retrieval_top_k=retrieval_top_k,
            )

            class NativeSSDClientAdapter:
                def __init__(self, backend_inst):
                    self.backend = backend_inst

                def generate(self, prompt, stream=True, max_tokens=4096, temperature=0.7, top_p=0.9, repeat_penalty=1.15, **kwargs):
                    tokens_limit = max_tokens if max_tokens and max_tokens > 0 else 4096
                    gen = self.backend.generate(
                        prompt=prompt,
                        max_new_tokens=tokens_limit,
                        temperature=temperature,
                        top_p=top_p,
                        repeat_penalty=repeat_penalty,
                        frequency_penalty=kwargs.get("frequency_penalty", 0.35),
                        presence_penalty=kwargs.get("presence_penalty", 0.30),
                    )
                    if stream:
                        return gen
                    return "".join(gen)

            class NativeSSDModelAdapter:
                def __init__(self, m_id, m_path, backend_inst, act_ctx, m_ssd_ctx):
                    self.model_id = m_id
                    self.model_path = m_path
                    self._model_path = m_path
                    self.backend = backend_inst
                    self.active_ctx_size = act_ctx
                    self.max_ssd_context = m_ssd_ctx
                    self._client = NativeSSDClientAdapter(backend_inst)
                    self._process = None
                    self._monitor = None
                    self._loaded = True

                def close(self):
                    if self.backend:
                        self.backend.close()
                        self.backend = None
                    self._loaded = False

            model = NativeSSDModelAdapter(model_id, str(model_path), native_backend, active_ctx, max_ssd_ctx)
            state.models[model_id] = model
            backend = "native-ssd"
        else:
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
