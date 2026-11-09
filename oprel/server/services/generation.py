from __future__ import annotations

import asyncio
import base64
import dataclasses
import json
import os
import time as time_module
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator

import httpx

from oprel.server.domain.state import get_state
from oprel.server.services.context import CONFIG, logger
from oprel.server.services.model_state import mark_model_used
from oprel.server.services.models import load_model
from oprel.server import db


@dataclass(frozen=True)
class GenerateParams:
    model_id: str
    prompt: Any
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    repeat_penalty: float
    stream: bool
    images: list[str] | None
    conversation_id: str | None
    system_prompt: str | None
    reset_conversation: bool
    thinking: bool
    rag: bool


@dataclass(frozen=True)
class GenerateResult:
    text: str
    model_id: str
    conversation_id: str
    message_count: int


@dataclass(frozen=True)
class StreamResult:
    iterator: AsyncIterator[str]
    conversation_id: str


@dataclass(frozen=True)
class EmbeddingParams:
    model: str
    input: str | list[str]


@dataclass(frozen=True)
class EmbeddingResult:
    embedding: list[float] | None = None
    embeddings: list[list[float]] | None = None


def build_chat_prompt(
    model_id: str,
    history: list[dict[str, str]],
    system_prompt: str | None = None,
    new_user_msg: str = "",
    thinking: bool = False,
) -> str:
    from oprel.utils.chat_templates import format_chat_prompt

    conversation_history = []
    conversation_history.extend(history)

    return format_chat_prompt(
        model_id=model_id,
        user_message=new_user_msg,
        system_prompt=system_prompt,
        conversation_history=conversation_history,
        thinking=thinking,
    )


def _resolve_model_id(model_id: str) -> str:
    from oprel.downloader.aliases import resolve_model_id
    from oprel.downloader.metadata import get_repo_id_from_filename, infer_repo_id_from_cache

    if model_id.endswith(".gguf"):
        repo_id = get_repo_id_from_filename(CONFIG.cache_dir, model_id)
        if not repo_id:
            repo_id = infer_repo_id_from_cache(CONFIG.cache_dir, model_id)

        if repo_id:
            logger.info(f"Resolved local filename '{model_id}' -> '{repo_id}'")
            return repo_id

        logger.warning(f"Could not resolve repo_id for local file: {model_id}")
        return model_id

    return resolve_model_id(model_id)


def _extract_prompt_and_images(prompt: Any, images: list[str] | None) -> tuple[str, list[str] | None, str | list[Any]]:
    import base64
    from oprel.utils.vision_processor import VisionImageProcessor, VisionProcessingResult

    def _process_one_image(b64: str) -> str:
        """
        Run VisionImageProcessor on a single base64-encoded image.

        Returns an optimized base64 string.  On any failure, logs clearly and
        returns the *original* b64 so we never silently lose the image —
        but the caller will know the full-size image is being sent.
        """
        if not CONFIG.vision_optimization_enabled:
            return b64

        try:
            img_bytes = base64.b64decode(b64)
        except Exception as decode_exc:
            logger.error(f"[Vision] Failed to base64-decode image: {decode_exc}")
            return b64

        try:
            processor = VisionImageProcessor(
                max_pixels=CONFIG.vision_max_pixels,
                quality=CONFIG.vision_image_quality,
            )
            result: VisionProcessingResult = processor.process(img_bytes)

            # ── [Vision Performance] benchmark summary ──────────────────────
            logger.info(
                f"[Vision Performance]\n"
                f"  Original:       {result.original_width}x{result.original_height} "
                f"({result.original_pixels / 1_000_000:.2f} MP)\n"
                f"  Optimized:      {result.output_width}x{result.output_height} "
                f"({result.output_pixels / 1_000_000:.2f} MP)\n"
                f"  Pixel budget:   {CONFIG.vision_max_pixels}\n"
                f"  Resized:        {result.was_resized}\n"
                f"  Scale factor:   {result.scale_factor:.3f}\n"
                f"  Preprocessing:  {result.preprocessing_ms:.1f} ms"
            )

            return base64.b64encode(result.image_bytes).decode("utf-8")

        except Exception as proc_exc:
            logger.error(
                f"[Vision] Preprocessing failed: {proc_exc}  "
                f"Falling back to original image (may be oversized).",
                exc_info=True,
            )
            return b64

    if isinstance(prompt, list):
        text_parts: list[str] = []
        if images is None:
            images = []
        for item in prompt:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif item.get("type") == "image_url":
                    url = item.get("image_url", {}).get("url", "")
                    if url.startswith("data:image"):
                        try:
                            _, b64 = url.split(",", 1)
                            optimized_b64 = _process_one_image(b64)
                            images.append(optimized_b64)
                        except Exception as e:
                            logger.error(f"[Vision] Failed to extract image from data URL: {e}")
        return " ".join(text_parts), images, prompt

    if images:
        images = [_process_one_image(b64) for b64 in images]

    return str(prompt), images, prompt



async def generate_text(params: GenerateParams) -> GenerateResult | StreamResult:
    state = get_state()
    resolved_model_id = _resolve_model_id(params.model_id)

    if resolved_model_id not in state.models:
        p_id = resolved_model_id
        if ":" in p_id:
            p_id = p_id.split(":", 1)[0]

        provider = db.get_provider(p_id)
        if not provider:
            await asyncio.to_thread(load_model, resolved_model_id)

    if resolved_model_id in state.models:
        model = state.models[resolved_model_id]

        if hasattr(model, "_process") and model._process is not None:
            if not model._process.is_running():
                logger.warning(f"Backend process for {resolved_model_id} died, reloading...")
                state.models.pop(resolved_model_id, None)
                state.model_configs.pop(resolved_model_id, None)
                await asyncio.to_thread(load_model, resolved_model_id)
                model = state.models[resolved_model_id]

        model._loaded = True

        if not hasattr(model, "_client") or model._client is None:
            raise RuntimeError("Model client not available")

        mark_model_used(resolved_model_id)
    else:
        p_id = resolved_model_id
        if "::" in p_id:
            p_id = p_id.split("::", 1)[0]
        elif ":" in p_id:
            p_id = p_id.split(":", 1)[0]

        provider = db.get_provider(p_id)
        if provider:
            from oprel.server.services.providers import provider_chat_proxy
            from oprel.server.schemas.providers import ProviderChatRequest

            m_name = (
                resolved_model_id.split("::", 1)[1]
                if "::" in resolved_model_id
                else (resolved_model_id.split(":", 1)[1] if ":" in resolved_model_id else None)
            )
            if not m_name:
                enabled = provider.get("enabled_model_ids", [])
                m_name = enabled[0] if enabled else resolved_model_id

            messages = []
            if params.system_prompt:
                messages.append({"role": "system", "content": params.system_prompt})
            messages.append({"role": "user", "content": params.prompt})

            proxy_body = ProviderChatRequest(
                model=m_name,
                messages=messages,
                max_tokens=params.max_tokens,
                temperature=params.temperature,
                top_p=params.top_p,
                stream=params.stream,
                conversation_id=params.conversation_id,
            )
            result = await provider_chat_proxy(p_id, proxy_body)
            if isinstance(result, StreamResult):
                return result
            return GenerateResult(
                text=result.text,
                model_id=result.model_id,
                conversation_id=result.conversation_id,
                message_count=result.message_count,
            )

        raise KeyError(f"Model '{resolved_model_id}' not found or not loaded")

    text_prompt, images, raw_prompt = _extract_prompt_and_images(params.prompt, params.images)

    conv_id = params.conversation_id
    is_persistent = False

    if conv_id and conv_id.startswith("chat_"):
        is_persistent = True
    elif not conv_id:
        conv_id = f"ephemeral_{uuid.uuid4().hex[:12]}"
        is_persistent = False

    if params.reset_conversation:
        if is_persistent:
            db.reset_conversation(conv_id)
        else:
            state.ephemeral_history.pop(conv_id, None)

    if is_persistent:
        history = db.get_conversation_messages(conv_id)
    else:
        history = state.ephemeral_history.get(conv_id, [])

    if params.thinking:
        if params.max_tokens < 8192:
            params = dataclasses.replace(params, max_tokens=8192)
    else:
        # Vision/multimodal OCR requests can produce long JSON; allow up to 4096 tokens
        if params.images:
            if params.max_tokens > 4096:
                params = dataclasses.replace(params, max_tokens=4096)
        elif params.max_tokens > 2048:
            params = dataclasses.replace(params, max_tokens=2048)

    context_text = ""
    if params.rag:
        try:
            from oprel.knowledge.orchestration.knowledge_service import KnowledgeService
            from oprel.downloader.aliases import resolve_model_id

            async def internal_embed(text: str | list[str], model: str | None = None) -> list[float] | list[list[float]]:
                resolved_embed_model = resolve_model_id(model or "nomic-embed-text")
                if isinstance(text, list):
                    res = await get_embeddings(EmbeddingParams(input=text, model=resolved_embed_model))
                    return res.embeddings or []
                res = await get_embeddings(EmbeddingParams(input=text, model=resolved_embed_model))
                return res.embedding or []

            service = KnowledgeService(embed_func=internal_embed)
            grounded = await service.ground_query(
                query=text_prompt,
                model_ctx_size=getattr(CONFIG, "ctx_size", 8192),
                conv_id=conv_id,
                force_rag=True,
            )

            rag_html_block = ""
            if grounded.evidence:
                text_prompt = grounded.prompt_text
                logger.info(f"RAG: Grounded with {len(grounded.evidence)} relevant chunks (est {grounded.estimated_context_tokens} tokens)")

                citations_data = []
                for ev in grounded.evidence:
                    source_path = ev.metadata.get('source')
                    full_text = ""
                    if source_path and os.path.exists(source_path):
                        try:
                            with open(source_path, 'r', encoding='utf-8', errors='ignore') as f:
                                full_text = f.read(32768)
                                if len(full_text) == 32768:
                                    full_text += "\n\n[... truncated ...]"
                        except:
                            pass

                    citations_data.append({
                        "name": ev.filename,
                        "text": ev.text,
                        "full_text": full_text,
                        "source_label": ev.source_label,
                    })

                encoded_citations = base64.b64encode(json.dumps(citations_data).encode('utf-8')).decode('utf-8')
                rag_html_block = f'<citations>{encoded_citations}</citations>\n\n'

        except Exception as exc:
            logger.error(f"RAG search failed: {exc}")

    effective_system_prompt = params.system_prompt


    # --- DYNAMIC CONTEXT BUDGET & MEMORY COMPACTION ---
    model = state.models[resolved_model_id]
    max_ctx = getattr(model, "ctx_size", None) or getattr(CONFIG, "ctx_size", 32768)
    reserve_tokens = min(params.max_tokens or 2048, max(256, int(max_ctx * 0.35)))

    active_history = list(history) if history else []
    if getattr(CONFIG, "supermemory_enabled", True) and conv_id and active_history:
        try:
            from oprel.knowledge.context_compactor import compact_conversation_context, estimate_messages_tokens
            from oprel.knowledge.supermemory import SuperMemoryStore
            from oprel.downloader.aliases import resolve_model_id

            recalled_facts = []
            if estimate_messages_tokens(active_history) > max_ctx - reserve_tokens - 1000:
                try:
                    async def memory_embed(text: str) -> list[float]:
                        try:
                            resolved_embed_model = resolve_model_id("nomic-embed-text")
                            res = await get_embeddings(EmbeddingParams(input=text, model=resolved_embed_model))
                            return res.embedding or []
                        except Exception:
                            return []

                    mem_store = SuperMemoryStore(embed_func=memory_embed)
                    results = await mem_store.search_memories(
                        conversation_id=conv_id,
                        query=text_prompt,
                        top_k=4,
                    )
                    if results:
                        recalled_facts = [m.get("content", "") for m in results]
                except Exception as mem_err:
                    logger.debug(f"[Compactor] SuperMemory lookup: {mem_err}")

            active_history, effective_system_prompt, was_compacted = compact_conversation_context(
                history=active_history,
                current_prompt=text_prompt,
                system_prompt=effective_system_prompt,
                max_context_tokens=max_ctx,
                reserve_generation_tokens=reserve_tokens,
                min_recent_turns=getattr(CONFIG, "supermemory_recency_window", 6),
                recalled_memories=recalled_facts,
            )
            if was_compacted:
                logger.info(f"[Compactor] Conversation compacted to stay within {max_ctx} tokens; earlier turns recalled from memory")
        except Exception as exc:
            logger.warning(f"[Compactor] Compaction fallback: {exc}")

    # --- HARD SLIDING WINDOW: FIFO TURN DROPPING TO PREVENT 400 OVERFLOW ---
    prompt_token_ceiling = max(256, max_ctx - reserve_tokens - 64)

    def _estimate_item_tokens(item: Any) -> int:
        if isinstance(item, dict):
            c = item.get("content", "")
        else:
            c = str(item)
        return int(len(str(c).split()) * 1.3) + 4

    sys_tokens = int(len(str(effective_system_prompt or "").split()) * 1.3) + 10
    new_user_tokens = int(len(str(text_prompt or "").split()) * 1.3) + 10

    current_prompt_tokens = sys_tokens + new_user_tokens + sum(_estimate_item_tokens(m) for m in active_history)

    dropped_history_count = 0
    while current_prompt_tokens > prompt_token_ceiling and active_history:
        dropped = active_history.pop(0)
        dropped_history_count += 1
        current_prompt_tokens = sys_tokens + new_user_tokens + sum(_estimate_item_tokens(m) for m in active_history)

    if dropped_history_count > 0:
        logger.info(
            f"[SlidingWindow] Dropped {dropped_history_count} oldest turns to fit within {prompt_token_ceiling} "
            f"tokens (model ctx_size: {max_ctx}). Zero 400 errors."
        )

    if current_prompt_tokens > prompt_token_ceiling:
        allowed_chars = max(100, int((prompt_token_ceiling - sys_tokens - 20) * 3.2))
        text_prompt = text_prompt[:allowed_chars] + "\n\n[...prompt truncated to fit model context window]"
        logger.warning(f"[SlidingWindow] User prompt truncated to {allowed_chars} chars to fit context ceiling")
    # --- END SLIDING WINDOW ---

    # Old chats may hold UI-only tool blocks from the parked connectors feature; never send them to a model.
    from oprel.utils.legacy import strip_legacy_tool_cards

    active_history = [
        {**m, "content": strip_legacy_tool_cards(m.get("content"))} if isinstance(m, dict) else m for m in active_history
    ]

    full_prompt = build_chat_prompt(
        resolved_model_id,
        active_history,
        effective_system_prompt,
        text_prompt,
        thinking=params.thinking,
    )

    async def _token_stream() -> AsyncIterator[str]:
        for tok in model._client.generate(
            prompt=full_prompt,
            stream=True,
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k,
            repeat_penalty=params.repeat_penalty,
            images=images if images else None,
            model=resolved_model_id,
            cache_prompt=getattr(CONFIG, "prompt_cache_enabled", True),
        ):
            yield tok
            await asyncio.sleep(0)

    if params.stream:
        async def generate_stream() -> AsyncIterator[str]:
            full_resp = rag_html_block if locals().get('rag_html_block') else ""
            try:
                start_gen_time = time_module.perf_counter()
                token_count = 0
                
                if full_resp:
                    for char in full_resp:
                        yield f"data: {json.dumps(char)}\n\n"

                async for token in _token_stream():
                    if token:
                        full_resp += token
                        yield f"data: {json.dumps(token)}\n\n"
                        token_count += 1

                end_gen_time = time_module.perf_counter()
                duration = end_gen_time - start_gen_time
                if duration > 0:
                    state.last_gen_speed = token_count / duration

                if is_persistent:
                    db.add_message(conv_id, "user", raw_prompt)
                    db.add_message(conv_id, "assistant", full_resp)
                else:
                    if conv_id not in state.ephemeral_history:
                        state.ephemeral_history[conv_id] = []
                    state.ephemeral_history[conv_id].append({"role": "user", "content": text_prompt})
                    state.ephemeral_history[conv_id].append({"role": "assistant", "content": full_resp})
                    if len(state.ephemeral_history[conv_id]) > 40:
                        state.ephemeral_history[conv_id] = state.ephemeral_history[conv_id][-40:]

                # Record into Supermemory store
                if getattr(CONFIG, "supermemory_enabled", True) and conv_id:
                    try:
                        from oprel.knowledge.supermemory import SuperMemoryStore
                        turn_idx = len(history) // 2
                        u_text = text_prompt if isinstance(text_prompt, str) else str(text_prompt)
                        asyncio.create_task(
                            SuperMemoryStore().record_turn(
                                conversation_id=conv_id,
                                user_text=u_text,
                                assistant_text=strip_legacy_tool_cards(full_resp),
                                turn_index=turn_idx,
                            )
                        )
                    except Exception as exc:
                        logger.debug(f"[Supermemory] Turn recording error: {exc}")

                yield "data: [DONE]\n\n"

                prompt_tokens_est = len(text_prompt.split()) * 1.3
                db.add_inference_log(
                    model_id=resolved_model_id,
                    prompt_tokens=int(prompt_tokens_est),
                    completion_tokens=token_count,
                    latency_ms=duration * 1000,
                    tps=token_count / duration if duration > 0 else 0,
                )
            except Exception as exc:
                yield f"data: [ERROR] {str(exc)}\n\n"

        return StreamResult(iterator=generate_stream(), conversation_id=conv_id)

    start_gen_time = time_module.perf_counter()
    generated = model._client.generate(
        prompt=full_prompt,
        max_tokens=params.max_tokens,
        temperature=params.temperature,
        top_p=params.top_p,
        top_k=params.top_k,
        repeat_penalty=params.repeat_penalty,
        stream=False,
        images=images if images else None,
        model=resolved_model_id,
        cache_prompt=getattr(CONFIG, "prompt_cache_enabled", True),
    )
    text = (rag_html_block if locals().get('rag_html_block') else "") + generated

    end_gen_time = time_module.perf_counter()
    duration = end_gen_time - start_gen_time
    if duration > 0:
        token_est = len(text.split()) * 1.3
        state.last_gen_speed = token_est / duration

    if is_persistent:
        db.add_message(conv_id, "user", raw_prompt)
        db.add_message(conv_id, "assistant", text)
    else:
        if conv_id not in state.ephemeral_history:
            state.ephemeral_history[conv_id] = []
        state.ephemeral_history[conv_id].append({"role": "user", "content": text_prompt})
        state.ephemeral_history[conv_id].append({"role": "assistant", "content": text})
        if len(state.ephemeral_history[conv_id]) > 40:
            state.ephemeral_history[conv_id] = state.ephemeral_history[conv_id][-40:]

    # Record into Supermemory store
    if getattr(CONFIG, "supermemory_enabled", True) and conv_id:
        try:
            from oprel.knowledge.supermemory import SuperMemoryStore
            turn_idx = len(history) // 2
            u_text = text_prompt if isinstance(text_prompt, str) else str(text_prompt)
            asyncio.create_task(
                SuperMemoryStore().record_turn(
                    conversation_id=conv_id,
                    user_text=u_text,
                    assistant_text=strip_legacy_tool_cards(text),
                    turn_index=turn_idx,
                )
            )
        except Exception as exc:
            logger.debug(f"[Supermemory] Turn recording error: {exc}")

    prompt_tokens_est = len(text_prompt.split()) * 1.3
    completion_tokens_est = len(text.split()) * 1.3
    token_est = len(text.split()) * 1.3
    db.add_inference_log(
        model_id=resolved_model_id,
        prompt_tokens=int(prompt_tokens_est),
        completion_tokens=int(completion_tokens_est),
        latency_ms=duration * 1000,
        tps=token_est / duration if duration > 0 else 0,
    )

    return GenerateResult(
        text=text,
        model_id=resolved_model_id,
        conversation_id=conv_id,
        message_count=len(history) + 2,
    )


_embedding_load_lock = asyncio.Lock()
EMBEDDING_READY_TIMEOUT_SECONDS = 90


async def get_embeddings(params: EmbeddingParams) -> EmbeddingResult:
    state = get_state()
    from oprel.downloader.aliases import resolve_model_id

    resolved_id = resolve_model_id(params.model)

    # Loading can take many seconds.  Do it off the event loop (so chat, approvals and the UI stay
    # responsive) and let concurrent callers wait for the same load instead of racing it.
    async with _embedding_load_lock:
        await asyncio.to_thread(load_model, resolved_id)

        if resolved_id not in state.models:
            logger.error(f"Model '{resolved_id}' not found after loading. Available: {list(state.models.keys())}")
            raise RuntimeError(f"Model '{resolved_id}' failed to stay in cache")

        model = state.models[resolved_id]

        if hasattr(model, "_process") and model._process:
            backend_port = model._process.port
            deadline = time_module.time() + EMBEDDING_READY_TIMEOUT_SECONDS
            ready = False
            while time_module.time() < deadline:
                try:
                    async with httpx.AsyncClient(timeout=1.0) as hc:
                        probe = await hc.get(f"http://127.0.0.1:{backend_port}/health")
                        # llama-server answers 503 while the model is still loading
                        if probe.status_code < 500:
                            ready = True
                            break
                except Exception:
                    pass
                await asyncio.sleep(0.25)

            if not ready:
                raise RuntimeError(f"Embedding backend on port {backend_port} did not become ready in time")

    is_single = isinstance(params.input, str)
    inputs = [params.input] if is_single else params.input
    embeddings: list[list[float]] = []
    backend_port = model._process.port

    async def embed_chunk(hc: httpx.AsyncClient, chunk: str) -> list[float]:
        last_exc: Exception | None = None
        for endpoint, payload in [
            (f"http://127.0.0.1:{backend_port}/v1/embeddings", {"input": chunk, "model": "nomic-embed-text"}),
            (f"http://127.0.0.1:{backend_port}/embedding", {"content": chunk}),
        ]:
            try:
                resp = await hc.post(endpoint, json=payload, timeout=30.0)
                resp.raise_for_status()
                break
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in (404, 501):
                    last_exc = exc
                    continue
                raise
        else:
            # last_exc is always set here (every iteration hit continue and assigned it)
            assert last_exc is not None
            raise last_exc

        res = resp.json()

        if isinstance(res, dict):
            if "embedding" in res:
                vec = res["embedding"]
                if vec and isinstance(vec[0], list):
                    vec = vec[0]
                return vec
            if "data" in res:
                return res["data"][0]["embedding"]
            raise ValueError(f"Unrecognised dict response from backend: {list(res.keys())}")
        if isinstance(res, list):
            first = res[0]
            vec = first["embedding"]
            if vec and isinstance(vec[0], list):
                vec = vec[0]
            return vec
        raise ValueError(f"Unexpected response type from backend: {type(res)}")

    async def embed_text(hc: httpx.AsyncClient, text: str) -> list[float]:
        try:
            return await embed_chunk(hc, text)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 500:
                raise
            logger.info(f"Embedding: text too long ({len(text.split())} words) — switching to chunked mode")

        chunk_size = 150
        overlap = 20
        floor = 32

        while chunk_size >= floor:
            words = text.split()
            chunks = []
            start = 0
            while start < len(words):
                end = min(start + chunk_size, len(words))
                chunks.append(" ".join(words[start:end]))
                if end == len(words):
                    break
                start += chunk_size - overlap

            chunk_vecs: list[list[float]] = []
            try:
                chunk_vecs = [await embed_chunk(hc, c) for c in chunks]
                break
            except httpx.HTTPStatusError as exc2:
                if exc2.response.status_code != 500 or chunk_size <= floor:
                    raise
                chunk_size = max(chunk_size // 2, floor)
                logger.debug(f"Embedding: chunk still too large, retrying with {chunk_size} words")
        else:
            raise RuntimeError(f"Could not embed text: backend returned 500 even for {floor}-word chunks")

        dim = len(chunk_vecs[0])
        pooled = [sum(v[i] for v in chunk_vecs) / len(chunk_vecs) for i in range(dim)]
        mag = sum(x * x for x in pooled) ** 0.5
        return [x / mag for x in pooled] if mag > 0 else pooled

    async with httpx.AsyncClient(timeout=60.0) as hc:
        for text in inputs:
            vec = await embed_text(hc, text)
            embeddings.append(vec)

    if is_single:
        return EmbeddingResult(embedding=embeddings[0])
    return EmbeddingResult(embeddings=embeddings)
