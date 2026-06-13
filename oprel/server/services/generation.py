from __future__ import annotations

import asyncio
import json
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
                            images.append(b64)
                        except Exception:
                            pass
        return " ".join(text_parts), images, prompt

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
            load_model(resolved_model_id)

    if resolved_model_id in state.models:
        model = state.models[resolved_model_id]

        if hasattr(model, "_process") and model._process is not None:
            if not model._process.is_running():
                logger.warning(f"Backend process for {resolved_model_id} died, reloading...")
                state.models.pop(resolved_model_id, None)
                state.model_configs.pop(resolved_model_id, None)
                load_model(resolved_model_id)
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
            params = GenerateParams(
                **{**params.__dict__, "max_tokens": 8192}
            )
    else:
        if params.max_tokens > 2048:
            params = GenerateParams(
                **{**params.__dict__, "max_tokens": 2048}
            )

    context_text = ""
    if params.rag:
        try:
            from oprel.knowledge.knowledge_store import KnowledgeStore
            from oprel.knowledge.config import TOP_K
            from oprel.downloader.aliases import resolve_model_id

            async def internal_embed(text: str, model: str | None = None) -> list[float]:
                resolved_embed_model = resolve_model_id(model or "nomic-embed-text")
                res = await get_embeddings(EmbeddingParams(input=text, model=resolved_embed_model))
                return res.embedding or []

            store = KnowledgeStore(embed_func=internal_embed)
            try:
                search_results = await store.search(text_prompt, top_k=TOP_K)
            except Exception as exc:
                logger.error(f"RAG search error: {exc}", exc_info=True)
                search_results = []

            if search_results:
                context_parts = []
                for i, res in enumerate(search_results):
                    source = res.get("metadata", {}).get("filename", "Unknown source")
                    context_parts.append(f"Source [{i+1}] ({source}):\n{res['text']}")

                context_text = "\n\n".join(context_parts)
                logger.info(f"RAG: Found {len(search_results)} relevant chunks")

                text_prompt = (
                    "CONTEXT FROM LOCAL KNOWLEDGE BASE:\n"
                    "----------------------------------------\n"
                    f"{context_text}\n"
                    "----------------------------------------\n\n"
                    f"QUESTION: {text_prompt}\n\n"
                    "INSTRUCTION: Use ONLY the provided context above to answer. "
                    "Cite source labels [1], [2], etc. If the answer isn't firmly supported by the context, "
                    "state that you don't have enough information."
                )

        except Exception as exc:
            logger.error(f"RAG search failed: {exc}")

    full_prompt = build_chat_prompt(
        resolved_model_id,
        history,
        params.system_prompt,
        text_prompt,
        thinking=params.thinking,
    )

    model = state.models[resolved_model_id]

    if params.stream:
        async def generate_stream() -> AsyncIterator[str]:
            full_resp = ""
            try:
                start_gen_time = time_module.perf_counter()
                token_count = 0

                for token in model._client.generate(
                    prompt=full_prompt,
                    max_tokens=params.max_tokens,
                    temperature=params.temperature,
                    top_p=params.top_p,
                    top_k=params.top_k,
                    repeat_penalty=params.repeat_penalty,
                    stream=True,
                    images=images if images else None,
                    model=resolved_model_id,
                ):
                    full_resp += token
                    token_count += 1
                    yield f"data: {token}\n\n"

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
    text = model._client.generate(
        prompt=full_prompt,
        max_tokens=params.max_tokens,
        temperature=params.temperature,
        top_p=params.top_p,
        top_k=params.top_k,
        repeat_penalty=params.repeat_penalty,
        stream=False,
        images=images if images else None,
        model=resolved_model_id,
    )

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


async def get_embeddings(params: EmbeddingParams) -> EmbeddingResult:
    state = get_state()
    from oprel.downloader.aliases import resolve_model_id

    resolved_id = resolve_model_id(params.model)

    load_model(resolved_id)

    if resolved_id not in state.models:
        logger.error(f"Model '{resolved_id}' not found after loading. Available: {list(state.models.keys())}")
        raise RuntimeError(f"Model '{resolved_id}' failed to stay in cache")

    model = state.models[resolved_id]

    if hasattr(model, "_process") and model._process:
        backend_port = model._process.port
        deadline = time_module.time() + 15
        ready = False
        while time_module.time() < deadline:
            try:
                async with httpx.AsyncClient(timeout=1.0) as hc:
                    probe = await hc.get(f"http://127.0.0.1:{backend_port}/health")
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
