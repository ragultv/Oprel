from __future__ import annotations

import json
import time as time_module
from typing import Any, AsyncIterator

import httpx

from oprel.server import db
from oprel.server.domain.state import get_state
from oprel.server.services.context import logger
from oprel.server.services.generation import GenerateResult, StreamResult


def _estimate_tokens(text: str, model: str | None = None) -> int:
    try:
        import tiktoken

        try:
            encoding = tiktoken.encoding_for_model(model) if model else tiktoken.get_encoding("cl100k_base")
        except Exception:
            encoding = tiktoken.get_encoding("cl100k_base")

        return len(encoding.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def _truncate_text_to_tokens(text: str, token_budget: int, model: str | None = None) -> str:
    if token_budget <= 0:
        return ""

    try:
        import tiktoken

        try:
            encoding = tiktoken.encoding_for_model(model) if model else tiktoken.get_encoding("cl100k_base")
        except Exception:
            encoding = tiktoken.get_encoding("cl100k_base")

        token_ids = encoding.encode(text)
        if len(token_ids) <= token_budget:
            return text

        if token_budget < 50:
            return encoding.decode(token_ids[:token_budget])

        keep_head = token_budget // 2
        keep_tail = token_budget - keep_head
        head = encoding.decode(token_ids[:keep_head])
        tail = encoding.decode(token_ids[-keep_tail:])
        return head + "\n\n... (truncated to fit provider budget) ...\n\n" + tail
    except Exception:
        approx_chars = token_budget * 4
        if len(text) <= approx_chars:
            return text
        if approx_chars < 200:
            return text[:max(0, approx_chars)]
        half = approx_chars // 2
        return text[: half - 40] + "\n\n... (truncated to fit provider budget) ...\n\n" + text[-(half - 40) :]


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif item.get("type") == "image_url":
                    parts.append("[image]")
        return "\n".join(parts)
    return str(content)


def list_providers() -> list[dict[str, Any]]:
    return db.list_providers()


def get_provider(provider_id: str) -> dict[str, Any] | None:
    return db.get_provider(provider_id)


def upsert_provider(data: dict[str, Any]) -> dict[str, Any]:
    return db.upsert_provider(data)


def delete_provider(provider_id: str) -> dict[str, Any]:
    db.delete_provider(provider_id)
    return {"success": True, "id": provider_id}


async def fetch_provider_models(provider_id: str) -> list[str]:
    p = db.get_provider(provider_id)
    if not p:
        raise KeyError("Provider not found")

    api_key = p.get("api_key")
    base_url = p.get("base_url")
    p_type = p.get("type", "openai")

    if p_type == "nvidia":
        # Curated NVIDIA NIM chat-compatible models — returned directly without API call
        return [
    "abacusai/dracarys-llama-3.1-70b-instruct",
    "deepseek-ai/deepseek-v4-flash",
    "deepseek-ai/deepseek-v4-pro",
    "google/diffusiongemma-26b-a4b-it",
    "google/gemma-3n-e2b-it",
    "google/gemma-3n-e4b-it",
    "google/gemma-4-31b-it",
    "meta/llama-3.1-70b-instruct",
    "meta/llama-3.1-8b-instruct",
    "meta/llama-3.2-11b-vision-instruct",
    "meta/llama-3.2-1b-instruct",
    "meta/llama-3.2-90b-vision-instruct",
    "meta/llama-guard-4-12b",
    "microsoft/phi-4-multimodal-instruct",
    "minimaxai/minimax-m2.7",
    "minimaxai/minimax-m3",
    "mistralai/ministral-14b-instruct-2512",
    "mistralai/mistral-large-3-675b-instruct-2512",
    "mistralai/mistral-medium-3.5-128b",
    "mistralai/mistral-small-4-119b-2603",
    "nvidia/gliner-pii",
    "nvidia/ising-calibration-1-35b-a3b",
    "nvidia/llama-3.1-nemoguard-8b-content-safety",
    "nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
    "nvidia/llama-3.1-nemotron-safety-guard-8b-v3",
    "nvidia/llama-3.3-nemotron-super-49b-v1",
    "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    "nvidia/nemotron-3-content-safety",
    "nvidia/nemotron-3-nano-30b-a3b",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    "nvidia/nemotron-3-super-120b-a12b",
    "nvidia/nemotron-3-ultra-550b-a55b",
    "nvidia/nemotron-3.5-content-safety",
    "nvidia/nemotron-content-safety-reasoning-4b",
    "nvidia/nemotron-mini-4b-instruct",
    "nvidia/nemotron-nano-12b-v2-vl",
    "nvidia/nvidia-nemotron-nano-9b-v2",
    "nvidia/riva-translate-4b-instruct-v1.1",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3-next-80b-a3b-instruct",
    "qwen/qwen3.5-122b-a10b",
    "qwen/qwen3.5-397b-a17b",
    "sarvamai/sarvam-m",
    "stepfun-ai/step-3.5-flash",
    "stepfun-ai/step-3.7-flash",
    "stockmark/stockmark-2-100b-instruct",
    "upstage/solar-10.7b-instruct",
    "z-ai/glm-5.2",
]

    presets = {
        "openai": "https://api.openai.com/v1",
        "gemini": "https://generativelanguage.googleapis.com/v1beta",
        "groq": "https://api.groq.com/openai/v1",
        "openrouter": "https://openrouter.ai/api/v1",
    }

    url = base_url or presets.get(p_type, "")
    if not url and p_type != "gemini":
        raise ValueError("Base URL is missing")

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            if p_type == "gemini":
                res = await client.get(f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}")
                res.raise_for_status()
                data = res.json()
                models = [
                    m["name"].replace("models/", "")
                    for m in data.get("models", [])
                    if "generateContent" in m.get("supportedGenerationMethods", [])
                ]
                return sorted(models)

            res = await client.get(
                f"{url}/models",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "HTTP-Referer": "https://oprel.dev",
                    "X-Title": "OPREL",
                },
            )
            res.raise_for_status()
            data = res.json()
            return sorted([m["id"] for m in data.get("data", [])])
        except Exception as exc:
            logger.error(f"Failed to fetch models for {provider_id}: {str(exc)}")
            raise


# ─── Context Preparation Helper ──────────────────────────────────────────────

async def _prepare_chat_context(body: Any, raw_messages: list[dict], p_type: str) -> list[dict]:
    messages = [dict(msg) for msg in raw_messages]

    if body.rag and messages:
        last_user_index = next((i for i in range(len(messages) - 1, -1, -1) if messages[i]["role"] == "user"), None)

        if last_user_index is not None:
            query = str(messages[last_user_index].get("content", ""))
            messages[last_user_index]["_original_content"] = query

            try:
                from oprel.knowledge.knowledge_store import KnowledgeStore
                from oprel.downloader.aliases import resolve_model_id
                from oprel.server.services.generation import get_embeddings, EmbeddingParams

                async def internal_embed(text: str, model: str | None = None) -> list[float]:
                    res = await get_embeddings(EmbeddingParams(input=text, model=resolve_model_id(model or "nomic-embed-text")))
                    return res.embedding or []

                try:
                    from oprel.knowledge.config import TOP_K
                except ImportError:
                    TOP_K = 5

                store = KnowledgeStore(embed_func=internal_embed)
                search_results = await store.search(query, top_k=TOP_K)

                if search_results:
                    provider_model = str(body.model or "")
                    reply_reserve = max(body.max_tokens or 1024, 1024)
                    request_token_budget = 11000 if p_type == "groq" else 12000

                    existing_tokens = sum(
                        _estimate_tokens(_message_content_to_text(message.get("content", "")), provider_model)
                        for message in messages
                    )
                    wrapper_overhead_tokens = 120
                    available_tokens = max(0, request_token_budget - reply_reserve - existing_tokens - wrapper_overhead_tokens)

                    context_parts: list[str] = []
                    used_tokens = 0

                    for i, result in enumerate(search_results):
                        source = result.get("metadata", {}).get("filename", "Unknown source")
                        chunk = f"Source [{i+1}] ({source}):\n{result['text']}"
                        chunk_tokens = _estimate_tokens(chunk, provider_model)

                        if used_tokens + chunk_tokens > available_tokens:
                            remaining = available_tokens - used_tokens
                            if remaining > 50:
                                context_parts.append(_truncate_text_to_tokens(chunk, remaining, provider_model))
                            break

                        context_parts.append(chunk)
                        used_tokens += chunk_tokens + 8

                    if context_parts:
                        context_text = "\n\n".join(context_parts)
                        messages[last_user_index]["content"] = (
                            "CONTEXT FROM LOCAL KNOWLEDGE BASE:\n"
                            "----------------------------------------\n"
                            f"{context_text}\n"
                            "----------------------------------------\n\n"
                            f"QUESTION: {query}\n\n"
                            "INSTRUCTION: Use ONLY the provided context above to answer. "
                            "Cite source labels [1], [2], etc. If the answer isn't firmly supported by the context, "
                            "state that you don't have enough information."
                        )
                        logger.info(f"Provider RAG: Injected {len(context_parts)} chunks, ~{used_tokens} tokens")
            except Exception as exc:
                logger.error(f"Provider RAG search failed: {exc}")

    if p_type == "groq" and messages:
        provider_model = str(body.model or "")
        request_token_budget = 11000
        reply_reserve = max(body.max_tokens or 1024, 1024)
        wrapper_overhead_tokens = 120

        def compute_prompt_tokens(items: list[dict[str, Any]]) -> int:
            return sum(_estimate_tokens(_message_content_to_text(item.get("content", "")), provider_model) for item in items)

        total_tokens = compute_prompt_tokens(messages)
        allowed_prompt_tokens = max(0, request_token_budget - reply_reserve - wrapper_overhead_tokens)

        if total_tokens > allowed_prompt_tokens:
            trimmed_messages: list[dict[str, Any]] = []
            running_tokens = 0

            for message in messages[:-1]:
                message_text = _message_content_to_text(message.get("content", ""))
                message_tokens = _estimate_tokens(message_text, provider_model)

                if running_tokens + message_tokens > allowed_prompt_tokens:
                    remaining = allowed_prompt_tokens - running_tokens
                    if remaining <= 0:
                        continue
                    trimmed_messages.append({**message, "content": _truncate_text_to_tokens(message_text, remaining, provider_model)})
                    running_tokens = allowed_prompt_tokens
                    break

                trimmed_messages.append(message)
                running_tokens += message_tokens

            last_message = messages[-1]
            last_text = _message_content_to_text(last_message.get("content", ""))
            last_budget = max(50, allowed_prompt_tokens - running_tokens)
            trimmed_messages.append({**last_message, "content": _truncate_text_to_tokens(last_text, last_budget, provider_model)})

            messages = trimmed_messages
            logger.warning(
                f"Provider {p_type}: trimmed prompt from ~{total_tokens} to ~{compute_prompt_tokens(messages)} tokens to fit request budget"
            )

    return messages


# ─── Provider Callers (Non-Streaming) ────────────────────────────────────────

async def _call_openai(api_key: str, base_url: str, body: Any, messages: list[dict]) -> str:
    clean_messages = [{k: v for k, v in message.items() if not k.startswith("_")} for message in messages]
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": body.model,
                "messages": clean_messages,
                "stream": False,
                "max_tokens": body.max_tokens,
                "temperature": body.temperature,
            },
        )
        if resp.status_code != 200:
            raise RuntimeError(f"OpenAI API Error: {resp.text}")
        data = resp.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")


async def _call_nvidia(api_key: str, base_url: str, body: Any, messages: list[dict]) -> str:
    url = base_url or "https://integrate.api.nvidia.com/v1"
    return await _call_openai(api_key, url, body, messages)


async def _call_groq(api_key: str, base_url: str, body: Any, messages: list[dict]) -> str:
    from groq import AsyncGroq

    # The groq SDK appends /openai/v1 internally — strip it if already present in base_url
    groq_root: str | None = None
    if base_url:
        groq_root = base_url.rstrip("/")
        if groq_root.endswith("/openai/v1"):
            groq_root = groq_root[: -len("/openai/v1")]

    clean_messages = [{k: v for k, v in message.items() if not k.startswith("_")} for message in messages]
    client = AsyncGroq(api_key=api_key, base_url=groq_root) if groq_root else AsyncGroq(api_key=api_key)

    response = await client.chat.completions.create(
        model=body.model,
        messages=clean_messages,
        max_tokens=body.max_tokens,
        temperature=body.temperature,
        stream=False,
    )
    return response.choices[0].message.content or ""


async def _call_openrouter(api_key: str, base_url: str, body: Any, messages: list[dict]) -> str:
    url = base_url or "https://openrouter.ai/api/v1"
    clean_messages = [{k: v for k, v in message.items() if not k.startswith("_")} for message in messages]
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://oprel.ai",
                "X-Title": "OPREL",
            },
            json={
                "model": body.model,
                "messages": clean_messages,
                "stream": False,
                "max_tokens": body.max_tokens,
                "temperature": body.temperature,
            },
        )
        if resp.status_code != 200:
            raise RuntimeError(f"OpenRouter API Error: {resp.text}")
        data = resp.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")


async def _call_gemini(api_key: str, body: Any, messages: list[dict]) -> str:
    from google import genai
    from google.genai import types

    model_name = body.model
    if model_name.startswith("models/"):
        model_name = model_name.replace("models/", "")

    system_msg = next((message for message in messages if message["role"] == "system"), None)
    contents = []
    use_system_instruction = "gemma" not in body.model.lower()

    for index, message in enumerate(messages):
        if message["role"] == "system":
            continue
        role = "model" if message["role"] == "assistant" else "user"
        content_text = str(message["content"])
        if not use_system_instruction and system_msg and index == 1:
            content_text = f"{system_msg['content']}\n\n{content_text}"
        contents.append({"role": role, "parts": [{"text": content_text}]})

    system_instruction = None
    if use_system_instruction and system_msg:
        system_instruction = str(system_msg["content"])

    config = types.GenerateContentConfig(
        max_output_tokens=body.max_tokens or 4096,
        temperature=body.temperature if body.temperature is not None else 0.7,
        system_instruction=system_instruction,
    )

    client = genai.Client(api_key=api_key)
    response = await client.aio.models.generate_content(
        model=model_name,
        contents=contents,
        config=config,
    )
    return response.text or ""


# ─── Provider Streamers (Streaming) ──────────────────────────────────────────

async def _stream_openai(api_key: str, base_url: str, body: Any, messages: list[dict]) -> AsyncIterator[str]:
    clean_messages = [{k: v for k, v in message.items() if not k.startswith("_")} for message in messages]
    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream(
            "POST",
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": body.model,
                "messages": clean_messages,
                "stream": True,
                "max_tokens": body.max_tokens,
                "temperature": body.temperature,
            },
        ) as resp:
            if resp.status_code not in (200, 206):
                err_body = await resp.aread()
                raise RuntimeError(f"OpenAI Stream Error {resp.status_code}: {err_body.decode()}")
            async for line in resp.aiter_lines():
                yield line + "\n"


async def _stream_nvidia(api_key: str, base_url: str, body: Any, messages: list[dict]) -> AsyncIterator[str]:
    url = base_url or "https://integrate.api.nvidia.com/v1"
    async for line in _stream_openai(api_key, url, body, messages):
        yield line


async def _stream_groq(api_key: str, base_url: str, body: Any, messages: list[dict]) -> AsyncIterator[str]:
    from groq import AsyncGroq

    # The groq SDK appends /openai/v1 internally — strip it if already present in base_url
    groq_root: str | None = None
    if base_url:
        groq_root = base_url.rstrip("/")
        if groq_root.endswith("/openai/v1"):
            groq_root = groq_root[: -len("/openai/v1")]

    clean_messages = [{k: v for k, v in message.items() if not k.startswith("_")} for message in messages]
    client = AsyncGroq(api_key=api_key, base_url=groq_root) if groq_root else AsyncGroq(api_key=api_key)

    response_stream = await client.chat.completions.create(
        model=body.model,
        messages=clean_messages,
        max_tokens=body.max_tokens,
        temperature=body.temperature,
        stream=True,
    )

    async for chunk in response_stream:
        yield f"data: {json.dumps(chunk.model_dump())}\n\n"


async def _stream_openrouter(api_key: str, base_url: str, body: Any, messages: list[dict]) -> AsyncIterator[str]:
    url = base_url or "https://openrouter.ai/api/v1"
    clean_messages = [{k: v for k, v in message.items() if not k.startswith("_")} for message in messages]
    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream(
            "POST",
            f"{url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://oprel.ai",
                "X-Title": "OPREL",
            },
            json={
                "model": body.model,
                "messages": clean_messages,
                "stream": True,
                "max_tokens": body.max_tokens,
                "temperature": body.temperature,
            },
        ) as resp:
            if resp.status_code not in (200, 206):
                err_body = await resp.aread()
                raise RuntimeError(f"OpenRouter Stream Error {resp.status_code}: {err_body.decode()}")
            async for line in resp.aiter_lines():
                yield line + "\n"


async def _stream_gemini(api_key: str, body: Any, messages: list[dict]) -> AsyncIterator[str]:
    from google import genai
    from google.genai import types

    model_name = body.model
    if model_name.startswith("models/"):
        model_name = model_name.replace("models/", "")

    system_msg = next((message for message in messages if message["role"] == "system"), None)
    contents = []
    use_system_instruction = "gemma" not in body.model.lower()

    for index, message in enumerate(messages):
        if message["role"] == "system":
            continue
        role = "model" if message["role"] == "assistant" else "user"
        content_text = str(message["content"])
        if not use_system_instruction and system_msg and index == 1:
            content_text = f"{system_msg['content']}\n\n{content_text}"
        contents.append({"role": role, "parts": [{"text": content_text}]})

    system_instruction = None
    if use_system_instruction and system_msg:
        system_instruction = str(system_msg["content"])

    config = types.GenerateContentConfig(
        max_output_tokens=body.max_tokens or 4096,
        temperature=body.temperature if body.temperature is not None else 0.7,
        system_instruction=system_instruction,
    )

    client = genai.Client(api_key=api_key)
    try:
        response_stream = await client.aio.models.generate_content_stream(
            model=model_name,
            contents=contents,
            config=config,
        )
        async for chunk in response_stream:
            token = chunk.text or ""
            if token:
                mock_data = {
                    "candidates": [{
                        "content": {
                            "parts": [{"text": token}]
                        }
                    }]
                }
                yield f"data: {json.dumps(mock_data)}\n\n"
    except Exception as exc:
        logger.error(f"Gemini SDK Stream Error: {exc}")
        raise


# ─── Unified Chat Proxy Entrypoint ───────────────────────────────────────────

async def provider_chat_proxy(provider_id: str, body: Any) -> GenerateResult | StreamResult:
    p = db.get_provider(provider_id)
    if not p:
        raise KeyError("Provider not found")

    api_key = p.get("api_key")
    base_url = p.get("base_url")
    p_type = p.get("type", "openai")

    # Determine conversation context
    effective_conv_id = body.conversation_id
    if not effective_conv_id:
        title = "New Chat"
        if body.messages:
            first_msg = body.messages[0].get("content", "")
            if isinstance(first_msg, str) and first_msg:
                title = first_msg[:60] + ("..." if len(first_msg) > 60 else "")
        effective_conv_id = db.create_conversation(model_id=body.model, title=title)

    # Context Preparation (RAG search, prompt trimming, limits processing)
    prepared_messages = await _prepare_chat_context(body, body.messages, p_type)

    # Save User message to history (use the original content before context wrapper expansion)
    user_msg = prepared_messages[-1] if prepared_messages else None
    if user_msg:
        db.add_message(effective_conv_id, user_msg["role"], user_msg.get("_original_content", user_msg["content"]))

    # 1. Non-Streaming Flow
    if not body.stream:
        start_gen_time = time_module.perf_counter()
        if p_type == "gemini":
            full_response = await _call_gemini(api_key, body, prepared_messages)
        elif p_type == "nvidia":
            full_response = await _call_nvidia(api_key, base_url, body, prepared_messages)
        elif p_type == "groq":
            full_response = await _call_groq(api_key, base_url, body, prepared_messages)
        elif p_type == "openrouter":
            full_response = await _call_openrouter(api_key, base_url, body, prepared_messages)
        else:
            full_response = await _call_openai(api_key, base_url, body, prepared_messages)

        duration = time_module.perf_counter() - start_gen_time

        if full_response and full_response.strip():
            completion_tokens = _estimate_tokens(full_response, body.model)
            tps = completion_tokens / duration if duration > 0 else 0.0
            if duration > 0:
                state = get_state()
                state.last_gen_speed = tps

            db.add_message(effective_conv_id, "assistant", full_response)
            db.add_inference_log(
                model_id=body.model,
                prompt_tokens=_estimate_tokens(_message_content_to_text(prepared_messages[-1]["content"] if prepared_messages else ""), body.model),
                completion_tokens=completion_tokens,
                latency_ms=duration * 1000.0,
                tps=tps,
            )

        return GenerateResult(
            text=full_response or "",
            model_id=body.model,
            conversation_id=effective_conv_id,
            message_count=len(prepared_messages) + 1,
        )

    # 2. Streaming Flow
    async def stream_generator(conv_id: str) -> AsyncIterator[str]:
        full_response = ""
        start_gen_time = time_module.perf_counter()
        try:
            if p_type == "gemini":
                async for line in _stream_gemini(api_key, body, prepared_messages):
                    if line.startswith("data: "):
                        try:
                            json_data = json.loads(line[6:])
                            token = (
                                json_data.get("candidates", [{}])[0]
                                .get("content", {})
                                .get("parts", [{}])[0]
                                .get("text", "")
                            )
                            if token:
                                full_response += token
                        except Exception:
                            pass
                    yield line
            else:
                if p_type == "nvidia":
                    stream_iter = _stream_nvidia(api_key, base_url, body, prepared_messages)
                elif p_type == "groq":
                    stream_iter = _stream_groq(api_key, base_url, body, prepared_messages)
                elif p_type == "openrouter":
                    stream_iter = _stream_openrouter(api_key, base_url, body, prepared_messages)
                else:
                    stream_iter = _stream_openai(api_key, base_url, body, prepared_messages)

                async for line in stream_iter:
                    if line.startswith("data: "):
                        try:
                            chunk = json.loads(line[6:])
                            token = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if token:
                                full_response += token
                        except Exception:
                            pass
                    yield line
        except Exception as exc:
            logger.error(f"Streaming error on provider {p_type}: {exc}")
            yield f"data: {json.dumps({'error': f'Streaming error: {str(exc)}'})}\n\n"
            return

        duration = time_module.perf_counter() - start_gen_time
        if full_response.strip():
            completion_tokens = _estimate_tokens(full_response, body.model)
            tps = completion_tokens / duration if duration > 0 else 0.0
            if duration > 0:
                state = get_state()
                state.last_gen_speed = tps

            db.add_message(conv_id, "assistant", full_response)
            db.add_inference_log(
                model_id=body.model,
                prompt_tokens=_estimate_tokens(_message_content_to_text(prepared_messages[-1]["content"] if prepared_messages else ""), body.model),
                completion_tokens=completion_tokens,
                latency_ms=duration * 1000.0,
                tps=tps,
            )

    return StreamResult(iterator=stream_generator(effective_conv_id), conversation_id=effective_conv_id)