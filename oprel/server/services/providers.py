from __future__ import annotations

import base64
import json
import os
import re
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


def _strip_citations(text: str) -> str:
    """Strip <citations>...</citations> block from prior turns to avoid context token bloat."""
    if not text or "<citations>" not in text:
        return text
    return re.sub(r"<citations>[\s\S]*?</citations>\s*", "", text).strip()


def _clean_message_text(content: Any) -> str:
    """Extract plain text from message content and strip citations XML blocks."""
    text = _message_content_to_text(content)
    return _strip_citations(text)


from oprel.utils.legacy import strip_legacy_tool_cards  # noqa: E402


async def _record_turn_supermemory(conv_id: str, user_text: str, assistant_text: str) -> None:
    """Store completed conversational turn in Supermemory long-term memory."""
    try:
        from oprel.core.config import Config
        from oprel.knowledge.supermemory import SuperMemoryStore
        from oprel.downloader.aliases import resolve_model_id
        from oprel.server.services.generation import get_embeddings, EmbeddingParams

        if not getattr(Config(), "supermemory_enabled", True) or not conv_id:
            return

        clean_user = _clean_message_text(user_text)
        clean_assistant = _clean_message_text(assistant_text)
        if not clean_user or not clean_assistant:
            return

        async def memory_embed(text: str) -> list[float]:
            try:
                resolved_model = resolve_model_id("nomic-embed-text")
                res = await get_embeddings(EmbeddingParams(input=text, model=resolved_model))
                return res.embedding or []
            except Exception:
                return []

        mem_store = SuperMemoryStore(embed_func=memory_embed)
        await mem_store.record_turn(conv_id, clean_user, clean_assistant)
    except Exception as exc:
        logger.debug(f"Failed to record turn into Supermemory: {exc}")


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
        "huggingface": "https://router.huggingface.co/v1",
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

async def _prepare_chat_context(body: Any, raw_messages: list[dict], p_type: str, conv_id: str | None = None) -> tuple[list[dict], str]:
    effective_conv_id = conv_id or getattr(body, "conversation_id", None)

    # 1. Sanitize past messages: strip citation XML blocks to keep token payload lightweight
    messages: list[dict[str, Any]] = []
    for msg in raw_messages:
        m = dict(msg)
        if m.get("role") == "assistant" and isinstance(m.get("content"), str):
            m["content"] = _clean_message_text(m["content"])
        messages.append(m)

    rag_html_block = ""

    # 2. Supermemory: recency window + semantic long-term retrieval for cloud conversations
    from oprel.core.config import Config
    config = Config()

    if getattr(config, "supermemory_enabled", True) and effective_conv_id and len(messages) > 1:
        recency_window = getattr(config, "supermemory_recency_window", 6)

        has_system = messages[0].get("role") == "system"
        system_turn = [messages[0]] if has_system else []
        dialogue = messages[1:] if has_system else messages

        if len(dialogue) > recency_window:
            active_dialogue = dialogue[-recency_window:]
            messages = system_turn + active_dialogue

            last_user_msg = next((m for m in reversed(active_dialogue) if m.get("role") == "user"), None)
            search_query = _clean_message_text(last_user_msg.get("content", "")) if last_user_msg else ""

            context_hint = None
            if len(active_dialogue) >= 2:
                context_hint = _clean_message_text(active_dialogue[-2].get("content", ""))

            if search_query:
                try:
                    from oprel.knowledge.supermemory import SuperMemoryStore
                    from oprel.downloader.aliases import resolve_model_id
                    from oprel.server.services.generation import get_embeddings, EmbeddingParams

                    async def memory_embed(text: str) -> list[float]:
                        try:
                            resolved_model = resolve_model_id("nomic-embed-text")
                            res = await get_embeddings(EmbeddingParams(input=text, model=resolved_model))
                            return res.embedding or []
                        except Exception:
                            return []

                    mem_store = SuperMemoryStore(embed_func=memory_embed)
                    recalled = await mem_store.search_memories(
                        conversation_id=effective_conv_id,
                        query=search_query,
                        top_k=getattr(config, "supermemory_top_k", 3),
                        context_hint=context_hint,
                    )
                    if recalled:
                        memory_block = mem_store.format_memory_context(recalled)
                        if messages and messages[0].get("role") == "system":
                            messages[0]["content"] = f"{messages[0]['content']}\n\n{memory_block}"
                        else:
                            messages.insert(0, {"role": "system", "content": memory_block})
                        logger.info(f"Provider Supermemory: Injected {len(recalled)} recalled historical turns into context")
                except Exception as mem_exc:
                    logger.warning(f"Provider Supermemory recall failed: {mem_exc}")

    # 3. Knowledge Grounding (RAG)
    if body.rag and messages:
        last_user_index = next((i for i in range(len(messages) - 1, -1, -1) if messages[i]["role"] == "user"), None)

        if last_user_index is not None:
            query = str(messages[last_user_index].get("content", ""))
            messages[last_user_index]["_original_content"] = query

            try:
                from oprel.knowledge.orchestration.knowledge_service import KnowledgeService
                from oprel.downloader.aliases import resolve_model_id
                from oprel.server.services.generation import get_embeddings, EmbeddingParams

                async def internal_embed(text: str | list[str], model: str | None = None) -> list[float] | list[list[float]]:
                    resolved_embed_model = resolve_model_id(model or "nomic-embed-text")
                    if isinstance(text, list):
                        res = await get_embeddings(EmbeddingParams(input=text, model=resolved_embed_model))
                        return res.embeddings or []
                    res = await get_embeddings(EmbeddingParams(input=text, model=resolved_embed_model))
                    return res.embedding or []

                provider_model = str(body.model or "")
                reply_reserve = max(body.max_tokens or 1024, 1024)
                request_token_budget = 11000 if p_type == "groq" else 12000

                existing_tokens = sum(
                    _estimate_tokens(_message_content_to_text(message.get("content", "")), provider_model)
                    for message in messages
                )
                wrapper_overhead_tokens = 120
                available_tokens = max(1000, request_token_budget - reply_reserve - existing_tokens - wrapper_overhead_tokens)

                service = KnowledgeService(embed_func=internal_embed)
                grounded = await service.ground_query(
                    query=query,
                    model_ctx_size=available_tokens,
                    conv_id=effective_conv_id,
                    force_rag=True,
                )

                if grounded.evidence:
                    messages[last_user_index]["content"] = grounded.prompt_text
                    logger.info(f"Provider RAG: Injected {len(grounded.evidence)} chunks, est ~{grounded.estimated_context_tokens} tokens")

                    citations_data = []
                    for ev in grounded.evidence:
                        source_path = ev.metadata.get('source') if ev.metadata else None
                        full_text = ""
                        if source_path and os.path.exists(source_path):
                            try:
                                with open(source_path, 'r', encoding='utf-8', errors='ignore') as f:
                                    full_text = f.read()
                            except:
                                pass

                        citations_data.append({
                            "name": ev.filename or "Unknown",
                            "text": ev.text,
                            "full_text": full_text,
                            "source_label": ev.source_label,
                        })
                    encoded_citations = base64.b64encode(json.dumps(citations_data).encode('utf-8')).decode('utf-8')
                    rag_html_block = f'<citations>{encoded_citations}</citations>\n\n'
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
        
    return messages, rag_html_block


# ─── Provider Callers (Non-Streaming) ────────────────────────────────────────

async def _call_openai(api_key: str, base_url: str, body: Any, messages: list[dict]) -> str:
    clean_messages = [{k: v for k, v in message.items() if not k.startswith("_")} for message in messages]
    async with httpx.AsyncClient(timeout=300.0) as client:
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
        return _message_text(data.get("choices", [{}])[0].get("message", {}))


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
    message = response.choices[0].message
    text = _message_text(message)
    if not text.strip():
        logger.warning(
            "Groq returned no text (finish_reason=%s, reasoning=%s)",
            response.choices[0].finish_reason,
            bool(getattr(message, "reasoning", None)),
        )
    return text


async def _call_openrouter(api_key: str, base_url: str, body: Any, messages: list[dict]) -> str:
    url = base_url or "https://openrouter.ai/api/v1"
    clean_messages = [{k: v for k, v in message.items() if not k.startswith("_")} for message in messages]
    async with httpx.AsyncClient(timeout=300.0) as client:
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
        return _message_text(data.get("choices", [{}])[0].get("message", {}))


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
        content_text = _message_content_to_text(message["content"])
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
    async with httpx.AsyncClient(timeout=300.0) as client:
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
    async with httpx.AsyncClient(timeout=300.0) as client:
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
        content_text = _message_content_to_text(message["content"])
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

def _drop_empty_messages(messages: list[dict]) -> list[dict]:
    """
    Providers reject chat messages whose content is empty or missing (e.g. 400 "messages.9.content
    must be a string or array").  Empty turns come from earlier failed replies, so drop them.
    """
    cleaned = []
    for m in messages:
        content = strip_legacy_tool_cards(m.get("content"))
        if isinstance(content, dict):  # an object is never valid message content; send it as text
            content = json.dumps(content, ensure_ascii=False)
        elif content is not None and not isinstance(content, (str, list)):
            content = str(content)
        m = {**m, "content": content}
        if content is None or (isinstance(content, str) and not content.strip()) or content == []:
            continue
        cleaned.append(m)
    return cleaned


def _message_text(message: Any) -> str:
    """Text of a chat-completion message (dict or SDK object); content parts are joined."""
    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


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
    prepared_messages, rag_html_block = await _prepare_chat_context(body, body.messages, p_type, conv_id=effective_conv_id)

    # Providers reject empty or malformed turns (left by earlier failed replies); drop them.
    prepared_messages = _drop_empty_messages(prepared_messages)

    # Save User message to history (use the original content before context wrapper expansion)
    user_msg = prepared_messages[-1] if prepared_messages else None
    if user_msg:
        db.add_message(effective_conv_id, user_msg["role"], user_msg.get("_original_content", user_msg["content"]))

    # 1. Non-Streaming Flow
    if not body.stream:
        start_gen_time = time_module.perf_counter()
        
        full_response = rag_html_block
        if p_type == "gemini":
            full_response += await _call_gemini(api_key, body, prepared_messages)
        elif p_type == "nvidia":
            full_response += await _call_nvidia(api_key, base_url, body, prepared_messages)
        elif p_type == "groq":
            full_response += await _call_groq(api_key, base_url, body, prepared_messages)
        elif p_type == "openrouter":
            full_response += await _call_openrouter(api_key, base_url, body, prepared_messages)
        else:
            full_response += await _call_openai(api_key, base_url, body, prepared_messages)

        duration = time_module.perf_counter() - start_gen_time

        if full_response and full_response.strip():
            completion_tokens = _estimate_tokens(full_response, body.model)
            tps = completion_tokens / duration if duration > 0 else 0.0
            if duration > 0:
                state = get_state()
                state.last_gen_speed = tps

            db.add_message(effective_conv_id, "assistant", full_response)
            user_orig_text = user_msg.get("_original_content", user_msg.get("content", "")) if user_msg else ""
            await _record_turn_supermemory(effective_conv_id, user_orig_text, strip_legacy_tool_cards(full_response))
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
        full_response = rag_html_block
        start_gen_time = time_module.perf_counter()
        
        if rag_html_block:
            yield f"data: {json.dumps({'choices': [{'delta': {'content': rag_html_block}}]})}\n\n"

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
            err_msg = f"{type(exc).__name__}: {exc}"
            logger.error(f"Streaming error on provider {p_type}: {err_msg}")
            yield f"data: {json.dumps({'error': f'Streaming error: {err_msg}'})}\n\n"
            return

        duration = time_module.perf_counter() - start_gen_time
        if full_response.strip():
            completion_tokens = _estimate_tokens(full_response, body.model)
            tps = completion_tokens / duration if duration > 0 else 0.0
            if duration > 0:
                state = get_state()
                state.last_gen_speed = tps

            db.add_message(conv_id, "assistant", full_response)
            user_orig_text = user_msg.get("_original_content", user_msg.get("content", "")) if user_msg else ""
            await _record_turn_supermemory(conv_id, user_orig_text, strip_legacy_tool_cards(full_response))
            db.add_inference_log(
                model_id=body.model,
                prompt_tokens=_estimate_tokens(_message_content_to_text(prepared_messages[-1]["content"] if prepared_messages else ""), body.model),
                completion_tokens=completion_tokens,
                latency_ms=duration * 1000.0,
                tps=tps,
            )

    return StreamResult(iterator=stream_generator(effective_conv_id), conversation_id=effective_conv_id)