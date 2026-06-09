from __future__ import annotations

import json
import time as time_module
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from oprel.server.domain.state import get_state
from oprel.server.schemas.openai import OpenAIChatRequest, OpenAICompletionRequest
from oprel.server.services.generation import GenerateParams, StreamResult, generate_text
from oprel.server.services.context import CONFIG
from oprel.server import db

router = APIRouter()


def _is_webui_request(referer: str) -> bool:
    return "/gui/" in referer or referer.endswith("/gui")


async def _handle_chat_completions(request: OpenAIChatRequest, referer: str):
    is_webui_request = _is_webui_request(referer)

    p_id = request.model
    m_name = None
    if "::" in p_id:
        p_id, m_name = p_id.split("::", 1)
    elif ":" in p_id:
        p_id, m_name = p_id.split(":", 1)

    provider = db.get_provider(p_id)
    if provider:
        from oprel.server.schemas.providers import ProviderChatRequest
        from oprel.server.services.providers import provider_chat_proxy

        if not m_name:
            enabled = provider.get("enabled_model_ids", [])
            m_name = enabled[0] if enabled else request.model

        proxy_body = ProviderChatRequest(
            model=m_name,
            messages=[{"role": m.role, "content": m.content} for m in request.messages],
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stream=request.stream,
            conversation_id=request.conversation_id,
            rag=request.rag,
        )
        resp = await provider_chat_proxy(p_id, proxy_body)
        if isinstance(resp, StreamResult):
            return StreamingResponse(resp.iterator, media_type="text/event-stream")
        return {
            "id": f"chatcmpl-{int(time_module.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time_module.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": resp.text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": len(resp.text.split()),
                "total_tokens": len(resp.text.split()),
            },
        }

    prompt = request.messages[-1].content if request.messages else ""

    system_prompt = None
    conversation_history = []
    for msg in request.messages[:-1]:
        if msg.role == "system":
            system_prompt = msg.content if isinstance(msg.content, str) else ""
        else:
            conversation_history.append({"role": msg.role, "content": msg.content})

    gen_params = GenerateParams(
        model_id=request.model,
        prompt=prompt,
        max_tokens=request.max_tokens or 8192,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        repeat_penalty=request.repeat_penalty,
        stream=request.stream,
        images=None,
        conversation_id=request.conversation_id,
        system_prompt=system_prompt,
        reset_conversation=False,
        thinking=request.thinking,
        rag=request.rag,
    )

    if isinstance(prompt, list):
        has_image = any(
            isinstance(item, dict) and item.get("type") == "image_url"
            for item in prompt
        )
        if has_image and gen_params.max_tokens < 8192:
            gen_params = GenerateParams(**{**gen_params.__dict__, "max_tokens": 8192})

    conv_id = request.conversation_id
    if is_webui_request:
        if not conv_id:
            from oprel.utils.chat_templates import _get_content_text

            title_text = _get_content_text(prompt)
            conv_id = db.create_conversation(
                request.model,
                title=title_text[:30] + "..." if len(title_text) > 30 else title_text,
            )
            for msg in conversation_history:
                db.add_message(conv_id, msg["role"], msg["content"])
        elif conv_id.startswith("temp-"):
            from oprel.utils.chat_templates import _get_content_text

            title_text = _get_content_text(prompt)
            conv_id = db.create_conversation(
                request.model,
                title=title_text[:30] + "..." if len(title_text) > 30 else title_text,
            )
            for msg in conversation_history:
                db.add_message(conv_id, msg["role"], msg["content"])
    else:
        if not conv_id:
            conv_id = f"ephemeral_{int(time_module.time() * 1000)}"

    gen_params = GenerateParams(**{**gen_params.__dict__, "conversation_id": conv_id})

    response = await generate_text(gen_params)

    if isinstance(response, StreamResult):
        async def openai_stream_wrapper():
            request_id = f"chatcmpl-{int(time_module.time() * 1000)}"
            buffer = ""
            async for chunk in response.iterator:
                chunk_str = chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
                buffer += chunk_str
                while "\n\n" in buffer:
                    line, buffer = buffer.split("\n\n", 1)
                    if line.startswith("data: "):
                        token = line[6:]
                        if token.startswith("[ERROR]"):
                            yield f"data: {token}\n\n"
                            continue
                        if token and token != "[DONE]":
                            chunk = {
                                "id": request_id,
                                "object": "chat.completion.chunk",
                                "created": int(time_module.time()),
                                "model": request.model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": token},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"

            if buffer.startswith("data: "):
                token = buffer[6:]
                if token.startswith("[ERROR]"):
                    yield f"data: {token}\n\n"
                elif token and token != "[DONE]":
                    yield f"data: {json.dumps({
                        'id': request_id,
                        'object': 'chat.completion.chunk',
                        'created': int(time_module.time()),
                        'model': request.model,
                        'choices': [{'index': 0, 'delta': {'content': token}, 'finish_reason': None}]
                    })}\n\n"

            final_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time_module.time()),
                "model": request.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            openai_stream_wrapper(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Conversation-ID": conv_id,
            },
        )

    return {
        "id": f"chatcmpl-{int(time_module.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time_module.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response.text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(str(prompt).split()),
            "completion_tokens": len(response.text.split()),
            "total_tokens": len(str(prompt).split()) + len(response.text.split()),
        },
    }


@router.post("/v1/chat/completions")
async def v1_chat_completions(request: OpenAIChatRequest, http_request: Request):
    referer = http_request.headers.get("referer", "")
    return await _handle_chat_completions(request, referer)


@router.post("/v1/completions")
async def v1_completions(request: OpenAICompletionRequest):
    gen_params = GenerateParams(
        model_id=request.model,
        prompt=request.prompt,
        max_tokens=request.max_tokens or 512,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        repeat_penalty=request.repeat_penalty,
        stream=request.stream,
        images=None,
        conversation_id=None,
        system_prompt=None,
        reset_conversation=False,
        thinking=False,
        rag=getattr(request, "rag", False),
    )

    response = await generate_text(gen_params)

    if isinstance(response, StreamResult):
        async def openai_stream_wrapper():
            request_id = f"cmpl-{int(time_module.time() * 1000)}"
            buffer = ""
            async for chunk in response.iterator:
                chunk_str = chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
                buffer += chunk_str
                while "\n\n" in buffer:
                    line, buffer = buffer.split("\n\n", 1)
                    if line.startswith("data: "):
                        token = line[6:]
                        if token.startswith("[ERROR]"):
                            yield f"data: {token}\n\n"
                            continue
                        if token and token != "[DONE]":
                            chunk = {
                                "id": request_id,
                                "object": "text_completion",
                                "created": int(time_module.time()),
                                "model": request.model,
                                "choices": [{"text": token, "index": 0, "finish_reason": None}],
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"

            if buffer.startswith("data: "):
                token = buffer[6:]
                if token.startswith("[ERROR]"):
                    yield f"data: {token}\n\n"
                elif token and token != "[DONE]":
                    yield f"data: {json.dumps({
                        'id': request_id,
                        'object': 'text_completion',
                        'created': int(time_module.time()),
                        'model': request.model,
                        'choices': [{'text': token, 'index': 0, 'finish_reason': None}]
                    })}\n\n"

            final_chunk = {
                "id": request_id,
                "object": "text_completion",
                "created": int(time_module.time()),
                "model": request.model,
                "choices": [{"text": "", "index": 0, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(openai_stream_wrapper(), media_type="text/event-stream")

    return {
        "id": f"cmpl-{int(time_module.time() * 1000)}",
        "object": "text_completion",
        "created": int(time_module.time()),
        "model": request.model,
        "choices": [{"text": response.text, "index": 0, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": len(request.prompt.split()),
            "completion_tokens": len(response.text.split()),
            "total_tokens": len(request.prompt.split()) + len(response.text.split()),
        },
    }


@router.get("/v1/models")
async def v1_list_models():
    state = get_state()
    from oprel.downloader.aliases import OFFICIAL_REPOS, get_model_category, get_best_alias_for_repo
    from oprel.downloader.cache import list_cached_models
    from oprel.downloader.metadata import get_repo_id_from_filename, infer_repo_id_from_cache

    models = []

    category_to_tags = {
        "text-generation": ["text llm", "text"],
        "coding": ["coding"],
        "reasoning": ["reasoning"],
        "vision": ["vision llms", "text+vision"],
        "text-to-image": ["image", "text-to-image"],
    }

    skip_categories = ["embeddings", "text-to-video", "text-to-image"]

    alias_to_repo: dict[str, str] = {}
    repo_to_alias: dict[str, str] = {}
    for _cat, _alias_dict in OFFICIAL_REPOS.items():
        for _alias, _repo_id in _alias_dict.items():
            alias_to_repo[_alias] = _repo_id
            repo_to_alias[_repo_id] = _alias

    try:
        cached = list_cached_models()
        downloaded_aliases: set[str] = set()
        unregistered_cached: list[tuple[str, dict]] = []

        for model_info in cached:
            filename = model_info.get("name", "")
            if not filename:
                continue

            fname_lower = filename.lower()
            if "mmproj" in fname_lower or fname_lower.startswith("vision-") or fname_lower.startswith("clip-"):
                continue

            repo_id = get_repo_id_from_filename(CONFIG.cache_dir, filename)
            repo_id = repo_id or infer_repo_id_from_cache(CONFIG.cache_dir, filename)

            if repo_id:
                best_alias = get_best_alias_for_repo(repo_id)
                if best_alias:
                    downloaded_aliases.add(best_alias)
                else:
                    unregistered_cached.append((repo_id, model_info))
            else:
                unregistered_cached.append((filename, model_info))

        for loaded_id in list(state.models.keys()):
            best_alias = get_best_alias_for_repo(loaded_id)
            if best_alias:
                downloaded_aliases.add(best_alias)
            elif loaded_id in alias_to_repo:
                downloaded_aliases.add(loaded_id)

        for category, alias_dict in OFFICIAL_REPOS.items():
            if category in skip_categories:
                continue

            tags = category_to_tags.get(category, ["text"])

            for alias, repo_id in alias_dict.items():
                is_downloaded = alias in downloaded_aliases
                is_loaded = alias in state.models or repo_id in state.models

                models.append(
                    {
                        "id": alias,
                        "object": "model",
                        "created": int(time_module.time()),
                        "owned_by": "oprel",
                        "tags": tags,
                        "category": category,
                        "loaded": is_loaded,
                        "downloaded": is_downloaded,
                    }
                )

        all_registry_repo_ids = set(alias_to_repo.values())
        seen_unregistered: set[str] = set()

        for raw_id, model_info in unregistered_cached:
            if raw_id in all_registry_repo_ids:
                continue
            if raw_id in seen_unregistered:
                continue
            seen_unregistered.add(raw_id)

            cat = get_model_category(raw_id)

            raw_id_lower = raw_id.lower()
            is_embedding = any(kw in raw_id_lower for kw in ["embed", "embedding", "nomic-embed", "bge-m3", "minilm"])

            if is_embedding or cat in skip_categories:
                continue

            tags = category_to_tags.get(cat, ["text"])
            display_name = raw_id.split("/")[-1] if "/" in raw_id else raw_id

            models.append(
                {
                    "id": raw_id,
                    "object": "model",
                    "created": int(model_info.get("modified", datetime.now()).timestamp()),
                    "owned_by": "oprel",
                    "tags": tags,
                    "category": cat or "text-generation",
                    "loaded": raw_id in state.models,
                    "downloaded": True,
                    "name": display_name,
                }
            )

        providers = db.list_providers()
        for p in providers:
            if not p.get("enabled", True):
                continue

            p_id = p["id"]
            p_type = p["type"]
            enabled_models = p.get("enabled_model_ids", [])

            if not enabled_models:
                models.append(
                    {
                        "id": p_id,
                        "object": "model",
                        "created": int(time_module.time()),
                        "owned_by": p_id,
                        "tags": ["external", p_type],
                        "category": "external",
                        "loaded": True,
                        "downloaded": True,
                        "name": f"{p['name']} (Provider)",
                    }
                )
            else:
                for target_m_id in enabled_models:
                    composite_id = f"{p_id}::{target_m_id}"
                    models.append(
                        {
                            "id": composite_id,
                            "object": "model",
                            "created": int(time_module.time()),
                            "owned_by": p_id,
                            "tags": ["external", p_type],
                            "category": "external",
                            "loaded": True,
                            "downloaded": True,
                            "name": f"{target_m_id} ({p['name']})",
                        }
                    )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not list models for V1 API: {exc}")

    return {"object": "list", "data": models}


@router.get("/v1/health")
async def v1_health():
    from oprel.server.routes.health import health_check

    return await health_check()
