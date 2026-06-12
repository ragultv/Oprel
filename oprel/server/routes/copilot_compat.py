from __future__ import annotations

"""GitHub Copilot-compatible endpoint.

Mounts under the ``/copilot`` prefix so the full URL is:
  POST http://localhost:11435/copilot/v1/chat/completions

Use this VS Code settings.json snippet (Copilot custom endpoint):
  {
    "name": "oprel",
    "vendor": "customendpoint",
    "apiType": "chat-completions",
    "models": [
      {
        "id": "<your-model-id>",
        "name": "<display-name>",
        "url": "http://localhost:11435/copilot/v1/chat/completions",
        "toolCalling": true,
        "vision": true,
        "maxInputTokens": 128000,
        "maxOutputTokens": 16000
      }
    ]
  }

The only difference from the standard /v1/chat/completions route is that every
response (streaming and non-streaming) carries the ``x-request-id`` header that
GitHub Copilot's client code requires.
"""

import json
import time as time_module
import uuid
from typing import Any

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from oprel.server.schemas.openai import OpenAIChatRequest
from oprel.server.services.generation import GenerateParams, StreamResult, _resolve_model_id, generate_text
from oprel.server.services.providers import _estimate_tokens, _message_content_to_text, _truncate_text_to_tokens
from oprel.server.domain.state import get_state
from oprel.server.services.context import logger
from oprel.server import db

router = APIRouter()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _new_request_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex}"


async def _get_local_context_window(model_id: str) -> int:
    state = get_state()
    candidate_ids = [model_id]

    try:
        resolved_model_id = _resolve_model_id(model_id)
        if resolved_model_id not in candidate_ids:
            candidate_ids.append(resolved_model_id)
    except Exception:
        pass

    model = None
    for candidate_id in candidate_ids:
        model = state.models.get(candidate_id)
        if model is not None:
            break

    if model is None or not getattr(model, "_client", None):
        return 4096

    base_url = getattr(model._client, "base_url", "")
    if not base_url:
        return 4096

    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{base_url}/props")
            response.raise_for_status()
            payload = response.json()

        n_ctx = payload.get("default_generation_settings", {}).get("n_ctx")
        if isinstance(n_ctx, int) and n_ctx > 0:
            return n_ctx
    except Exception as exc:
        logger.debug(f"Copilot could not read backend props for {model_id}: {exc}")

    return 4096


def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for message in messages:
        normalized.append(
            {
                "role": str(message.get("role", "user")),
                "content": _message_content_to_text(message.get("content", "")),
            }
        )
    return normalized


def _trim_messages_to_budget(
    messages: list[dict[str, Any]],
    allowed_prompt_tokens: int,
    model_id: str,
) -> tuple[list[dict[str, str]], int, int]:
    normalized_messages = _normalize_messages(messages)
    if not normalized_messages:
        return normalized_messages, 0, 0

    def message_tokens(message: dict[str, str]) -> int:
        return _estimate_tokens(message.get("content", ""), model_id)

    total_tokens = sum(message_tokens(message) for message in normalized_messages)
    if total_tokens <= allowed_prompt_tokens:
        return normalized_messages, total_tokens, total_tokens

    if len(normalized_messages) == 1:
        only_message = normalized_messages[0]
        trimmed_content = _truncate_text_to_tokens(only_message["content"], allowed_prompt_tokens, model_id)
        trimmed_messages = [{**only_message, "content": trimmed_content}]
        trimmed_tokens = sum(message_tokens(message) for message in trimmed_messages)
        return trimmed_messages, total_tokens, trimmed_tokens

    first_message = normalized_messages[0]
    system_message = first_message if first_message.get("role") == "system" else None
    last_message = normalized_messages[-1]
    middle_messages = normalized_messages[1:-1] if system_message else normalized_messages[:-1]

    remaining_budget = allowed_prompt_tokens
    trimmed_messages: list[dict[str, str]] = []

    if system_message:
        system_budget_cap = max(
            0,
            allowed_prompt_tokens - min(message_tokens(last_message), allowed_prompt_tokens) - 200,
        )
        desired_system_budget = max(200, min(800, allowed_prompt_tokens // 4))
        system_budget = min(message_tokens(system_message), system_budget_cap, desired_system_budget)
        if system_budget > 0:
            trimmed_system = _truncate_text_to_tokens(system_message["content"], system_budget, model_id)
            trimmed_messages.append({**system_message, "content": trimmed_system})
            remaining_budget -= message_tokens(trimmed_messages[-1])

    trimmed_last = _truncate_text_to_tokens(last_message["content"], max(1, remaining_budget), model_id)
    trimmed_last_message = {**last_message, "content": trimmed_last}
    remaining_budget -= message_tokens(trimmed_last_message)

    recent_messages: list[dict[str, str]] = []
    for message in reversed(middle_messages):
        if remaining_budget <= 0:
            break

        current_tokens = message_tokens(message)
        if current_tokens <= remaining_budget:
            recent_messages.append(message)
            remaining_budget -= current_tokens
            continue

        if remaining_budget >= 120:
            trimmed_content = _truncate_text_to_tokens(message["content"], remaining_budget, model_id)
            recent_messages.append({**message, "content": trimmed_content})
            remaining_budget = 0
        break

    recent_messages.reverse()
    trimmed_messages.extend(recent_messages)
    trimmed_messages.append(trimmed_last_message)

    trimmed_tokens = sum(message_tokens(message) for message in trimmed_messages)
    return trimmed_messages, total_tokens, trimmed_tokens


def _build_non_stream_response(request_id: str, model: str, text: str, prompt: object) -> JSONResponse:
    prompt_str = str(prompt)
    body = {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time_module.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt_str.split()),
            "completion_tokens": len(text.split()),
            "total_tokens": len(prompt_str.split()) + len(text.split()),
        },
    }
    return JSONResponse(
        content=body,
        headers={
            "x-request-id": request_id,
            "Cache-Control": "no-cache",
        },
    )


async def _stream_response(request_id: str, model: str, stream_result: StreamResult) -> StreamingResponse:
    async def generator():
        buffer = ""
        token_count = 0
        try:
            async for chunk in stream_result.iterator:
                chunk_str = chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
                buffer += chunk_str
                while "\n\n" in buffer:
                    frame, buffer = buffer.split("\n\n", 1)
                    if frame.startswith("data: "):
                        token = frame[6:]
                        if token.startswith("[ERROR]"):
                            error_msg = token[7:].strip()
                            logger.error(f"Copilot stream error: {error_msg}")
                            yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': int(time_module.time()), 'model': model, 'choices': [{'index': 0, 'delta': {'content': f'[Error: {error_msg}]'}, 'finish_reason': None}]})}\n\n"
                            continue
                        if token and token != "[DONE]":
                            token_count += 1
                            yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': int(time_module.time()), 'model': model, 'choices': [{'index': 0, 'delta': {'content': token}, 'finish_reason': None}]})}\n\n"

            if buffer.startswith("data: "):
                token = buffer[6:]
                if token and token not in ("[DONE]", "") and not token.startswith("[ERROR]"):
                    token_count += 1
                    yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': int(time_module.time()), 'model': model, 'choices': [{'index': 0, 'delta': {'content': token}, 'finish_reason': None}]})}\n\n"

        except Exception as exc:
            logger.error(f"Copilot stream generator error: {exc}", exc_info=True)
            yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': int(time_module.time()), 'model': model, 'choices': [{'index': 0, 'delta': {'content': f'[Error: {exc}]'}, 'finish_reason': None}]})}\n\n"

        logger.info(f"Copilot stream finished: {token_count} tokens (request_id={request_id})")
        # Final stop chunk
        yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': int(time_module.time()), 'model': model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "x-request-id": request_id,
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post("/v1/chat/completions")
async def copilot_chat_completions(request: OpenAIChatRequest, http_request: Request):
    """OpenAI-compatible chat completions with headers required by GitHub Copilot."""

    request_id = _new_request_id()

    # Build prompt / conversation history
    prompt = request.messages[-1].content if request.messages else ""
    system_prompt = None
    conversation_history = []
    for msg in request.messages[:-1]:
        if msg.role == "system":
            system_prompt = msg.content if isinstance(msg.content, str) else str(msg.content)
        else:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            conversation_history.append({"role": msg.role, "content": content})

    # --- Provider model fast-path (stream already in OpenAI format) -----------
    p_id = request.model
    if "::" in p_id:
        p_id = p_id.split("::", 1)[0]
    elif ":" in p_id:
        p_id = p_id.split(":", 1)[0]

    provider = db.get_provider(p_id)
    if provider:
        from oprel.server.schemas.providers import ProviderChatRequest
        from oprel.server.services.providers import provider_chat_proxy

        m_name = (
            request.model.split("::", 1)[1]
            if "::" in request.model
            else (request.model.split(":", 1)[1] if ":" in request.model else None)
        )
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
        )
        resp = await provider_chat_proxy(p_id, proxy_body)
        if isinstance(resp, StreamResult):
            return StreamingResponse(
                resp.iterator,
                media_type="text/event-stream",
                headers={
                    "x-request-id": request_id,
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        return _build_non_stream_response(request_id, request.model, resp.text, prompt)

    # --- Local model path -----------------------------------------------------
    raw_messages = [{"role": message.role, "content": message.content} for message in request.messages]
    context_window = await _get_local_context_window(request.model)
    reply_reserve = min(request.max_tokens or 2048, max(512, context_window // 4))
    allowed_prompt_tokens = max(600, context_window - reply_reserve - 256)
    trimmed_messages, total_tokens, trimmed_tokens = _trim_messages_to_budget(
        raw_messages,
        allowed_prompt_tokens,
        request.model,
    )

    if total_tokens > trimmed_tokens:
        logger.warning(
            f"Copilot prompt trimmed from ~{total_tokens} to ~{trimmed_tokens} tokens "
            f"to fit local context window {context_window}"
        )

    prompt = trimmed_messages[-1]["content"] if trimmed_messages else ""
    system_prompt = None
    conversation_history = []
    for message in trimmed_messages[:-1]:
        if message["role"] == "system":
            system_prompt = message["content"]
        else:
            conversation_history.append({"role": message["role"], "content": message["content"]})

    conv_id = request.conversation_id or f"copilot_{int(time_module.time() * 1000)}"

    # Pre-populate ephemeral history so generate_text sees the full conversation
    if conversation_history:
        state = get_state()
        if conv_id not in state.ephemeral_history:
            state.ephemeral_history[conv_id] = list(conversation_history)

    # Copilot always expects streaming — force it on regardless of request field
    use_stream = True

    gen_params = GenerateParams(
        model_id=request.model,
        prompt=prompt,
        max_tokens=reply_reserve,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        repeat_penalty=request.repeat_penalty,
        stream=use_stream,
        images=None,
        conversation_id=conv_id,
        system_prompt=system_prompt,
        reset_conversation=False,
        thinking=request.thinking,
        rag=False,
    )

    logger.info(
        f"Copilot request: model={request.model}, stream={use_stream}, conv_id={conv_id}, "
        f"ctx={context_window}, prompt_tokens~={trimmed_tokens}, max_tokens={reply_reserve}"
    )

    try:
        response = await generate_text(gen_params)
    except Exception as exc:
        logger.error(f"Copilot generate_text error: {exc}", exc_info=True)
        return _build_non_stream_response(request_id, request.model, f"[Error: {exc}]", prompt)

    if isinstance(response, StreamResult):
        return await _stream_response(request_id, request.model, response)

    return _build_non_stream_response(request_id, request.model, response.text, prompt)
