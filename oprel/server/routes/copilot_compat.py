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

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from oprel.server.schemas.openai import OpenAIChatRequest
from oprel.server.services.generation import GenerateParams, StreamResult, generate_text
from oprel.server.domain.state import get_state
from oprel.server.services.context import logger
from oprel.server import db

router = APIRouter()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _new_request_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex}"


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
        max_tokens=request.max_tokens or 8192,
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

    logger.info(f"Copilot request: model={request.model}, stream={use_stream}, conv_id={conv_id}")

    try:
        response = await generate_text(gen_params)
    except Exception as exc:
        logger.error(f"Copilot generate_text error: {exc}", exc_info=True)
        return _build_non_stream_response(request_id, request.model, f"[Error: {exc}]", prompt)

    if isinstance(response, StreamResult):
        return await _stream_response(request_id, request.model, response)

    return _build_non_stream_response(request_id, request.model, response.text, prompt)
