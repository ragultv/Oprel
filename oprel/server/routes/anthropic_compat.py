from __future__ import annotations

import json
import time as time_module
import uuid
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from oprel.server.services.generation import GenerateParams, StreamResult, generate_text
from oprel.server.domain.state import get_state
from oprel.server import db

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class AnthropicContentBlock(BaseModel):
    type: str = "text"
    text: str = ""


class AnthropicMessage(BaseModel):
    role: str
    content: Any  # str or list of content blocks


class AnthropicMessagesRequest(BaseModel):
    model: str
    messages: list[AnthropicMessage]
    max_tokens: int = 8192
    system: Any = None  # str or list[content block] (Anthropic allows both)
    stream: bool = False
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    stop_sequences: list[str] | None = None
    metadata: dict | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _content_to_str(content: Any) -> str:
    """Normalise Anthropic message content to a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
            elif hasattr(block, "text"):
                parts.append(block.text)
        return "".join(parts)
    return str(content)


def _make_message_id() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"


async def _openai_stream_to_anthropic(model: str, stream_result: StreamResult):
    """Convert an OpenAI-format SSE stream (from providers) to Anthropic SSE."""
    msg_id = _make_message_id()
    output_tokens = 0
    buffer = ""

    yield "event: message_start\n"
    yield f"data: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'content': [], 'model': model, 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"
    yield "event: content_block_start\n"
    yield f"data: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
    yield "event: ping\n"
    yield f"data: {json.dumps({'type': 'ping'})}\n\n"

    async for chunk in stream_result.iterator:
        chunk_str = chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
        buffer += chunk_str
        while "\n\n" in buffer:
            frame, buffer = buffer.split("\n\n", 1)
            for line in frame.splitlines():
                if not line.startswith("data: "):
                    continue
                raw = line[6:]
                if raw in ("[DONE]", "") or raw.startswith("[ERROR]"):
                    continue
                # Extract text from OpenAI chunk JSON or use raw text
                text = _extract_content_from_token(raw)
                if not text:
                    continue
                output_tokens += 1
                yield "event: content_block_delta\n"
                yield f"data: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': text}})}\n\n"

    # Flush remaining buffer
    if buffer.startswith("data: "):
        raw = buffer[6:]
        if raw and raw not in ("[DONE]", "") and not raw.startswith("[ERROR]"):
            text = _extract_content_from_token(raw)
            if text:
                output_tokens += 1
                yield "event: content_block_delta\n"
                yield f"data: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': text}})}\n\n"

    yield "event: content_block_stop\n"
    yield f"data: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
    yield "event: message_delta\n"
    yield f"data: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': output_tokens}})}\n\n"
    yield "event: message_stop\n"
    yield f"data: {json.dumps({'type': 'message_stop'})}\n\n"


def _extract_content_from_token(token: str) -> str:
    """Return the text content from an OpenAI delta JSON chunk, or the token itself."""
    if token.startswith("{"):
        try:
            chunk = json.loads(token)
            content = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
            if content is not None:
                return content  # May be "" for finish chunks; caller should skip empties
        except (json.JSONDecodeError, IndexError, KeyError, TypeError):
            pass
    return token


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post("/v1/messages")
async def anthropic_messages(request: AnthropicMessagesRequest):
    """Anthropic Messages API — used by Claude Code and the Anthropic SDK."""

    system_prompt = _content_to_str(request.system) if request.system is not None else None
    conversation_history: list[dict[str, str]] = []

    # Build history from all messages except the last user turn
    for msg in request.messages[:-1]:
        if msg.role == "system":
            system_prompt = system_prompt or _content_to_str(msg.content)
        else:
            conversation_history.append(
                {"role": msg.role, "content": _content_to_str(msg.content)}
            )

    prompt = _content_to_str(request.messages[-1].content) if request.messages else ""

    # --- Provider model fast-path ---------------------------------------------
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
            messages=[{"role": m.role, "content": _content_to_str(m.content)} for m in request.messages],
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stream=request.stream,
        )
        resp = await provider_chat_proxy(p_id, proxy_body)

        if isinstance(resp, StreamResult):
            # Convert OpenAI-format provider stream → Anthropic SSE
            return StreamingResponse(
                _openai_stream_to_anthropic(request.model, resp),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        # Non-streaming provider response
        text = resp.text
        return {
            "id": _make_message_id(),
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "model": request.model,
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": sum(
                    len(_content_to_str(m.content).split()) for m in request.messages
                ),
                "output_tokens": len(text.split()),
            },
        }

    # --- Local model path -----------------------------------------------------
    conv_id = f"anthropic_{int(time_module.time() * 1000)}"

    # Pre-populate ephemeral history so generate_text sees the full conversation
    if conversation_history:
        state = get_state()
        state.ephemeral_history[conv_id] = list(conversation_history)

    gen_params = GenerateParams(
        model_id=request.model,
        prompt=prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        repeat_penalty=1.1,
        stream=request.stream,
        images=None,
        conversation_id=conv_id,
        system_prompt=system_prompt,
        reset_conversation=False,
        thinking=False,
        rag=False,
    )

    response = await generate_text(gen_params)

    # --- Streaming response ---------------------------------------------------
    if isinstance(response, StreamResult):
        async def anthropic_stream():
            msg_id = _make_message_id()
            created = int(time_module.time())

            # message_start
            yield "event: message_start\n"
            yield f"data: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'content': [], 'model': request.model, 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"

            # content_block_start
            yield "event: content_block_start\n"
            yield f"data: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"

            # ping
            yield "event: ping\n"
            yield f"data: {json.dumps({'type': 'ping'})}\n\n"

            output_tokens = 0
            buffer = ""

            async for chunk in response.iterator:
                chunk_str = chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
                buffer += chunk_str

                # Drain complete SSE frames from buffer
                while "\n\n" in buffer:
                    frame, buffer = buffer.split("\n\n", 1)
                    for line in frame.splitlines():
                        if line.startswith("data: "):
                            token = line[6:]
                            if token in ("[DONE]", "") or token.startswith("[ERROR]"):
                                continue
                            output_tokens += 1
                            yield "event: content_block_delta\n"
                            yield f"data: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': token}})}\n\n"

            # Flush remaining buffer
            if buffer.startswith("data: "):
                token = buffer[6:]
                if token and token not in ("[DONE]", "") and not token.startswith("[ERROR]"):
                    output_tokens += 1
                    yield "event: content_block_delta\n"
                    yield f"data: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': token}})}\n\n"

            # content_block_stop
            yield "event: content_block_stop\n"
            yield f"data: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"

            # message_delta
            yield "event: message_delta\n"
            yield f"data: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': output_tokens}})}\n\n"

            # message_stop
            yield "event: message_stop\n"
            yield f"data: {json.dumps({'type': 'message_stop'})}\n\n"

        return StreamingResponse(
            anthropic_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Conversation-ID": conv_id,
            },
        )

    # --- Non-streaming response -----------------------------------------------
    text = response.text
    input_tokens = sum(
        len(_content_to_str(m.content).split()) for m in request.messages
    )
    output_tokens = len(text.split())

    return {
        "id": _make_message_id(),
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": request.model,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }
