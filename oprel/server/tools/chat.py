"""/v1/chat/completions handling when `enable_tools` is set."""

from __future__ import annotations

import time
from typing import Any

from fastapi import HTTPException

from oprel.server import db
from oprel.server.schemas.openai import OpenAIChatRequest
from oprel.server.tools.adapters import SamplingParams
from oprel.server.tools.connections import resolve_active_tools
from oprel.server.tools.loop import NO_TOOLS_MESSAGE, NoToolsAvailable, run_tool_loop
from oprel.server.tools.responses import json_response, prime, streaming_response
from oprel.utils.chat_templates import _get_content_text


def _resolve_conversation(request: OpenAIChatRequest, messages: list[dict[str, Any]], is_webui: bool) -> str:
    """Same conversation semantics as the plain chat path: the web UI gets a persisted chat."""
    conv_id = request.conversation_id
    last_user = next((m for m in reversed(messages) if m["role"] == "user"), None)
    if is_webui and (not conv_id or conv_id.startswith("temp-")):
        title = _get_content_text(last_user["content"]) if last_user else "New Chat"
        conv_id = db.create_conversation(request.model, title=title[:30] + "..." if len(title) > 30 else title)
        for m in messages[:-1]:
            if m["role"] in ("user", "assistant") and m.get("content"):
                db.add_message(conv_id, m["role"], m["content"])
    elif not conv_id:
        conv_id = f"ephemeral_{int(time.time() * 1000)}"

    if conv_id.startswith("chat_") and last_user is not None:
        db.add_message(conv_id, "user", last_user["content"])
    return conv_id


async def handle_tool_chat(request: OpenAIChatRequest, is_webui: bool):
    messages = [m.model_dump(exclude_none=True) for m in request.messages]
    if not any(m["role"] == "user" for m in messages):
        raise HTTPException(status_code=400, detail="At least one user message is required")

    params = SamplingParams(
        max_tokens=min(request.max_tokens or 2048, 8192),
        temperature=request.temperature,
        top_p=request.top_p,
    )
    # Fail fast (HTTP 400) when nothing is connected, before creating a conversation.
    if not resolve_active_tools(request.enable_tools or None):
        raise HTTPException(status_code=400, detail=NO_TOOLS_MESSAGE)

    conv_id = _resolve_conversation(request, messages, is_webui)
    gen = run_tool_loop(
        model=request.model,
        messages=messages,
        integration_ids=request.enable_tools or None,
        params=params,
        conversation_id=conv_id,
    )
    try:
        first, gen = await prime(gen)
    except NoToolsAvailable as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if request.stream:
        return streaming_response(first, gen, model=request.model, conversation_id=conv_id)
    return await json_response(first, gen, model=request.model) | {"conversation_id": conv_id}
