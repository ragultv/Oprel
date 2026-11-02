"""
Turn tool-loop events into OpenAI-compatible responses.

Streaming: `content` events become normal `chat.completion.chunk` deltas, so existing
clients keep working. Tool events ride along as chunks with `choices: []` and an
`oprel_event` field, which OpenAI-style parsers simply ignore.
"""

from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator

from fastapi.responses import StreamingResponse


async def prime(gen: AsyncIterator[dict[str, Any]]) -> tuple[dict[str, Any] | None, AsyncIterator[dict[str, Any]]]:
    """
    Pull the first event *before* the HTTP response starts, so setup errors
    (no tools, approval already decided, ...) surface as proper status codes.
    """
    try:
        first = await gen.__anext__()
    except StopAsyncIteration:
        first = None
    return first, gen


async def _chain(first: dict[str, Any] | None, gen: AsyncIterator[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]:
    if first is not None:
        yield first
    async for ev in gen:
        yield ev


def _chunk(request_id: str, model: str, **extra: Any) -> str:
    payload = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        **extra,
    }
    return f"data: {json.dumps(payload, default=str)}\n\n"


def streaming_response(
    first: dict[str, Any] | None,
    gen: AsyncIterator[dict[str, Any]],
    *,
    model: str,
    conversation_id: str | None,
) -> StreamingResponse:
    request_id = f"chatcmpl-{int(time.time() * 1000)}"

    async def body() -> AsyncIterator[str]:
        finish = "stop"
        try:
            async for ev in _chain(first, gen):
                if ev["type"] == "content":
                    if ev["text"]:
                        yield _chunk(request_id, model, choices=[
                            {"index": 0, "delta": {"content": ev["text"]}, "finish_reason": None}
                        ])
                elif ev["type"] == "done":
                    finish = ev.get("finish_reason", "stop")
                    yield _chunk(request_id, model, choices=[], oprel_event=ev)
                else:
                    yield _chunk(request_id, model, choices=[], oprel_event=ev)
        except Exception as exc:  # stream already started: report in-band
            yield _chunk(request_id, model, choices=[], oprel_event={"type": "error", "message": str(exc)})
            finish = "error"
        yield _chunk(request_id, model, choices=[
            {"index": 0, "delta": {}, "finish_reason": "stop" if finish in ("stop", "max_iterations") else finish}
        ])
        yield "data: [DONE]\n\n"

    headers = {"Cache-Control": "no-cache", "Connection": "keep-alive"}
    if conversation_id:
        headers["X-Conversation-ID"] = conversation_id
    return StreamingResponse(body(), media_type="text/event-stream", headers=headers)


async def json_response(
    first: dict[str, Any] | None, gen: AsyncIterator[dict[str, Any]], *, model: str
) -> dict[str, Any]:
    content: list[str] = []
    events: list[dict[str, Any]] = []
    finish = "stop"
    async for ev in _chain(first, gen):
        if ev["type"] == "content":
            content.append(ev["text"])
        elif ev["type"] == "done":
            finish = ev.get("finish_reason", "stop")
            events.append(ev)
        else:
            events.append(ev)
    text = "".join(content)
    return {
        "id": f"chatcmpl-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop" if finish in ("stop", "max_iterations") else finish,
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": len(text.split()), "total_tokens": len(text.split())},
        "oprel_events": events,
    }
