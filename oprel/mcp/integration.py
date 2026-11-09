"""
Glue that connects the parked MCP feature to the rest of Oprel.

NOTHING in Oprel imports this file right now.  It holds the code that used to live inside
`oprel/server/services/providers.py` and `generation.py`, so turning the feature back on is a matter
of calling these functions from the places listed in docs/internal/mcp_connectors.md ("Turning it
back on").  Keep it in sync with the loop in `tool_loop.py`.

Four hooks:

1. Local models   (generation.py)   local_system_prompt() + tool_loop.tool_aware_generate()
2. Cloud models   (providers.py)    with_mcp_block() + tool_loop.tool_aware_generate_messages()
                                    with call_provider_once() as its `generate_once`
3. History        (both)            oprel.utils.legacy.strip_legacy_tool_cards() is already active
4. Startup        (server/app.py)   manager.startup() / manager.shutdown() and include routes.router
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from oprel.mcp import manager, tool_loop

logger = logging.getLogger("oprel.mcp.integration")


# -- 1. local models -----------------------------------------------------------------------------

def local_system_prompt(effective_system_prompt: Optional[str]) -> Tuple[Optional[str], bool]:
    """Append the tool description to the system prompt. Returns (prompt, tools_active)."""
    block = tool_loop.prompt_block()
    if not block:
        return effective_system_prompt, False
    return (f"{effective_system_prompt}\n\n{block}" if effective_system_prompt else block), True


# -- 2. cloud models -----------------------------------------------------------------------------

def with_mcp_block(messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], bool]:
    """Prepend the tool description as a system message when connectors are running."""
    block = tool_loop.prompt_block()
    if not block:
        return messages, False
    return [{"role": "system", "content": block}, *messages], True


def as_tool_call_text(name: Any, args: Any) -> Optional[str]:
    """Our <tool_call> text for a provider-native call named "Connector.tool" (or just "tool")."""
    if not name or not isinstance(name, str) or not isinstance(args, dict):
        return None
    if "." in name:
        connector, tool = name.split(".", 1)
    else:
        tool = name
        connector = next((t["connector_name"] for t in manager.all_live_tools() if t["name"] == tool), "")
    if not connector:
        return None
    return "<tool_call>" + json.dumps({"connector": connector, "tool": tool, "arguments": args}) + "</tool_call>"


def recover_native_tool_call(exc: Exception) -> Optional[str]:
    """
    Some hosted models (gpt-oss on Groq) emit their own native tool-call format even though we sent no
    tool definitions, and the provider rejects it (`tool_use_failed`).  The rejected call is inside the
    error; turn it back into our <tool_call> text.
    """
    body = getattr(exc, "body", None)
    failed = ""
    if isinstance(body, dict):
        err = body.get("error", body)
        failed = str(err.get("failed_generation", "")) if isinstance(err, dict) else ""
    if not failed:
        m = re.search(r"failed_generation['\"]?\s*:\s*'(.*)'\}", str(exc), re.S)
        failed = m.group(1) if m else ""
    if not failed:
        return None

    name, args = None, None
    fn = re.search(r"<function=([^>\s]+)>\s*(\{.*\})\s*(?:</function>)?", failed, re.S)
    try:
        if fn:
            name, args = fn.group(1), json.loads(fn.group(2))
        else:
            data = json.loads(failed)
            data = data[0] if isinstance(data, list) and data else data
            name, args = data.get("name"), data.get("arguments", data.get("parameters", {}))
    except Exception:
        return None
    return as_tool_call_text(name, args)


def message_text(message: Any) -> str:
    """
    Text of a chat-completion message (dict or SDK object).  A native tool call the provider accepted
    (content empty, tool_calls set) is converted to our <tool_call> text so the loop can run it.
    """
    get = (lambda k: message.get(k)) if isinstance(message, dict) else (lambda k: getattr(message, k, None))
    text = get("content") or ""
    if isinstance(text, list):
        text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
    for call in get("tool_calls") or []:
        fn = call.get("function") if isinstance(call, dict) else getattr(call, "function", None)
        name = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", None)
        raw = fn.get("arguments") if isinstance(fn, dict) else getattr(fn, "arguments", None)
        try:
            args = json.loads(raw) if isinstance(raw, str) and raw.strip() else (raw or {})
        except json.JSONDecodeError:
            args = None
        converted = as_tool_call_text(name, args)
        if converted:
            text += "\n" + converted
    return text


async def call_provider_once(
    callers: Dict[str, Callable[..., Awaitable[str]]],
    p_type: str,
    api_key: str,
    base_url: str,
    body: Any,
    messages: List[Dict[str, Any]],
    drop_empty: Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]],
) -> str:
    """
    One non-streaming provider call, used as `generate_once` by tool_loop.tool_aware_generate_messages.

    `callers` maps provider type to the existing _call_* coroutine, e.g.
      {"gemini": lambda k, b, m: _call_gemini(k, body, m), ...}  (the default is "openai")
    Native tool calls that the provider rejects are recovered from the error text.
    """
    messages = drop_empty(messages)
    call = callers.get(p_type) or callers["openai"]
    try:
        return await call(api_key, base_url, body, messages)
    except Exception as exc:  # noqa: BLE001
        recovered = recover_native_tool_call(exc)
        if recovered is None:
            raise
        logger.info("Recovered a native tool call rejected by the provider")
        return recovered
