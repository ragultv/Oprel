from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from typing import Any, AsyncIterator

from oprel.server import db
from oprel.mcp import manager as mcp_manager

logger = logging.getLogger("oprel.mcp.tool_loop")

MAX_ITERATIONS = 8
_TOOL_RE = re.compile(r"<tool_call>\s*([\s\S]*?)\s*</tool_call>", re.IGNORECASE)


# ── system prompt builder ─────────────────────────────────────────────────────

def build_mcp_system_prompt(base_system: str | None = None) -> str:
    tools = db.get_all_mcp_enabled_tools()
    if not tools:
        return base_system or ""

    by_connector: dict[str, list[dict]] = {}
    for t in tools:
        name = t.get("connector_name") or t.get("connector_id", "unknown")
        by_connector.setdefault(name, []).append(t)

    lines: list[str] = []
    for cname, ctools in by_connector.items():
        lines.append(f"\n### {cname} tools")
        for t in ctools:
            schema = t.get("input_schema", {})
            props = schema.get("properties", {})
            required = schema.get("required", [])
            lines.append(f"- **{t['name']}**: {t.get('description', '')}")
            for pname, pmeta in props.items():
                req = " (required)" if pname in required else " (optional)"
                lines.append(f"    - {pname} [{pmeta.get('type', 'any')}]{req}: {pmeta.get('description', '')}")

    mcp_block = (
        "You have access to external tools via MCP connectors.\n\n"
        "## Available Tools\n"
        + "".join(lines)
        + "\n\n## How to call a tool\n"
        "Output EXACTLY this format when you need a tool:\n\n"
        "<tool_call>\n"
        '{"connector": "<connector_name>", "tool": "<tool_name>", "arguments": {...}}\n'
        "</tool_call>\n\n"
        "Rules:\n"
        "- ONE <tool_call> block per tool invocation.\n"
        "- ALWAYS wait for <tool_result> before calling the next tool.\n"
        "- If a tool result contains a URL or link, include it in your final answer.\n"
        "- If you are unsure about a required argument, ASK the user before calling.\n"
        "- If a tool fails, explain the error clearly and suggest what the user can do.\n"
        "- When all tool calls are done, give your final answer in plain text (no <tool_call> blocks).\n"
        "- NEVER fabricate tool results.\n"
    )
    return f"{base_system}\n\n{mcp_block}" if base_system else mcp_block


# ── tool call parsing ─────────────────────────────────────────────────────────

def extract_tool_calls(text: str) -> list[dict]:
    calls = []
    for m in _TOOL_RE.finditer(text):
        try:
            p = json.loads(m.group(1).strip())
            calls.append({
                "connector": p.get("connector", ""),
                "tool": p.get("tool", ""),
                "arguments": p.get("arguments", {}),
            })
        except json.JSONDecodeError:
            pass
    return calls


def strip_tool_calls(text: str) -> str:
    return _TOOL_RE.sub("", text).strip()


def _resolve_connector_id(name: str) -> str | None:
    nl = name.lower()
    connectors = db.list_mcp_connectors()
    # exact id or name match first
    for c in connectors:
        if c["id"].lower() == nl or c["name"].lower() == nl:
            return c["id"]
    # partial match
    for c in connectors:
        if nl in c["name"].lower() or c["name"].lower() in nl:
            return c["id"]
    # fallback: match builtin_id
    for c in connectors:
        if c.get("builtin_id", "").lower() == nl:
            return c["id"]
    return None


def _fmt_result(tool_name: str, result: dict) -> str:
    parts = [b["text"] for b in result.get("content", []) if b.get("type") == "text"]
    status = "ERROR" if result.get("is_error") else "SUCCESS"
    body = "\n".join(parts) or "(empty)"
    return f"<tool_result>\ntool: {tool_name}\nstatus: {status}\nresult:\n{body}\n</tool_result>"


# ── single-turn generation (local model OR cloud provider) ────────────────────

async def _generate_single_turn(
    *,
    model_id: str,
    working_history: list[dict],
    user_message: str,
    system_prompt: str | None,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repeat_penalty: float,
    images: list[str] | None,
) -> str:
    from oprel.server.domain.state import get_state
    from oprel.server.services.generation import _resolve_model_id, build_chat_prompt
    from oprel.server.services.model_state import mark_model_used

    state = get_state()
    resolved = _resolve_model_id(model_id)

    # Load local model if not yet in state
    if resolved not in state.models:
        p_id = resolved
        if "::" in p_id:
            p_id = p_id.split("::", 1)[0]
        elif ":" in p_id:
            p_id = p_id.split(":", 1)[0]

        if not db.get_provider(p_id):
            from oprel.server.services.models import load_model
            load_model(resolved)

    if resolved in state.models:
        model = state.models[resolved]
        mark_model_used(resolved)
        full_prompt = build_chat_prompt(resolved, working_history, system_prompt, user_message)
        text = model._client.generate(
            prompt=full_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            stream=False,
            images=images or None,
            model=resolved,
        )
        return text

    # Cloud provider path — build messages and call the provider directly
    p_id = resolved
    if "::" in p_id:
        p_id = p_id.split("::", 1)[0]
    elif ":" in p_id:
        p_id = p_id.split(":", 1)[0]

    provider = db.get_provider(p_id)
    if not provider:
        raise RuntimeError(f"Model '{model_id}' is neither loaded locally nor a known provider.")

    m_name = (
        resolved.split("::", 1)[1] if "::" in resolved
        else (resolved.split(":", 1)[1] if ":" in resolved else resolved)
    )

    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(working_history)
    messages.append({"role": "user", "content": user_message})

    return await _call_provider(provider, m_name, messages, max_tokens, temperature)


async def _call_provider(
    provider: dict,
    model_name: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
) -> str:
    p_type = provider.get("type", "openai")
    api_key = provider.get("api_key", "")
    base_url = provider.get("base_url", "")

    _presets = {
        "openai": "https://api.openai.com/v1",
        "groq": "https://api.groq.com/openai/v1",
        "openrouter": "https://openrouter.ai/api/v1",
        "nvidia": "https://integrate.api.nvidia.com/v1",
    }

    class _Body:
        def __init__(self):
            self.model = model_name
            self.max_tokens = max_tokens
            self.temperature = temperature
            self.rag = False

    body = _Body()
    clean_msgs = [{k: v for k, v in m.items() if not k.startswith("_")} for m in messages]

    if p_type == "gemini":
        from oprel.server.services.providers import _call_gemini
        return await _call_gemini(api_key, body, clean_msgs)

    if p_type == "groq":
        from oprel.server.services.providers import _call_groq
        return await _call_groq(api_key, base_url or _presets["groq"], body, clean_msgs)

    if p_type == "openrouter":
        from oprel.server.services.providers import _call_openrouter
        return await _call_openrouter(api_key, base_url or _presets["openrouter"], body, clean_msgs)

    if p_type == "nvidia":
        from oprel.server.services.providers import _call_nvidia
        return await _call_nvidia(api_key, base_url or _presets["nvidia"], body, clean_msgs)

    # openai / anthropic (OpenAI-compatible) / custom
    from oprel.server.services.providers import _call_openai
    url = base_url or _presets.get(p_type, "https://api.openai.com/v1")
    return await _call_openai(api_key, url, body, clean_msgs)


# ── local-model streaming loop (called from generation service) ──────────────

async def run_tool_loop(
    *,
    model_client,
    model_id: str,
    text_prompt: str,
    history: list[dict],
    system_prompt: str | None,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repeat_penalty: float,
    conversation_id: str | None,
    images: list[str] | None,
) -> AsyncIterator[str]:
    """
    Streaming MCP tool loop for local models.
    Yields SSE chunks (data: X\\n\\n) but does NOT yield [DONE] or persist — the
    caller (generate_stream in services/generation.py) owns those.
    """
    from oprel.utils.chat_templates import format_chat_prompt

    mcp_system = build_mcp_system_prompt(system_prompt)
    working_history = list(history)
    working_user = text_prompt
    llm_resp = ""

    for _ in range(MAX_ITERATIONS):
        full_prompt = format_chat_prompt(
            model_id=model_id,
            user_message=working_user,
            system_prompt=mcp_system,
            conversation_history=working_history,
            thinking=False,
        )

        llm_resp = ""
        for token in model_client.generate(
            prompt=full_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            stream=True,
            images=images,
            model=model_id,
        ):
            llm_resp += token

        tool_calls = extract_tool_calls(llm_resp)
        if not tool_calls:
            for ch in strip_tool_calls(llm_resp):
                yield f"data: {ch}\n\n"
                await asyncio.sleep(0)
            return

        result_contexts: list[str] = []
        for tc in tool_calls:
            cid = _resolve_connector_id(tc["connector"])
            if not cid:
                result_contexts.append(_fmt_result(tc["tool"], {
                    "content": [{"type": "text", "text": f"No connected connector named '{tc['connector']}'."}],
                    "is_error": True,
                }))
                continue

            indicator = f"\n\n🔧 *Calling **{tc['connector']}** → `{tc['tool']}`...*\n\n"
            for ch in indicator:
                yield f"data: {ch}\n\n"
                await asyncio.sleep(0)

            try:
                result = await mcp_manager.call_tool(cid, tc["tool"], tc["arguments"], conversation_id)
            except RuntimeError as exc:
                result = {"content": [{"type": "text", "text": str(exc)}], "is_error": True}

            result_contexts.append(_fmt_result(tc["tool"], result))

            icon = "❌" if result.get("is_error") else "✅"
            first = next((b["text"][:300] for b in result.get("content", []) if b.get("type") == "text"), "(no output)")
            for ch in f"{icon} **{tc['tool']}**: {first}\n\n":
                yield f"data: {ch}\n\n"
                await asyncio.sleep(0)

        working_history.append({"role": "user", "content": working_user})
        working_history.append({"role": "assistant", "content": llm_resp})
        working_user = (
            "\n\n".join(result_contexts)
            + "\n\nBased on the tool results above, provide your final answer. "
            "Do not call more tools unless absolutely necessary."
        )

    # Max iterations hit — emit the last response
    msg = f"\n\n⚠️ *Reached maximum tool iterations.*\n\n{strip_tool_calls(llm_resp)}"
    for ch in msg:
        yield f"data: {ch}\n\n"
        await asyncio.sleep(0)


# ── non-streaming tool loop ───────────────────────────────────────────────────

async def run_mcp_tool_loop(request: Any, state: Any, conv_id: str) -> tuple[str, str]:
    """
    Run the full agentic MCP loop and return (final_text, conversation_id).
    Handles both local models and all cloud providers.
    """
    from oprel.server.services.generation import _resolve_model_id

    mcp_system = build_mcp_system_prompt(request.system_prompt)
    is_persistent = conv_id.startswith("chat_")

    if is_persistent:
        history = db.get_conversation_messages(conv_id)
    else:
        history = state.ephemeral_history.get(conv_id, [])

    text_prompt = request.prompt if isinstance(request.prompt, str) else json.dumps(request.prompt)
    working_history = list(history)
    working_user = text_prompt
    final_text = ""
    llm_resp = ""

    for _ in range(MAX_ITERATIONS):
        llm_resp = await _generate_single_turn(
            model_id=request.model_id,
            working_history=working_history,
            user_message=working_user,
            system_prompt=mcp_system,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repeat_penalty=request.repeat_penalty,
            images=request.images,
        )

        tool_calls = extract_tool_calls(llm_resp)
        if not tool_calls:
            final_text = strip_tool_calls(llm_resp)
            break

        result_contexts: list[str] = []
        for tc in tool_calls:
            cid = _resolve_connector_id(tc["connector"])
            if not cid:
                result_contexts.append(_fmt_result(tc["tool"], {
                    "content": [{"type": "text", "text": f"No connector named '{tc['connector']}'."}],
                    "is_error": True,
                }))
                continue
            try:
                result = await mcp_manager.call_tool(cid, tc["tool"], tc["arguments"], conv_id)
            except RuntimeError as exc:
                result = {"content": [{"type": "text", "text": str(exc)}], "is_error": True}
            result_contexts.append(_fmt_result(tc["tool"], result))

        working_history.append({"role": "user", "content": working_user})
        working_history.append({"role": "assistant", "content": llm_resp})
        working_user = (
            "\n\n".join(result_contexts)
            + "\n\nBased on the tool results above, provide your final answer. "
            "Do not call more tools unless absolutely necessary."
        )
    else:
        final_text = strip_tool_calls(llm_resp)

    _persist(conv_id, is_persistent, text_prompt, final_text, state)
    return final_text, conv_id


# ── streaming tool loop ───────────────────────────────────────────────────────

async def stream_mcp_tool_loop(request: Any, state: Any, conv_id: str) -> AsyncIterator[str]:
    """
    Stream the agentic MCP loop as SSE events.
    Yields tool-call indicators during execution then streams the final answer.
    """
    mcp_system = build_mcp_system_prompt(request.system_prompt)
    is_persistent = conv_id.startswith("chat_")

    if is_persistent:
        history = db.get_conversation_messages(conv_id)
    else:
        history = state.ephemeral_history.get(conv_id, [])

    text_prompt = request.prompt if isinstance(request.prompt, str) else json.dumps(request.prompt)
    working_history = list(history)
    working_user = text_prompt
    final_text = ""
    llm_resp = ""

    try:
        for _ in range(MAX_ITERATIONS):
            try:
                llm_resp = await _generate_single_turn(
                    model_id=request.model_id,
                    working_history=working_history,
                    user_message=working_user,
                    system_prompt=mcp_system,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    repeat_penalty=request.repeat_penalty,
                    images=request.images,
                )
            except Exception as exc:
                yield f"data: [ERROR] Generation failed: {exc}\n\n"
                return

            tool_calls = extract_tool_calls(llm_resp)

            if not tool_calls:
                final_text = strip_tool_calls(llm_resp)
                for ch in final_text:
                    yield f"data: {ch}\n\n"
                    await asyncio.sleep(0)
                break

            result_contexts: list[str] = []
            for tc in tool_calls:
                cid = _resolve_connector_id(tc["connector"])
                if not cid:
                    result_contexts.append(_fmt_result(tc["tool"], {
                        "content": [{"type": "text", "text": f"No connector named '{tc['connector']}'."}],
                        "is_error": True,
                    }))
                    continue

                # Stream tool-call indicator
                indicator = f"\n\n🔧 *Calling **{tc['connector']}** → `{tc['tool']}`...*\n\n"
                for ch in indicator:
                    yield f"data: {ch}\n\n"
                    await asyncio.sleep(0)

                try:
                    result = await mcp_manager.call_tool(cid, tc["tool"], tc["arguments"], conv_id)
                except RuntimeError as exc:
                    result = {"content": [{"type": "text", "text": str(exc)}], "is_error": True}

                result_contexts.append(_fmt_result(tc["tool"], result))

                icon = "❌" if result.get("is_error") else "✅"
                first_text = next(
                    (b["text"][:300] for b in result.get("content", []) if b.get("type") == "text"),
                    "(no output)",
                )
                summary = f"{icon} **{tc['tool']}**: {first_text}\n\n"
                for ch in summary:
                    yield f"data: {ch}\n\n"
                    await asyncio.sleep(0)

            working_history.append({"role": "user", "content": working_user})
            working_history.append({"role": "assistant", "content": llm_resp})
            working_user = (
                "\n\n".join(result_contexts)
                + "\n\nBased on the tool results above, provide your final answer. "
                "Do not call more tools unless absolutely necessary."
            )
        else:
            final_text = strip_tool_calls(llm_resp)
            msg = f"\n\n⚠️ *Reached maximum tool iterations.*\n\n{final_text}"
            for ch in msg:
                yield f"data: {ch}\n\n"
                await asyncio.sleep(0)

    finally:
        yield "data: [DONE]\n\n"
        _persist(conv_id, is_persistent, text_prompt, final_text, state)


# ── persistence helper ────────────────────────────────────────────────────────

def _persist(
    conv_id: str,
    is_persistent: bool,
    user_text: str,
    assistant_text: str,
    state: Any,
) -> None:
    if not assistant_text:
        return
    if is_persistent:
        db.add_message(conv_id, "user", user_text)
        db.add_message(conv_id, "assistant", assistant_text)
    else:
        hist = state.ephemeral_history.setdefault(conv_id, [])
        hist.extend([
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ])
        if len(hist) > 40:
            state.ephemeral_history[conv_id] = hist[-40:]
