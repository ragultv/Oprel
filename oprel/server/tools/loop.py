"""
The model <-> tool agent loop, with pause/resume for manual approvals.

run_tool_loop() / resume_after_approval() are async generators of *events*:

  {"type": "tool_call",          "call_id", "tool", "title", "kind", "arguments", "approval": "auto"|"manual"}
  {"type": "approval_required",  "approval_id", "call_id", "tool", "title", "kind", "arguments", "expires_in"}
  {"type": "tool_result",        "call_id", "tool", "ok", "data", "duration_ms"}
  {"type": "content",            "text"}
  {"type": "error",              "message"}
  {"type": "done",               "finish_reason": "stop"|"approval_required"|"max_iterations"|"error",
                                 "conversation_id"}

When a manual-approval tool is called, the full loop state (messages so far, the
pending call, any queued calls) is persisted server-side in `tool_approvals`, so
the client can only approve/reject/edit — never inject arbitrary history.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, replace
from typing import Any, AsyncIterator

from oprel.server import db
from oprel.server.services.context import logger
from oprel.server.tools import store
from oprel.server.tools.adapters import ContextFullError, ModelAdapterError, SamplingParams, resolve_adapter
from oprel.server.tools.connections import BoundTool, resolve_active_tools
from oprel.server.tools.executor import error_result, execute, prepare_arguments
from oprel.tools.base import ToolCall, ToolResult
from oprel.tools.validation import ToolArgumentError
from oprel.utils.redaction import redact

MAX_ITERATIONS = 8
# Even on auto-approval, cap writes per user turn so a looping model can't spam work items.
MAX_AUTO_WRITES_PER_TURN = 10

BASE_TOOL_PROMPT = (
    "You can call tools to act in the user's connected apps. Call a tool only when the user's request "
    "needs it; otherwise answer normally. Never invent IDs, URLs or results: rely on tool output. "
    "If a tool returns an error, explain it briefly or fix the arguments and retry once. "
    "If the user rejected an action, acknowledge it and do not retry it unless asked."
)


# Small local models often end a turn with "Let me fix that." and never make the call.
_ANNOUNCES_ACTION = re.compile(
    r"\b(let me(?! know)|i'll|i will|i'm going to|i am going to|i'm now going to|next,? i)\b", re.I
)
FOLLOW_THROUGH_NUDGE = (
    "You said what you will do next but did not call a tool. Make that tool call now instead of describing it. "
    "If nothing needs to change, give your final answer."
)


WARNINGS_NUDGE = (
    "Before you answer: the last tool result has warnings you have not dealt with:\n{warnings}\n"
    "Fix them now with another tool call (use any numbers they suggest). Only if the user explicitly asked "
    "for exactly this, answer and say why you kept it."
)


def _open_warnings(state: dict[str, Any]) -> list[str]:
    """Warnings in the most recent tool result, if it succeeded."""
    last = next((e for e in reversed(state.get("events", [])) if e["type"] == "tool_result"), None)
    if not last or not last.get("ok"):
        return []
    warnings = (last.get("data") or {}).get("warnings")
    return [str(w) for w in warnings] if isinstance(warnings, list) else []


MAX_REPEATS = 2
REPEAT_MESSAGE = (
    "This is exactly the call that already failed, so it was not run again. Read that error and change "
    "the arguments it names; if you can't, tell the user what went wrong."
)


def _signature(name: str, arguments: dict[str, Any]) -> str:
    return name + json.dumps(arguments, sort_keys=True, default=str)


def _announces_action(text: str) -> bool:
    """Ends by promising an action (not by asking the user something)."""
    tail = text.strip()[-240:]
    return bool(tail) and "?" not in tail and bool(_ANNOUNCES_ACTION.search(tail))


NO_TOOLS_MESSAGE = (
    "No integration tools are available. Connect an app in Settings → Integrations, "
    "or check that its tools are enabled."
)


class NoToolsAvailable(Exception):
    pass


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _system_prompt(bound: list[BoundTool], existing: str | None, conversation_id: str | None = None) -> str:
    hints = []
    seen = set()
    for b in bound:
        if b.integration.id in seen:
            continue
        seen.add(b.integration.id)
        if b.integration.system_hint:
            hints.append(b.integration.system_hint)
        try:
            # Live state (e.g. what is already drawn): the client's history holds only chat text,
            # so without this a follow-up message can't see what earlier tool calls made.
            context = b.integration.context_hint(conversation_id)
        except Exception as exc:  # never let a state summary break the chat
            logger.warning(f"context_hint failed for {b.integration.id}: {exc}")
            context = ""
        if context:
            hints.append(context)
    parts = [BASE_TOOL_PROMPT, *hints]
    if existing and existing.strip():
        parts.append(existing.strip())
    return "\n\n".join(parts)


def _normalize_messages(messages: list[dict[str, Any]], bound: list[BoundTool],
                        conversation_id: str | None = None) -> list[dict[str, Any]]:
    """Fold any client system messages into one leading system prompt that also explains the tools."""
    existing_system = "\n\n".join(
        m["content"]
        for m in messages
        if m.get("role") == "system" and isinstance(m.get("content"), str) and m["content"].strip()
    )
    rest = [
        {k: v for k, v in m.items() if k in ("role", "content", "tool_calls", "tool_call_id", "name")}
        for m in messages
        if m.get("role") != "system"
    ]
    return [{"role": "system", "content": _system_prompt(bound, existing_system, conversation_id)}] + rest


def _tool_message(result: ToolResult) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": result.tool_call_id, "name": result.name, "content": result.content}


_COMPACT_KEYS = ("ok", "error", "summary", "warnings", "notes", "scene_version")


def _add_result(state: dict[str, Any], bound: BoundTool | None, result: ToolResult) -> None:
    """Append a tool result to the history; remember a short form if the tool allows it."""
    state["messages"].append(_tool_message(result))
    if bound is not None and bound.tool.compact_history:
        try:
            full = json.loads(result.content)
            short = json.dumps({k: full[k] for k in _COMPACT_KEYS if k in full}, ensure_ascii=False)
        except (ValueError, TypeError):
            short = result.content[:300]
        state.setdefault("compact", {})[result.tool_call_id] = short


def _history_for_model(state: dict[str, Any]) -> list[dict[str, Any]]:
    """
    The messages to send. Calls of compact_history tools before the latest tool round lose
    their arguments and most of their result: their state lives on in the latest result, and
    re-sending every past drawing step would fill a small (4k) context within a few rounds.
    """
    compact: dict[str, str] = state.get("compact") or {}
    messages = state["messages"]
    if not compact:
        return messages
    last_round = max((i for i, m in enumerate(messages) if m.get("role") == "assistant" and m.get("tool_calls")),
                     default=-1)
    out = []
    for i, m in enumerate(messages):
        if i < last_round and m.get("role") == "tool" and m.get("tool_call_id") in compact:
            m = {**m, "content": compact[m["tool_call_id"]]}
        elif i < last_round and m.get("role") == "assistant" and m.get("tool_calls"):
            m = {**m, "tool_calls": [
                {**c, "function": {**c["function"], "arguments": "{}"}} if c.get("id") in compact else c
                for c in m["tool_calls"]
            ]}
        out.append(m)
    return out


def _call_to_dict(call: ToolCall) -> dict[str, Any]:
    return asdict(call)


def _call_from_dict(d: dict[str, Any]) -> ToolCall:
    return ToolCall(**d)


def _public_args(arguments: dict[str, Any]) -> dict[str, Any]:
    return json.loads(redact(json.dumps(arguments, default=str)))


def _event_for_call(bound: BoundTool, call: ToolCall, arguments: dict[str, Any], approval: str) -> dict[str, Any]:
    return {
        "type": "tool_call",
        "call_id": call.id,
        "tool": call.name,
        "title": bound.tool.title or call.name,
        "kind": bound.tool.kind,
        "integration": bound.integration.id,
        "arguments": _public_args(arguments),
        "approval": approval,
    }


def _event_for_result(result: ToolResult) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "call_id": result.tool_call_id,
        "tool": result.name,
        "ok": result.ok,
        "data": result.data,
        "duration_ms": round(result.duration_ms, 1),
    }


def _persist_final(state: dict[str, Any], content: str) -> None:
    conv_id = state.get("conversation_id")
    if not conv_id or not conv_id.startswith("chat_"):
        return
    try:
        metadata = {"tool_events": state.get("events", [])} if state.get("events") else None
        db.add_message(conv_id, "assistant", content, metadata=metadata)
    except Exception as exc:
        logger.warning(f"Could not persist tool-loop reply: {exc}")


# ---------------------------------------------------------------------------
# core
# ---------------------------------------------------------------------------

async def _drive(state: dict[str, Any], bound: list[BoundTool]) -> AsyncIterator[dict[str, Any]]:
    """
    Continue the loop from `state`: first drain queued tool calls, then alternate
    model turns and tool executions until the model answers, needs approval, or
    hits the iteration cap.
    """
    by_name = {b.name: b for b in bound}
    schemas = [b.tool.to_openai_schema() for b in bound]
    params = SamplingParams(**state["params"])
    conv_id = state.get("conversation_id")

    def record(event: dict[str, Any]) -> dict[str, Any]:
        if event["type"] in ("tool_call", "tool_result", "approval_required"):
            state.setdefault("events", []).append(event)
        return event

    async def process_queue() -> AsyncIterator[dict[str, Any]]:
        """Run queued calls in order. Stops (leaving the rest queued) at the first that needs approval."""
        while state["queue"]:
            item = state["queue"][0]
            call = _call_from_dict(item["call"])
            b = by_name.get(call.name)
            if b is None:
                state["queue"].pop(0)
                result = error_result(call, f"Tool '{call.name}' is not available. Available tools: {', '.join(by_name) or 'none'}.")
                state["messages"].append(_tool_message(result))
                yield record(_event_for_result(result))
                continue
            try:
                arguments = prepare_arguments(b, call)
            except ToolArgumentError as exc:
                state["queue"].pop(0)
                result = error_result(call, f"Invalid arguments: {exc}")
                state["messages"].append(_tool_message(result))
                yield record({**_event_for_call(b, call, call.arguments, "auto"), "invalid": True})
                yield record(_event_for_result(result))
                continue

            needs_manual = b.approval_mode == "manual" or (
                b.tool.kind == "write"
                and b.integration.requires_connection
                and state["auto_writes"] >= MAX_AUTO_WRITES_PER_TURN
            )
            if needs_manual:
                state["queue"].pop(0)
                approval_id = store.create_approval(
                    conversation_id=conv_id,
                    connection_id=b.connection["id"],
                    tool_call_id=call.id,
                    tool_name=call.name,
                    arguments=arguments,
                    loop_state={**state, "pending_call": _call_to_dict(call)},
                )
                yield record({
                    "type": "approval_required",
                    "approval_id": approval_id,
                    "call_id": call.id,
                    "tool": call.name,
                    "title": b.tool.title or call.name,
                    "kind": b.tool.kind,
                    "integration": b.integration.id,
                    "arguments": _public_args(arguments),
                    "expires_in": int(store.APPROVAL_TTL.total_seconds()),
                    "reason": "policy" if b.approval_mode == "manual" else "write_limit",
                })
                state["paused"] = True
                return

            state["queue"].pop(0)
            signature = _signature(call.name, arguments)
            if signature in state.get("failed_calls", []):
                # Small models resend a failing call unchanged until the context is full.
                state["repeats"] = state.get("repeats", 0) + 1
                result = error_result(call, REPEAT_MESSAGE)
                _add_result(state, None, result)
                yield record({**_event_for_call(b, call, arguments, "auto"), "repeat": True})
                yield record(_event_for_result(result))
                continue
            yield record(_event_for_call(b, call, arguments, "auto"))
            result = await execute(b, call, arguments, conversation_id=conv_id, approval="auto")
            if b.tool.kind == "write" and b.integration.requires_connection:
                state["auto_writes"] += 1
            if not result.ok:
                state.setdefault("failed_calls", []).append(signature)
                state["last_error"] = (result.data or {}).get("error", "")
            _add_result(state, b, result)
            yield record(_event_for_result(result))

    async for ev in process_queue():
        yield ev
    if state.get("paused"):
        yield {"type": "done", "finish_reason": "approval_required", "conversation_id": conv_id}
        return

    try:
        adapter = resolve_adapter(state["model"])
    except Exception as exc:  # unknown provider, model failed to load, ...
        yield {"type": "error", "message": f"Could not start the model: {redact(exc)}"}
        yield {"type": "done", "finish_reason": "error", "conversation_id": conv_id}
        return

    max_iterations = max([MAX_ITERATIONS, *(b.integration.max_tool_iterations for b in bound)])
    while state["iteration"] < max_iterations:
        state["iteration"] += 1
        try:
            turn = await adapter.complete(_history_for_model(state), schemas, params)
        except ContextFullError as exc:
            succeeded = [e for e in state.get("events", []) if e["type"] == "tool_result" and e.get("ok")]
            if not succeeded:
                yield {"type": "error", "message": str(exc)}
                yield {"type": "done", "finish_reason": "error", "conversation_id": conv_id}
                return
            # The work is done (the tool cards show it); only the closing sentence didn't fit.
            last = succeeded[-1].get("data") or {}
            text = f"Done. {last['summary']}." if last.get("summary") else "Done: see the steps above."
            text += " (The model ran out of context before it could write its own summary.)"
            yield {"type": "content", "text": text}
            _persist_final(state, state.get("text", "") + text)
            yield {"type": "done", "finish_reason": "stop", "conversation_id": conv_id}
            return
        except ModelAdapterError as exc:
            yield {"type": "error", "message": str(exc)}
            yield {"type": "done", "finish_reason": "error", "conversation_id": conv_id}
            return

        if not turn.tool_calls and not state.get("nudged") and _announces_action(turn.content or "") \
                and state["iteration"] < max_iterations:
            # One chance to do what it just announced. The text stays visible, like text sent
            # alongside a tool call; approval rules still apply to whatever it calls next.
            state["nudged"] = True
            state["messages"].append(turn.as_message())
            state["messages"].append({"role": "user", "content": FOLLOW_THROUGH_NUDGE})
            state["text"] = state.get("text", "") + turn.content + "\n\n"
            yield {"type": "content", "text": turn.content + "\n\n"}
            continue

        warnings = _open_warnings(state)
        if not turn.tool_calls and warnings and not state.get("warned") and state["iteration"] < max_iterations:
            # Small models reply "done" right past warnings (a 4 m-wide 'vase'). One chance to fix
            # them. The premature reply isn't shown: the user sees the corrected answer instead.
            state["warned"] = True
            state["messages"].append(turn.as_message())
            state["messages"].append({"role": "user", "content": WARNINGS_NUDGE.format(
                warnings="\n".join(f"- {w}" for w in warnings[:5]))})
            continue

        if not turn.tool_calls:
            text = turn.content or ""
            if not text.strip() and not state.get("events") and not state.get("text"):
                # Never leave the user looking at an empty reply.
                text = "The model returned an empty reply. Try again, rephrase, or switch to a larger model."
            yield {"type": "content", "text": text}
            _persist_final(state, state.get("text", "") + text)
            yield {"type": "done", "finish_reason": "stop", "conversation_id": conv_id}
            return

        state["messages"].append(turn.as_message())
        if turn.content:
            # Text the model wrote alongside its tool calls ("Let me create that…")
            state["text"] = state.get("text", "") + turn.content + "\n\n"
            yield {"type": "content", "text": turn.content + "\n\n"}
        state["queue"] = [{"call": _call_to_dict(c)} for c in turn.tool_calls]

        async for ev in process_queue():
            yield ev
        if state.get("paused"):
            yield {"type": "done", "finish_reason": "approval_required", "conversation_id": conv_id}
            return
        if state.get("repeats", 0) >= MAX_REPEATS:
            text = ("I couldn't finish: the model kept sending a tool call that fails. "
                    f"The last error was: {state.get('last_error') or 'unknown'}")
            yield {"type": "content", "text": text}
            _persist_final(state, state.get("text", "") + text)
            yield {"type": "done", "finish_reason": "error", "conversation_id": conv_id}
            return

    text = "I stopped after several tool steps without finishing. Tell me how you'd like to continue."
    yield {"type": "content", "text": text}
    _persist_final(state, state.get("text", "") + text)
    yield {"type": "done", "finish_reason": "max_iterations", "conversation_id": conv_id}


async def run_tool_loop(
    *,
    model: str,
    messages: list[dict[str, Any]],
    integration_ids: list[str] | None,
    params: SamplingParams,
    conversation_id: str | None,
) -> AsyncIterator[dict[str, Any]]:
    bound = resolve_active_tools(integration_ids)
    if not bound:
        raise NoToolsAvailable(NO_TOOLS_MESSAGE)
    caps = [b.integration.max_temperature for b in bound if b.integration.max_temperature is not None]
    if caps:
        params = replace(params, temperature=min([params.temperature, *caps]) if params.temperature is not None
                         else min(caps))
    state: dict[str, Any] = {
        "model": model,
        "messages": _normalize_messages(messages, bound, conversation_id),
        "integration_ids": integration_ids,
        "params": asdict(params),
        "conversation_id": conversation_id,
        "iteration": 0,
        "auto_writes": 0,
        "queue": [],
        "events": [],
    }
    async for ev in _drive(state, bound):
        yield ev


class ApprovalError(ValueError):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


async def resume_after_approval(
    approval_id: str,
    *,
    approve: bool,
    edited_arguments: dict[str, Any] | None = None,
    reason: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    approval = store.get_approval(approval_id)
    if not approval:
        raise ApprovalError("Approval not found", 404)
    if approval["status"] != "pending":
        raise ApprovalError(f"This action was already {approval['status']}", 409)

    state = approval["loop_state"]
    state.pop("paused", None)
    call = _call_from_dict(state.pop("pending_call"))
    conv_id = state.get("conversation_id")
    bound = resolve_active_tools(state.get("integration_ids"))
    b = next((x for x in bound if x.name == call.name), None)

    arguments = approval["arguments"]
    if approve and b is not None and edited_arguments is not None:
        try:
            arguments = prepare_arguments(b, ToolCall(call.id, call.name, edited_arguments))
        except ToolArgumentError as exc:
            # Validate before claiming, so a typo in the edit form doesn't consume the approval.
            raise ApprovalError(f"Invalid edited arguments: {exc}") from exc

    if not store.claim_approval(approval_id, "approved" if approve else "rejected"):
        raise ApprovalError("This action was already decided", 409)

    def record(event: dict[str, Any]) -> dict[str, Any]:
        state.setdefault("events", []).append(event)
        return event

    if not approve:
        note = "The user rejected this action."
        if reason:
            note += f" Their note: {reason[:500]}"
        result = error_result(call, note)
        result.data = {"rejected": True, **({"reason": reason[:500]} if reason else {})}
        yield record({"type": "tool_result", "call_id": call.id, "tool": call.name, "ok": False,
                      "data": result.data, "duration_ms": 0, "rejected": True})
        store.add_audit_entry(conversation_id=conv_id, connection_id=approval.get("connection_id"),
                              tool_name=call.name, arguments=approval["arguments"], status="rejected",
                              approval="manual", duration_ms=0, result_summary=note)
    elif b is None:
        result = error_result(call, "This tool is no longer available (integration disconnected or tool disabled).")
        yield record(_event_for_result(result))
    else:
        yield record(_event_for_call(b, call, arguments, "manual"))
        result = await execute(b, call, arguments, conversation_id=conv_id, approval="manual")
        yield record(_event_for_result(result))

    _add_result(state, b if approve else None, result)
    async for ev in _drive(state, bound):
        yield ev

