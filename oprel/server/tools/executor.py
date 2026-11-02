"""
Runs one tool call: validate -> inject credentials -> execute with timeout -> sanitize -> audit.

The model only ever sees `ToolResult.content`, which is redacted and size-capped.
Credentials are resolved from the keychain here, per call, and dropped afterwards.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from oprel.server.services.context import logger
from oprel.server.tools import store
from oprel.server.tools.connections import BoundTool, ConnectionError_, load_secrets
from oprel.tools.base import ToolCall, ToolContext, ToolError, ToolResult
from oprel.tools.validation import ToolArgumentError, validate_arguments
from oprel.utils.redaction import redact

TOOL_TIMEOUT_SECONDS = 45.0
MAX_RESULT_CHARS = 6000


def prepare_arguments(bound: BoundTool, call: ToolCall) -> dict[str, Any]:
    """Validate/coerce model-supplied arguments. Raises ToolArgumentError."""
    if call.parse_error:
        raise ToolArgumentError(f"Arguments were not valid JSON: {call.parse_error}")
    return validate_arguments(bound.tool.parameters, call.arguments)


def _encode(payload: dict[str, Any]) -> str:
    text = redact(json.dumps(payload, ensure_ascii=False, default=str))
    if len(text) > MAX_RESULT_CHARS:
        text = text[:MAX_RESULT_CHARS] + '…"[truncated]"'
    return text


def error_result(call: ToolCall, message: str) -> ToolResult:
    return ToolResult(
        tool_call_id=call.id,
        name=call.name,
        ok=False,
        content=_encode({"ok": False, "error": message}),
        data={"error": redact(message)},
    )


async def execute(
    bound: BoundTool,
    call: ToolCall,
    arguments: dict[str, Any],
    *,
    conversation_id: str | None,
    approval: str,
) -> ToolResult:
    started = time.perf_counter()
    status = "ok"
    try:
        secrets = load_secrets(bound.connection)
        ctx = ToolContext(
            user_id=bound.connection.get("user_id", store.DEFAULT_USER_ID),
            connection_id=bound.connection["id"],
            config=dict(bound.connection["config"]),
            secrets=secrets,
            conversation_id=conversation_id,
        )
        data = await asyncio.wait_for(bound.tool.handler(ctx, arguments), timeout=TOOL_TIMEOUT_SECONDS)
        data = data if isinstance(data, dict) else {"result": data}
        result = ToolResult(call.id, call.name, True, _encode({"ok": True, **data}), data=json.loads(redact(json.dumps(data, default=str))))
    except ToolError as exc:
        status = "error"
        if getattr(exc, "status_code", None) == 401:
            store.update_connection_status(bound.connection["id"], "expired", str(exc))
        result = error_result(call, str(exc))
    except ConnectionError_ as exc:
        status = "error"
        result = error_result(call, str(exc))
    except asyncio.TimeoutError:
        status = "timeout"
        result = error_result(call, f"The tool did not finish within {int(TOOL_TIMEOUT_SECONDS)}s.")
    except Exception as exc:  # unexpected bug: log details server-side, keep the model message generic
        status = "error"
        logger.error(f"Tool '{call.name}' crashed: {type(exc).__name__}: {redact(exc)}", exc_info=True)
        result = error_result(call, "The tool failed due to an internal error.")

    result.duration_ms = (time.perf_counter() - started) * 1000
    try:
        store.add_audit_entry(
            conversation_id=conversation_id,
            connection_id=bound.connection["id"],
            tool_name=call.name,
            arguments=arguments,
            status=status,
            approval=approval,
            duration_ms=result.duration_ms,
            result_summary=result.content[:500],
        )
    except Exception as exc:
        logger.warning(f"Could not write tool audit entry: {exc}")
    return result
