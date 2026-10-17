"""
In-memory queue of tool calls waiting for the user's decision.

The tool loop awaits `request()`; the UI polls `list_pending()` and answers via
`resolve()`.  An unanswered request is denied after the timeout, so a closed
window can never leave a call hanging or silently approved.
"""

import asyncio
import time
import uuid
from typing import Any, Dict, List, Tuple

APPROVAL_TIMEOUT_SECONDS = 120

# id -> (public description, future resolving to "once" | "always" | "deny")
_pending: Dict[str, Tuple[Dict[str, Any], "asyncio.Future[str]"]] = {}


async def request(connector: Dict[str, Any], tool: str, args_preview: str,
                  timeout: float = APPROVAL_TIMEOUT_SECONDS, read_only: bool = False) -> str:
    loop = asyncio.get_running_loop()
    future: "asyncio.Future[str]" = loop.create_future()
    approval_id = uuid.uuid4().hex[:12]
    _pending[approval_id] = (
        {
            "id": approval_id,
            "connector_id": connector["id"],
            "connector_name": connector["name"],
            "network": connector.get("network", "unknown"),
            "tool": tool,
            "read_only": read_only,
            "arguments": args_preview,
            "created_at": time.time(),
            "expires_in": timeout,
        },
        future,
    )
    try:
        return await asyncio.wait_for(future, timeout=timeout)
    except asyncio.TimeoutError:
        return "deny"
    finally:
        _pending.pop(approval_id, None)


def list_pending() -> List[Dict[str, Any]]:
    return [info for info, _ in _pending.values()]


def resolve(approval_id: str, decision: str) -> bool:
    entry = _pending.get(approval_id)
    if not entry or decision not in ("once", "always", "deny"):
        return False
    future = entry[1]
    if not future.done():
        future.set_result(decision)
    return True
