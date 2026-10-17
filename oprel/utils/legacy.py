"""
Leftovers from the parked MCP connectors feature (see docs/internal/mcp_connectors.md).

Chats created while the feature was enabled may contain UI-only "tool call" blocks:
<toolcall>BASE64</toolcall>.  They must never be sent to a model or stored as memory.
"""

import re
from typing import Any

_CARD_RE = re.compile(r"<toolcall>[\s\S]*?</toolcall>")


def strip_legacy_tool_cards(text: Any) -> Any:
    if not isinstance(text, str) or "<toolcall>" not in text:
        return text
    return _CARD_RE.sub("", text).strip()
