from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal

ToolKind = Literal["read", "write"]


class ToolError(Exception):
    """
    An expected, user-facing failure inside a tool handler (bad input, 404, permission denied).
    The message is shown to the model, so it must never contain credentials.
    """


@dataclass
class ToolContext:
    """Everything a handler needs to act. `secrets` is resolved from the keychain just-in-time."""

    user_id: int
    connection_id: str
    config: dict[str, Any]
    secrets: dict[str, str]
    conversation_id: str | None = None


ToolHandler = Callable[[ToolContext, dict[str, Any]], Awaitable[Any]]


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler
    kind: ToolKind = "read"
    # Shown on approval cards, e.g. "Create Task in Azure DevOps"
    title: str = ""

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    # Kept for error reporting when the model emits malformed JSON
    raw_arguments: str = ""
    parse_error: str | None = None


@dataclass
class ToolResult:
    tool_call_id: str
    name: str
    ok: bool
    # Compact JSON string handed back to the model
    content: str
    # Structured payload for the UI (links, ids). Never sent to the model.
    data: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
