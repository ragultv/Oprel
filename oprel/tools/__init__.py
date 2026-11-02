"""
Provider-agnostic tool primitives.

Integrations (oprel.integrations.*) define Tools with these types; the server's
tool engine (oprel.server.tools) binds them to connections and runs them.
This package must not import from oprel.server.
"""

from oprel.tools.base import Tool, ToolCall, ToolContext, ToolError, ToolKind, ToolResult
from oprel.tools.validation import ToolArgumentError, validate_arguments

__all__ = [
    "Tool",
    "ToolCall",
    "ToolContext",
    "ToolError",
    "ToolKind",
    "ToolResult",
    "ToolArgumentError",
    "validate_arguments",
]
