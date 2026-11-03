"""
Oprel Server Module

Production-ready FastAPI server with:
- OpenAI API compatibility
- Ollama API compatibility
- Smart model management
- SSE streaming
- Conversation history
"""

__all__ = ["run_server", "app"]


def __getattr__(name):
    # Lazy, so `from oprel.server import db` (used by the MCP servers and the CLI) doesn't
    # build the whole FastAPI app as a side effect.
    if name in __all__:
        import importlib

        return getattr(importlib.import_module("oprel.server.app"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
