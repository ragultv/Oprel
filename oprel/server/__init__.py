"""
Oprel Server Module

Production-ready FastAPI server with:
- OpenAI API compatibility
- Ollama API compatibility  
- Smart model management
- SSE streaming
- Conversation history
"""

from oprel.server.app import run_server, app

__all__ = ["run_server", "app"]
