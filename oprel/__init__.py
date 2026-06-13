"""
Oprel SDK - Local-first AI Runtime
The SQLite of LLMs

Simple, embedded AI inference with zero configuration.
"""

from oprel.core.model import Model
from oprel.core.config import Config
from oprel.core.exceptions import (
    OprelError,
    ModelNotFoundError,
    MemoryError,
    BackendError,
)
from oprel.downloader.hub import download_model
from oprel.telemetry.hardware import get_hardware_info
from oprel.version import __version__

# Ollama-compatible API
from oprel.client_api import (
    Client,
    AsyncClient,
    chat,
    generate,
    generate_image,
    list,
    show,
    create,
    pull,
    delete,
    embed,  # NEW: Embedding support
)
from oprel.api_models import (
    ChatResponse,
    GenerateResponse,
    ImageResponse,
    ListResponse,
    ShowResponse,
    ModelInfo,
    Message,
)

__all__ = [
    # Core API
    "Model",
    "Config",
    # Ollama-compatible API
    "Client",
    "AsyncClient",
    "chat",
    "generate",
    "generate_image",
    "list",
    "show",
    "create",
    "pull",
    "delete",
    "embed",  # NEW: Embedding support
    # Response models
    "ChatResponse",
    "GenerateResponse",
    "ImageResponse",
    "ListResponse",
    "ShowResponse",
    "ModelInfo",
    "Message",
    # Utilities
    "download_model",
    "get_hardware_info",
    # Exceptions
    "OprelError",
    "ModelNotFoundError",
    "MemoryError",
    "BackendError",
    # Version
    "__version__",
]
