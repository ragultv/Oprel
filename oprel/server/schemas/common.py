from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class LoadRequest(BaseModel):
    model_id: str
    quantization: str | None = None
    max_memory_mb: int | None = None
    backend: str = "llama.cpp"


class PullRequest(BaseModel):
    model_id: str
    quantization: str | None = None


class MetricsResponse(BaseModel):
    cpu_usage: float
    ram_total_gb: float
    ram_used_gb: float
    gpu_name: str | None = None
    gpu_usage: float | None = None
    vram_total_mb: float | None = None
    vram_used_mb: float | None = None
    generation_speed: float = 0.0


class HealthResponse(BaseModel):
    status: str
    models_loaded: int
    active_conversations: int


class LoadResponse(BaseModel):
    success: bool
    model_id: str
    message: str


class UnloadRequest(BaseModel):
    model_id: str


class UnloadResponse(BaseModel):
    success: bool
    model_id: str
    message: str


class ModelInfo(BaseModel):
    model_id: str
    name: str
    size_gb: float = 0.0
    quantization: str | None = None
    backend: str = "llama.cpp"
    loaded: bool = False
    status: str = "cached"
    category: str | None = None


class RenameConversationRequest(BaseModel):
    title: str


class UserProfile(BaseModel):
    name: str
    role: str
    initials: str | None = None


class UserSettings(BaseModel):
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    max_tokens: int = 4096
    system_instruction: str | None = None


class DocumentInfo(BaseModel):
    id: str
    filename: str
    size_bytes: int
    indexed_at: str
    chunks: int


class IngestRequest(BaseModel):
    text: str | None = None
    file_path: str | None = None
    metadata: dict[str, Any] | None = None


class CanvasDocumentRequest(BaseModel):
    title: str
    content: str
    card_timestamp: str | None = None


class CanvasDocumentResponse(BaseModel):
    conversation_id: str
    title: str
    content: str
    card_timestamp: str | None = None
    created_at: str
    updated_at: str
