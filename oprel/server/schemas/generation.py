from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class GenerateRequest(BaseModel):
    model_id: str
    prompt: Any
    max_tokens: int = 8192
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    stream: bool = False
    images: list[str] | None = None
    conversation_id: str | None = None
    system_prompt: str | None = None
    reset_conversation: bool = False
    thinking: bool = False
    rag: bool = False


class GenerateResponse(BaseModel):
    text: str
    model_id: str
    conversation_id: str
    message_count: int


class EmbedRequest(BaseModel):
    model: str
    input: str | list[str]
