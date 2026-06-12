from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class OpenAIChatMessage(BaseModel):
    role: str
    content: Any


class OpenAIChatRequest(BaseModel):
    model: str
    messages: list[OpenAIChatMessage]
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    max_tokens: int | None = 8192
    stream: bool = False
    conversation_id: str | None = None
    thinking: bool = False
    rag: bool = False
    skill: dict | None = None


class OpenAICompletionRequest(BaseModel):
    model: str
    prompt: str
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    max_tokens: int | None = 8192
    stream: bool = False
