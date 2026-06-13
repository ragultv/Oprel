from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ProviderChatRequest(BaseModel):
    model: str
    messages: list[dict[str, Any]]
    max_tokens: int | None = 4096
    temperature: float | None = 0.7
    top_p: float | None = 1.0
    stream: bool = True
    conversation_id: str | None = None
    rag: bool = False


class ProviderUpsertRequest(BaseModel):
    id: str
    name: str
    type: str
    api_key: str = ""
    base_url: str = ""
    enabled: bool = True
    enabled_model_ids: list[str] = []
    available_model_ids: list[str] = []
    last_fetched: str | None = None
