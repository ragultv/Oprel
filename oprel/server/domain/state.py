from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AppState:
    models: dict[str, Any] = field(default_factory=dict)
    model_configs: dict[str, dict[str, Any]] = field(default_factory=dict)
    model_last_used: dict[str, float] = field(default_factory=dict)
    # Speeds of the last user-facing generation (title requests excluded).
    # last_gen_speed: decode tok/s, measured from the first to the last output token.
    last_gen_speed: float = 0.0
    last_prompt_speed: float = 0.0
    last_prompt_tokens: int = 0
    last_completion_tokens: int = 0
    last_ttft_ms: float = 0.0
    cleanup_task: asyncio.Task | None = None
    ephemeral_history: dict[str, list[dict[str, str]]] = field(default_factory=dict)


_state = AppState()


def get_state() -> AppState:
    return _state
