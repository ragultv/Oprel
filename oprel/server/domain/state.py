from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AppState:
    models: dict[str, Any] = field(default_factory=dict)
    model_configs: dict[str, dict[str, Any]] = field(default_factory=dict)
    model_last_used: dict[str, float] = field(default_factory=dict)
    last_gen_speed: float = 0.0
    cleanup_task: asyncio.Task | None = None
    ephemeral_history: dict[str, list[dict[str, str]]] = field(default_factory=dict)


_state = AppState()


def get_state() -> AppState:
    return _state
