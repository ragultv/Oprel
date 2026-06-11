from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfoData:
    model_id: str
    name: str
    size_gb: float = 0.0
    quantization: str | None = None
    backend: str = "llama.cpp"
    loaded: bool = False
    status: str = "cached"
    category: str | None = None
