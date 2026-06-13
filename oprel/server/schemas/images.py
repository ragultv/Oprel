from __future__ import annotations

from pydantic import BaseModel


class ImageGenerationRequest(BaseModel):
    prompt: str
    model: str | None = None
    n: int = 1
    quality: str | None = "standard"
    response_format: str | None = "url"
    size: str | None = "1024x1024"
    style: str | None = "vivid"
    user: str | None = None
    negative_prompt: str | None = None
    steps: int | None = None
    cfg_scale: float | None = None
    seed: int | None = None
    sampler: str | None = None


class ImageGenerationResponse(BaseModel):
    created: int
    data: list[dict[str, str]]
