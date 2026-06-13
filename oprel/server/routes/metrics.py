from __future__ import annotations

from fastapi import APIRouter

from oprel.server.schemas.common import MetricsResponse
from oprel.server.services.metrics import get_metrics

router = APIRouter()


@router.get("/system/metrics", response_model=MetricsResponse)
async def metrics() -> MetricsResponse:
    return MetricsResponse(**get_metrics())
