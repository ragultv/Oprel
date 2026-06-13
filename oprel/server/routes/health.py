from __future__ import annotations

from fastapi import APIRouter, Response

from oprel.server.domain.state import get_state
from oprel.server.schemas.common import HealthResponse
from oprel.server.services.webui import get_webui_dir
from oprel.server import db

router = APIRouter()


@router.get("/", response_model=None)
async def root():
    webui_dir = get_webui_dir()
    if webui_dir:
        return Response(status_code=307, headers={"Location": "/gui/"})
    return {"status": "ok", "version": "0.3.3"}


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    state = get_state()
    return HealthResponse(
        status="ok",
        models_loaded=len(state.models),
        active_conversations=db.get_active_conversation_count(),
    )
