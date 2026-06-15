from __future__ import annotations

from fastapi import APIRouter

from oprel.server.schemas.common import RenameConversationRequest, CanvasDocumentRequest, CreateConversationRequest
from oprel.server.services import conversations as conversation_service

router = APIRouter()


@router.get("/conversations")
async def list_conversations():
    return conversation_service.list_conversations()


@router.post("/conversations")
async def create_conversation(request: CreateConversationRequest):
    return conversation_service.create_conversation(request.model_id, request.title)


@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    return conversation_service.get_conversation(conversation_id)


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    return conversation_service.delete_conversation(conversation_id)


@router.post("/conversations/{conversation_id}/reset")
async def reset_conversation(conversation_id: str):
    return conversation_service.reset_conversation(conversation_id)


@router.put("/conversations/{conversation_id}/title")
async def rename_conversation(conversation_id: str, request: RenameConversationRequest):
    return conversation_service.rename_conversation(conversation_id, request.title)


@router.get("/analytics/summary")
async def get_analytics_summary(days: int = 7):
    return conversation_service.analytics_summary(days)


@router.get("/conversations/{conversation_id}/canvas")
async def get_canvas(conversation_id: str):
    doc = conversation_service.get_canvas(conversation_id)
    return doc


@router.post("/conversations/{conversation_id}/canvas")
async def update_canvas(conversation_id: str, request: CanvasDocumentRequest):
    return conversation_service.upsert_canvas(conversation_id, request.title, request.content, request.card_timestamp)
