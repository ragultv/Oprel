from __future__ import annotations

from fastapi import APIRouter

from oprel.server.schemas.common import RenameConversationRequest
from oprel.server.services import conversations as conversation_service

router = APIRouter()


@router.get("/conversations")
async def list_conversations():
    return conversation_service.list_conversations()


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
