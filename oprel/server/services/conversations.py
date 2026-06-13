from __future__ import annotations

from oprel.server import db


def list_conversations() -> list[dict[str, str]]:
    return db.list_conversations()


def get_conversation(conversation_id: str) -> list[dict[str, str]]:
    messages = db.get_conversation_messages(conversation_id)
    if not messages:
        return []
    return messages


def delete_conversation(conversation_id: str) -> dict[str, bool]:
    db.delete_conversation(conversation_id)
    return {"success": True}


def reset_conversation(conversation_id: str) -> dict[str, bool]:
    db.reset_conversation(conversation_id)
    return {"success": True}


def rename_conversation(conversation_id: str, title: str) -> dict[str, bool]:
    db.rename_conversation(conversation_id, title)
    return {"success": True}


def analytics_summary(days: int = 7) -> dict:
    return db.get_inference_summary(days)
