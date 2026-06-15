from __future__ import annotations

from oprel.server import db


def list_conversations() -> list[dict[str, str]]:
    return db.list_conversations()


def create_conversation(model_id: str, title: str = "New Chat") -> dict[str, str]:
    conversation_id = db.create_conversation(model_id=model_id, title=title)
    return {
        "id": conversation_id,
        "title": title,
        "model_id": model_id,
    }


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


def get_canvas(conversation_id: str) -> dict | None:
    return db.get_canvas_document(conversation_id)


def upsert_canvas(conversation_id: str, title: str, content: str, card_timestamp: str = None) -> dict:
    return db.upsert_canvas_document(conversation_id, title, content, card_timestamp)
