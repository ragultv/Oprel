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


def compact_conversation(conversation_id: str) -> dict:
    from oprel.knowledge.context_compactor import estimate_messages_tokens
    messages = db.get_conversation_messages(conversation_id)
    if not messages or len(messages) <= 4:
        return {
            "compacted": False,
            "message": "Conversation too short to compact (minimum 5 turns required)",
            "message_count": len(messages),
            "estimated_tokens": estimate_messages_tokens(messages),
            "messages": messages,
        }

    # Retain the most recent 6 messages verbatim for conversational immediacy
    recent_window = 6
    evicted_turns = messages[:-recent_window]
    active_turns = messages[-recent_window:]

    # Archive evicted turns into conversation memories (SuperMemory)
    for i in range(0, len(evicted_turns)):
        turn = evicted_turns[i]
        if turn.get("role") == "user":
            next_turn = evicted_turns[i + 1] if i + 1 < len(evicted_turns) else {"content": ""}
            db.add_conversation_memory(
                conversation_id=conversation_id,
                user_text=str(turn.get("content", "")),
                assistant_text=str(next_turn.get("content", "")),
                turn_index=i // 2,
            )

    # Build structured compaction memory notice
    evicted_digest = []
    for turn in evicted_turns[-6:]:
        role = str(turn.get("role", "user")).capitalize()
        content = str(turn.get("content", "")).strip()
        short = content[:140] + ("..." if len(content) > 140 else "")
        evicted_digest.append(f"{role}: {short}")

    digest_text = "\n".join(evicted_digest)
    compacted_intro = {
        "role": "system",
        "content": (
            f"[CONVERSATION COMPACTED (CONTEXT > 32K TOKENS)]\n"
            f"{len(evicted_turns)} earlier turns were compacted and archived into memory to maintain fast inference within 32k tokens.\n"
            f"Preceding context digest:\n{digest_text}\n"
            f"[END COMPACTED CONTEXT]"
        ),
    }

    new_messages = [compacted_intro] + active_turns
    db.replace_conversation_messages(conversation_id, new_messages)

    return {
        "compacted": True,
        "evicted_turns": len(evicted_turns),
        "remaining_turns": len(new_messages),
        "estimated_tokens": estimate_messages_tokens(new_messages),
        "messages": new_messages,
    }


def get_conversation_context(conversation_id: str) -> dict:
    from oprel.knowledge.context_compactor import estimate_messages_tokens
    from oprel.server.domain.state import get_state
    messages = db.get_conversation_messages(conversation_id)
    tokens = estimate_messages_tokens(messages)
    memories = db.get_conversation_memories(conversation_id)
    is_compacted = any("[CONVERSATION COMPACTED" in str(m.get("content", "")) for m in messages) or len(memories) > 0

    state = get_state()
    active_ctx = 32768
    if state.models:
        first_model = next(iter(state.models.values()))
        active_ctx = getattr(first_model, "ctx_size", 32768)

    reserve = min(2048, max(256, int(active_ctx * 0.25)))
    return {
        "conversation_id": conversation_id,
        "message_count": len(messages),
        "history_tokens": tokens,
        "max_context_tokens": active_ctx,
        "reserve_tokens": reserve,
        "headroom_tokens": max(0, active_ctx - tokens - reserve),
        "is_compacted": is_compacted,
        "memory_count": len(memories),
    }

