"""Unit tests for the 32k context compactor and memory manager."""

from oprel.knowledge.context_compactor import (
    estimate_tokens,
    estimate_messages_tokens,
    compact_conversation_context,
)


def test_estimate_tokens():
    text = "Hello world! This is a test sentence."
    tokens = estimate_tokens(text)
    assert tokens > 0
    assert estimate_tokens("") == 0
    assert estimate_tokens(None) == 0


def test_estimate_messages_tokens():
    messages = [
        {"role": "user", "content": "Tell me a story about space exploration."},
        {"role": "assistant", "content": "Once upon a time, humanity launched a deep-space probe."},
    ]
    tokens = estimate_messages_tokens(messages)
    assert tokens > 10


def test_compact_within_budget_keeps_intact():
    history = [
        {"role": "user", "content": "Step 1: Download files"},
        {"role": "assistant", "content": "Step 1 complete"},
        {"role": "user", "content": "Step 2: Parse headers"},
        {"role": "assistant", "content": "Step 2 complete"},
    ]
    active, sys_out, was_compacted = compact_conversation_context(
        history=history,
        current_prompt="What is next?",
        system_prompt="You are a helpful assistant.",
        max_context_tokens=32768,
    )
    assert was_compacted is False
    assert len(active) == 4
    assert sys_out == "You are a helpful assistant."


def test_compact_beyond_budget_triggers_memory_recall():
    long_turn = "A" * 1500  # ~400 tokens per message
    history = []
    for i in range(20):
        history.append({"role": "user", "content": f"User turn {i}: {long_turn}"})
        history.append({"role": "assistant", "content": f"Assistant turn {i}: {long_turn}"})

    recalled_facts = [
        "Historical key fact: user UID is 123456789",
        "User lives in Sangrur, Punjab",
    ]

    active, sys_out, was_compacted = compact_conversation_context(
        history=history,
        current_prompt="What was my UID and city?",
        system_prompt="Assistant prompt",
        max_context_tokens=4096,  # test boundary
        reserve_generation_tokens=512,
        min_recent_turns=4,
        recalled_memories=recalled_facts,
    )

    assert was_compacted is True
    assert len(active) < len(history)
    assert "CONVERSATION COMPACTED" in sys_out
    assert "Historical key fact: user UID is 123456789" in sys_out
    assert "User lives in Sangrur, Punjab" in sys_out


def test_conversation_service_compaction(tmp_path, monkeypatch):
    """Verify service-level compaction updates database messages and archives memories."""
    monkeypatch.setenv("OPREL_CACHE_DIR", str(tmp_path))
    from oprel.server import db
    db_file = tmp_path / "test_chat.db"
    monkeypatch.setattr(db, "get_db_path", lambda: db_file)
    monkeypatch.setattr(db, "DB_PATH", db_file)
    db.init_db()

    from oprel.server.services.conversations import compact_conversation, get_conversation_context

    conv_id = db.create_conversation("test-model", "Test Chat")
    for i in range(8):
        db.add_message(conv_id, "user", f"User question {i}")
        db.add_message(conv_id, "assistant", f"Assistant answer {i}")

    ctx_before = get_conversation_context(conv_id)
    assert ctx_before["message_count"] == 16
    assert ctx_before["max_context_tokens"] == 32768
    assert ctx_before["history_tokens"] > 0
    assert ctx_before["is_compacted"] is False

    res = compact_conversation(conv_id)
    assert res["compacted"] is True
    assert res["evicted_turns"] > 0
    assert res["remaining_turns"] <= 8

    ctx_after = get_conversation_context(conv_id)
    assert ctx_after["is_compacted"] is True
    assert ctx_after["memory_count"] > 0
    assert ctx_after["message_count"] == res["remaining_turns"]
