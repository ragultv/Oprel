"""
Unit tests for Supermemory: Semantic Long-Term Conversational Memory.
"""

import pytest

from oprel.core.config import Config
from oprel.knowledge.supermemory import SuperMemoryStore, _lexical_similarity, _tokenize
from oprel.server import db


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path, monkeypatch):
    """Point DB to isolated temporary database file for test isolation."""
    test_db_path = tmp_path / "test_supermemory.db"
    monkeypatch.setattr(db, "DB_PATH", test_db_path)
    db.init_db()
    yield


def test_tokenization_and_lexical_similarity():
    """Verify tokenizer and Jaccard token overlap metrics."""
    tokens1 = _tokenize("My favorite database is PostgreSQL and SQLite")
    assert "postgresql" in tokens1
    assert "sqlite" in tokens1

    tokens2 = _tokenize("Which database do you prefer: PostgreSQL?")
    sim = _lexical_similarity(tokens1, tokens2)
    assert sim > 0.0

    tokens3 = _tokenize("Quantum physics and string theory")
    assert _lexical_similarity(tokens1, tokens3) == 0.0


@pytest.mark.asyncio
async def test_record_and_search_memories():
    """Verify storing conversational turns and recalling them via search."""
    store = SuperMemoryStore(embed_func=None)  # tests robust lexical fallback
    conv_id = "test_chat_conv_123"

    # Seed conversation memories
    await store.record_turn(
        conversation_id=conv_id,
        user_text="I live in Tokyo and work on quantum computing algorithms.",
        assistant_text="That's great! Quantum computing in Tokyo has many active research labs.",
        turn_index=1,
    )
    await store.record_turn(
        conversation_id=conv_id,
        user_text="What is a good recipe for homemade sourdough bread?",
        assistant_text="To make sourdough bread, you need active starter, flour, water, and salt.",
        turn_index=2,
    )

    # Search for quantum computing
    results = await store.search_memories(
        conversation_id=conv_id,
        query="Where do I work and what is my field?",
        top_k=1,
    )

    assert len(results) == 1
    assert "quantum computing" in results[0]["user_text"].lower()

    # Format context
    context = store.format_memory_context(results)
    assert "RELEVANT CONVERSATION MEMORY" in context
    assert "Tokyo" in context


@pytest.mark.asyncio
async def test_clear_memories():
    """Verify memory clearance upon conversation reset or deletion."""
    store = SuperMemoryStore(embed_func=None)
    conv_id = "test_clear_conv"

    await store.record_turn(
        conversation_id=conv_id,
        user_text="Remember my secret code is 4242.",
        assistant_text="I will remember your code 4242.",
        turn_index=1,
    )

    memories_before = db.get_conversation_memories(conv_id)
    assert len(memories_before) == 1

    store.clear_memories(conv_id)

    memories_after = db.get_conversation_memories(conv_id)
    assert len(memories_after) == 0


def test_supermemory_config_defaults():
    """Verify Supermemory configuration options and defaults."""
    cfg = Config()
    assert cfg.supermemory_enabled is True
    assert cfg.supermemory_recency_window == 10
    assert cfg.supermemory_top_k == 3


@pytest.mark.asyncio
async def test_anaphora_resolution_query_expansion():
    """Verify that ambiguous short queries are enriched with context hints to retrieve relevant memories."""
    from oprel.knowledge.supermemory import expand_query_with_context

    expanded = expand_query_with_context(
        query="can you explain 20 now?",
        context_hint="how many emojis do you know?",
    )
    assert "emoji" in expanded.lower() or "emojis" in expanded.lower()

    store = SuperMemoryStore(embed_func=None)
    conv_id = "test_anaphora_conv"

    await store.record_turn(
        conversation_id=conv_id,
        user_text="How many emojis do you understand?",
        assistant_text="I understand thousands of emojis across unicode catalogs.",
        turn_index=1,
    )
    await store.record_turn(
        conversation_id=conv_id,
        user_text="What is the weather like in Seattle?",
        assistant_text="Seattle often has mild, overcast weather with frequent drizzle.",
        turn_index=2,
    )

    # Short follow-up without the word 'emoji'
    results = await store.search_memories(
        conversation_id=conv_id,
        query="can you explain 20 now?",
        top_k=1,
        context_hint="how many emojis do you know?",
    )

    assert len(results) == 1
    assert "emojis" in results[0]["user_text"].lower()
