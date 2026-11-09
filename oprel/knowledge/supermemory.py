"""
Supermemory: Long-term Semantic Memory Engine for Local LLM Conversations.

Allows local models to scale conversation histories indefinitely without
resending thousands of redundant tokens on every request. Older conversation
turns are automatically indexed and semantically retrieved, while only the
immediate working window is passed as active dialogue.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
import math
import re
from typing import Any, Callable, Dict, List, Optional

from oprel.knowledge.config import CHROMA_PERSIST_DIR
from oprel.server import db

logger = logging.getLogger("oprel.knowledge.supermemory")

# Optional ChromaDB for vector-accelerated search
try:
    import chromadb
except ImportError:
    chromadb = None


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
    "to", "was", "were", "will", "with", "my", "your", "this", "do", "you",
}


def _tokenize(text: str) -> set[str]:
    """Extract lowercase alphanumeric tokens for keyword matching, excluding common stopwords."""
    words = set(re.findall(r"\w+", text.lower()))
    content_words = words - STOPWORDS
    return content_words if content_words else words


def _lexical_similarity(query_tokens: set[str], doc_tokens: set[str]) -> float:
    """Compute Jaccard token overlap between query and memory turn."""
    if not query_tokens or not doc_tokens:
        return 0.0
    intersection = query_tokens.intersection(doc_tokens)
    union = query_tokens.union(doc_tokens)
    return len(intersection) / len(union) if union else 0.0


def expand_query_with_context(query: str, context_hint: Optional[str] = None) -> str:
    """
    Perform query expansion / anaphora resolution for short follow-up prompts.
    If the query has few content tokens (e.g. 'can you explain 20 now'),
    enrich it with key subject tokens from the immediate preceding turn.
    """
    if not context_hint or not query:
        return query

    content_tokens = _tokenize(query)
    if len(content_tokens) <= 4:
        hint_tokens = _tokenize(context_hint)
        if hint_tokens:
            selected_hints = list(hint_tokens)[:5]
            return f"{query} {' '.join(selected_hints)}"
    return query


class SuperMemoryStore:
    """
    Production-grade Long-Term Conversational Memory (Supermemory).

    Stores conversation turns in SQLite for durability and utilizes
    vector similarity (ChromaDB) with lexical token-overlap fallback
    to recall relevant historical context.
    """

    COLLECTION_NAME = "oprel_supermemory"

    def __init__(self, embed_func: Optional[Callable] = None):
        self.embed_func = embed_func
        self._init_vector_store()

    def _init_vector_store(self) -> None:
        """Initialize ChromaDB collection if chromadb is available."""
        self.client = None
        self.collection = None
        if chromadb is None:
            logger.debug("ChromaDB not installed; using hybrid lexical fallback for Supermemory.")
            return

        try:
            self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
            self.collection = self.client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"SuperMemory initialized with ChromaDB at {CHROMA_PERSIST_DIR}")
        except Exception as exc:
            logger.warning(f"Failed to initialize ChromaDB for Supermemory: {exc}")
            self.client = None
            self.collection = None

    async def record_turn(
        self,
        conversation_id: str,
        user_text: str,
        assistant_text: str,
        turn_index: int = 0,
    ) -> str:
        """
        Record a completed conversation turn into persistent Supermemory.
        """
        if not user_text and not assistant_text:
            return ""

        # 1. Store in SQLite for ACID persistence
        mem_id = db.add_conversation_memory(
            conversation_id=conversation_id,
            user_text=user_text,
            assistant_text=assistant_text,
            turn_index=turn_index,
        )

        # 2. Add to ChromaDB vector index if available
        if self.collection and self.embed_func:
            combined_text = f"User: {user_text}\nAssistant: {assistant_text}"
            try:
                if inspect.iscoroutinefunction(self.embed_func):
                    embedding = await self.embed_func(combined_text)
                else:
                    embedding = self.embed_func(combined_text)

                if embedding:
                    self.collection.add(
                        ids=[mem_id],
                        embeddings=[embedding],
                        documents=[combined_text],
                        metadatas=[{
                            "conversation_id": conversation_id,
                            "turn_index": turn_index,
                            "user_text": user_text[:1000],
                            "assistant_text": assistant_text[:1000],
                        }],
                    )
            except Exception as exc:
                logger.warning(f"Vector embedding for turn failed, preserved in SQLite: {exc}")

        return mem_id

    async def search_memories(
        self,
        conversation_id: str,
        query: str,
        top_k: int = 3,
        threshold: float = 0.1,
        context_hint: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant historical memories from the given conversation.
        Supports anaphora resolution via optional context_hint.
        """
        if not conversation_id or not query:
            return []

        # Fetch SQLite memories for this conversation
        all_memories = db.get_conversation_memories(conversation_id, limit=200)
        if not all_memories:
            return []

        effective_query = expand_query_with_context(query, context_hint)
        query_tokens = _tokenize(effective_query)

        # If ChromaDB collection and embeddings are active, run vector search
        vector_matches: Dict[str, float] = {}
        if self.collection and self.embed_func:
            try:
                if inspect.iscoroutinefunction(self.embed_func):
                    query_vec = await self.embed_func(effective_query)
                else:
                    query_vec = self.embed_func(effective_query)

                if query_vec:
                    results = self.collection.query(
                        query_embeddings=[query_vec],
                        n_results=min(top_k * 2, len(all_memories)),
                        where={"conversation_id": conversation_id},
                    )
                    ids = results.get("ids", [[]])[0]
                    distances = results.get("distances", [[]])[0]
                    for mem_id, dist in zip(ids, distances):
                        # Cosine distance to similarity: 1.0 - dist
                        similarity = max(0.0, 1.0 - dist)
                        vector_matches[mem_id] = similarity
            except Exception as exc:
                logger.debug(f"ChromaDB query fallback to lexical: {exc}")

        # Combine vector and lexical scores
        scored_memories = []
        for mem in all_memories:
            mem_id = mem["id"]
            turn_text = f"{mem['user_text']} {mem['assistant_text']}"
            turn_tokens = _tokenize(turn_text)

            lex_score = _lexical_similarity(query_tokens, turn_tokens)
            vec_score = vector_matches.get(mem_id, 0.0)

            # Combined hybrid score (70% vector if available, 30% lexical)
            if mem_id in vector_matches:
                hybrid_score = (0.7 * vec_score) + (0.3 * lex_score)
            else:
                hybrid_score = lex_score

            if hybrid_score >= threshold:
                scored_memories.append({
                    "id": mem_id,
                    "user_text": mem["user_text"],
                    "assistant_text": mem["assistant_text"],
                    "turn_index": mem.get("turn_index", 0),
                    "score": round(hybrid_score, 4),
                })

        # Rank by score descending
        scored_memories.sort(key=lambda m: m["score"], reverse=True)
        return scored_memories[:top_k]

    def format_memory_context(self, memories: List[Dict[str, Any]]) -> str:
        """
        Format retrieved memories into a clean prompt context block.
        """
        if not memories:
            return ""

        formatted_turns = []
        for i, mem in enumerate(memories, start=1):
            user_part = mem['user_text'].strip()
            # Truncate assistant part if overly long to protect context budget
            asst_part = mem['assistant_text'].strip()
            if len(asst_part) > 500:
                asst_part = asst_part[:500] + "..."

            formatted_turns.append(
                f"[Past Turn {i}]\nUser: {user_part}\nAssistant: {asst_part}"
            )

        return (
            "--- RELEVANT CONVERSATION MEMORY (SUPERMEMORY) ---\n"
            "The following context was retrieved from earlier messages in this conversation:\n"
            f"{chr(10).join(formatted_turns)}\n"
            "--------------------------------------------------"
        )

    def clear_memories(self, conversation_id: str) -> None:
        """Erase all memories for a conversation."""
        db.delete_conversation_memories(conversation_id)
        if self.collection:
            try:
                self.collection.delete(where={"conversation_id": conversation_id})
            except Exception as exc:
                logger.debug(f"Failed to clear ChromaDB memories for {conversation_id}: {exc}")
