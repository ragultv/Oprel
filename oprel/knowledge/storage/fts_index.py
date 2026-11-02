"""
SQLite FTS5 lexical full-text index implementation conforming to LexicalIndex interface.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from pathlib import Path
from typing import List, Optional, Tuple

from oprel.knowledge.interfaces import LexicalIndex
from oprel.knowledge.models import Chunk

logger = logging.getLogger("oprel.knowledge.storage.fts_index")


class FtsIndex(LexicalIndex):
    """
    High-performance, zero-RAM persistent lexical search index powered by SQLite FTS5.
    Provides native BM25 scoring and replaces legacy JSON-based BM25 caching.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_fts()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_fts(self) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    text,
                    tokenize='porter unicode61'
                );
            """)

    def _sanitize_query(self, query: str) -> str:
        """Sanitize query for FTS5 syntax, escaping special characters."""
        words = re.findall(r"\w+", query)
        if not words:
            return ""
        # Combine words into an OR or NEAR expression for robust recall
        return " OR ".join(f'"{w}"' for w in words)

    async def upsert(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return
        with self._get_connection() as conn:
            # Delete existing instances of these chunk_ids
            chunk_ids = [c.id for c in chunks]
            placeholders = ",".join("?" for _ in chunk_ids)
            conn.execute(f"DELETE FROM chunks_fts WHERE chunk_id IN ({placeholders})", chunk_ids)

            conn.executemany(
                "INSERT INTO chunks_fts(chunk_id, text) VALUES (?, ?)",
                [(c.id, c.text) for c in chunks]
            )

    async def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        clean_q = self._sanitize_query(query)
        if not clean_q:
            return []

        with self._get_connection() as conn:
            try:
                # In SQLite FTS5, lower bm25() score represents higher relevance (more negative).
                # We invert it: relevance = -bm25_score so higher is better.
                cursor = conn.execute("""
                    SELECT chunk_id, -bm25(chunks_fts) AS score
                    FROM chunks_fts
                    WHERE chunks_fts MATCH ?
                    ORDER BY score DESC
                    LIMIT ?
                """, (clean_q, top_k))
                rows = cursor.fetchall()
                return [(r["chunk_id"], float(r["score"])) for r in rows]
            except sqlite3.OperationalError as e:
                logger.warning(f"FTS5 search error for query '{query}': {e}")
                return []

    async def delete(self, chunk_ids: List[str]) -> None:
        if not chunk_ids:
            return
        placeholders = ",".join("?" for _ in chunk_ids)
        with self._get_connection() as conn:
            conn.execute(f"DELETE FROM chunks_fts WHERE chunk_id IN ({placeholders})", chunk_ids)
