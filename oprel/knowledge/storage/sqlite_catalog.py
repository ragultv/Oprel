"""
Transactional SQLite catalog for documents, versions, chunks, and ingestion jobs.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from oprel.knowledge.models import (
    Chunk,
    Document,
    DocumentMetadata,
    DocumentVersion,
    IngestionJob,
    JobStatus,
)

logger = logging.getLogger("oprel.knowledge.storage.sqlite_catalog")


class SqliteCatalog:
    """
    Local SQLite system of record for document metadata, version lineage, chunks, and job states.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_schema(self) -> None:
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    source_path TEXT,
                    active_version INTEGER DEFAULT 1,
                    status TEXT DEFAULT 'pending',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS document_versions (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    version_num INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    parser_name TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    is_active INTEGER DEFAULT 1,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    version_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    start_char INTEGER DEFAULT 0,
                    end_char INTEGER DEFAULT 0,
                    token_count INTEGER,
                    metadata_json TEXT,
                    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS ingestion_jobs (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempts INTEGER DEFAULT 0,
                    max_attempts INTEGER DEFAULT 3,
                    last_error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_version ON chunks(version_id);
            """)

    def save_document(self, doc: Document) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO documents (
                    id, content_hash, filename, file_type, size_bytes, source_path,
                    active_version, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    content_hash = excluded.content_hash,
                    filename = excluded.filename,
                    size_bytes = excluded.size_bytes,
                    active_version = excluded.active_version,
                    status = excluded.status,
                    updated_at = excluded.updated_at
            """, (
                doc.id, doc.content_hash, doc.metadata.filename, doc.metadata.file_type,
                doc.metadata.size_bytes, doc.metadata.source_path, doc.active_version,
                doc.status.value, doc.created_at, doc.updated_at
            ))

    def get_document(self, document_id: str) -> Optional[Document]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
            if not row:
                return None
            return Document(
                id=row["id"],
                content_hash=row["content_hash"],
                metadata=DocumentMetadata(
                    filename=row["filename"],
                    file_type=row["file_type"],
                    size_bytes=row["size_bytes"],
                    source_path=row["source_path"]
                ),
                active_version=row["active_version"],
                status=JobStatus(row["status"]),
                created_at=row["created_at"],
                updated_at=row["updated_at"]
            )

    def list_documents(self) -> List[Document]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM documents ORDER BY updated_at DESC").fetchall()
            return [
                Document(
                    id=row["id"],
                    content_hash=row["content_hash"],
                    metadata=DocumentMetadata(
                        filename=row["filename"],
                        file_type=row["file_type"],
                        size_bytes=row["size_bytes"],
                        source_path=row["source_path"]
                    ),
                    active_version=row["active_version"],
                    status=JobStatus(row["status"]),
                    created_at=row["created_at"],
                    updated_at=row["updated_at"]
                )
                for row in rows
            ]

    def delete_document(self, document_id: str) -> bool:
        with self._get_connection() as conn:
            cur = conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
            return cur.rowcount > 0

    def save_version(self, version: DocumentVersion) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO document_versions (
                    id, document_id, version_num, content_hash, parser_name,
                    chunk_count, is_active, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    is_active = excluded.is_active,
                    chunk_count = excluded.chunk_count
            """, (
                version.id, version.document_id, version.version_num, version.content_hash,
                version.parser_name, version.chunk_count, 1 if version.is_active else 0,
                version.created_at
            ))

    def save_chunks(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return
        with self._get_connection() as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO chunks (
                    id, document_id, version_id, chunk_index, text,
                    start_char, end_char, token_count, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (
                    c.id, c.document_id, c.version_id, c.chunk_index, c.text,
                    c.start_char, c.end_char, c.token_count, json.dumps(c.metadata)
                )
                for c in chunks
            ])

    def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[Chunk]:
        if not chunk_ids:
            return []
        placeholders = ",".join("?" for _ in chunk_ids)
        with self._get_connection() as conn:
            rows = conn.execute(
                f"SELECT * FROM chunks WHERE id IN ({placeholders})", chunk_ids
            ).fetchall()
            return [
                Chunk(
                    id=row["id"],
                    document_id=row["document_id"],
                    version_id=row["version_id"],
                    chunk_index=row["chunk_index"],
                    text=row["text"],
                    start_char=row["start_char"],
                    end_char=row["end_char"],
                    token_count=row["token_count"],
                    metadata=json.loads(row["metadata_json"] or "{}")
                )
                for row in rows
            ]
