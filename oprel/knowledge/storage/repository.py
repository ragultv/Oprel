"""
Unified transactional repository implementing the KnowledgeRepository interface.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from oprel.knowledge.interfaces import KnowledgeRepository
from oprel.knowledge.models import (
    Chunk,
    Document,
    DocumentVersion,
    IngestionJob,
    JobStatus,
)
from oprel.knowledge.storage.chroma_index import ChromaIndex
from oprel.knowledge.storage.fts_index import FtsIndex
from oprel.knowledge.storage.sqlite_catalog import SqliteCatalog

logger = logging.getLogger("oprel.knowledge.storage.repository")


class UnifiedKnowledgeRepository(KnowledgeRepository):
    """
    Combines the SQLite catalog, SQLite FTS5 lexical index, and ChromaDB vector index
    into a coordinated, crash-resilient repository.
    """

    def __init__(
        self,
        base_dir: Path,
        collection_name: str = "oprel_knowledge",
    ):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

        catalog_db = self.base_dir / "catalog.db"
        self.catalog = SqliteCatalog(catalog_db)
        self.fts = FtsIndex(catalog_db)
        self.vector_index = ChromaIndex(self.base_dir / "chroma_db", collection_name=collection_name)

    async def save_document(self, document: Document) -> None:
        self.catalog.save_document(document)

    async def get_document(self, document_id: str) -> Optional[Document]:
        return self.catalog.get_document(document_id)

    async def get_document_by_hash(self, content_hash: str) -> Optional[Document]:
        for doc in self.catalog.list_documents():
            if doc.content_hash == content_hash:
                return doc
        return None

    async def list_documents(self) -> List[Document]:
        return self.catalog.list_documents()

    async def delete_document(self, document_id: str) -> bool:
        doc = self.catalog.get_document(document_id)
        if not doc:
            # Check by filename
            for d in self.catalog.list_documents():
                if d.metadata.filename == document_id:
                    doc = d
                    break
        if not doc:
            return False

        target_id = doc.id
        # Find all chunk IDs associated with this document
        with self.catalog._get_connection() as conn:
            rows = conn.execute("SELECT id FROM chunks WHERE document_id = ?", (target_id,)).fetchall()
            chunk_ids = [r["id"] for r in rows]

        # 1. Delete from FTS5
        await self.fts.delete(chunk_ids)

        # 2. Delete from Vector Index
        await self.vector_index.delete(chunk_ids)

        # 3. Delete from catalog (cascades versions and chunks)
        return self.catalog.delete_document(target_id)

    async def delete_all(self) -> None:
        """Clear all documents, versions, chunks, FTS entries, and vector collection records."""
        with self.catalog._get_connection() as conn:
            conn.execute("DELETE FROM chunks;")
            conn.execute("DELETE FROM document_versions;")
            conn.execute("DELETE FROM ingestion_jobs;")
            conn.execute("DELETE FROM documents;")
        with self.fts._get_connection() as conn:
            conn.execute("DELETE FROM chunks_fts;")
        if self.vector_index.collection:
            try:
                all_records = self.vector_index.collection.get()
                if all_records and all_records.get("ids"):
                    self.vector_index.collection.delete(ids=all_records["ids"])
            except Exception as e:
                logger.warning(f"Error clearing ChromaDB collection: {e}")

    async def save_version(self, version: DocumentVersion) -> None:
        self.catalog.save_version(version)

    async def save_chunks(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return
        # 1. Persist chunk records in relational catalog
        self.catalog.save_chunks(chunks)

        # 2. Index in FTS5 lexical engine
        await self.fts.upsert(chunks)

        # 3. Index in Vector engine if embeddings are present
        if any(c.embedding is not None for c in chunks):
            await self.vector_index.upsert(chunks)

    async def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[Chunk]:
        return self.catalog.get_chunks_by_ids(chunk_ids)

    async def create_job(self, document_id: str) -> IngestionJob:
        from datetime import datetime, timezone
        jid = f"job_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{document_id[:8]}"
        job = IngestionJob(id=jid, document_id=document_id, status=JobStatus.PENDING)
        with self.catalog._get_connection() as conn:
            conn.execute("""
                INSERT INTO ingestion_jobs (id, document_id, status, attempts, max_attempts, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (job.id, job.document_id, job.status.value, job.attempts, job.max_attempts, job.created_at, job.updated_at))
        return job

    async def update_job_status(
        self, job_id: str, status: JobStatus, error: Optional[str] = None
    ) -> None:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        with self.catalog._get_connection() as conn:
            if error:
                conn.execute("""
                    UPDATE ingestion_jobs
                    SET status = ?, attempts = attempts + 1, last_error = ?, updated_at = ?
                    WHERE id = ?
                """, (status.value, error, now, job_id))
            else:
                conn.execute("""
                    UPDATE ingestion_jobs
                    SET status = ?, updated_at = ?
                    WHERE id = ?
                """, (status.value, now, job_id))
