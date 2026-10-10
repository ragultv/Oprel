"""
Legacy data migration adapter to safely backfill legacy BM25 JSON documents into SQLite FTS5.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

from oprel.knowledge.models import Chunk, Document, DocumentMetadata, DocumentVersion, JobStatus
from oprel.knowledge.storage.repository import UnifiedKnowledgeRepository

logger = logging.getLogger("oprel.knowledge.storage.migration")


class LegacyDataMigrator:
    """
    Safely migrates legacy bm25_docs.json entries into the SQLite catalog and FTS5 index.
    Additive and non-destructive: keeps existing ChromaDB collections readable.
    """

    def __init__(self, knowledge_dir: Path, repository: UnifiedKnowledgeRepository):
        self.knowledge_dir = knowledge_dir
        self.repository = repository

    async def migrate_legacy_bm25(self) -> int:
        bm25_file = self.knowledge_dir / "bm25_docs.json"
        if not bm25_file.exists():
            logger.info("No legacy bm25_docs.json found to migrate.")
            return 0

        try:
            with open(bm25_file, "r", encoding="utf-8") as f:
                docs_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read legacy bm25_docs.json: {e}")
            return 0

        migrated_count = 0
        grouped_by_doc: Dict[str, List[Chunk]] = {}

        for item in docs_data:
            chunk_id = item.get("id")
            text = item.get("text", "")
            meta = item.get("metadata", {}) or {}
            filename = meta.get("filename", "legacy_document.txt")
            doc_id = meta.get("doc_id", filename.replace(" ", "_").lower())
            version_id = f"{doc_id}_v1"

            chunk = Chunk(
                id=chunk_id,
                document_id=doc_id,
                version_id=version_id,
                chunk_index=meta.get("chunk_index", 0),
                text=text,
                metadata=meta,
            )

            if doc_id not in grouped_by_doc:
                grouped_by_doc[doc_id] = []
            grouped_by_doc[doc_id].append(chunk)

        for doc_id, chunks in grouped_by_doc.items():
            first_chunk = chunks[0]
            filename = first_chunk.metadata.get("filename", "legacy_doc.txt")
            combined_text = "\n".join(c.text for c in chunks)
            import hashlib
            content_hash = hashlib.sha256(combined_text.encode("utf-8")).hexdigest()

            doc = Document(
                id=doc_id,
                content_hash=content_hash,
                metadata=DocumentMetadata(
                    filename=filename,
                    file_type="txt",
                    size_bytes=len(combined_text.encode("utf-8")),
                ),
                active_version=1,
                status=JobStatus.COMPLETED,
            )
            version = DocumentVersion(
                id=f"{doc_id}_v1",
                document_id=doc_id,
                version_num=1,
                content_hash=content_hash,
                parser_name="LegacyMigration",
                chunk_count=len(chunks),
                is_active=True,
            )

            await self.repository.save_document(doc)
            await self.repository.save_version(version)
            await self.repository.save_chunks(chunks)
            migrated_count += len(chunks)

        logger.info(f"Successfully migrated {migrated_count} legacy chunks into SQLite catalog and FTS5.")
        return migrated_count
