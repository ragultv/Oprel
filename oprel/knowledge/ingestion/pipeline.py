"""
Production-grade document ingestion pipeline for Oprel Knowledge & RAG.
Orchestrates parsing, deduplication, semantic chunking, and durable job tracking.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from oprel.knowledge.ingestion.chunker import SemanticChunker
from oprel.knowledge.ingestion.jobs import JobManager
from oprel.knowledge.ingestion.parser_registry import ParserRegistry
from oprel.knowledge.models import (
    Chunk,
    Document,
    DocumentMetadata,
    DocumentVersion,
    IngestionJob,
    JobStatus,
)

logger = logging.getLogger("oprel.knowledge.ingestion.pipeline")


class IngestionPipeline:
    """
    End-to-end ingestion pipeline handling file discovery, hashing,
    versioned parsing, chunking, and durable job lifecycle.
    """

    def __init__(
        self,
        storage_dir: Path,
        parser_registry: Optional[ParserRegistry] = None,
        chunker: Optional[SemanticChunker] = None,
    ):
        self.storage_dir = storage_dir
        self.parsers = parser_registry or ParserRegistry()
        self.chunker = chunker or SemanticChunker()
        self.jobs = JobManager(storage_dir)

    def compute_content_hash(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    async def process_file(
        self,
        file_path: Path,
        existing_doc: Optional[Document] = None,
    ) -> Tuple[Document, DocumentVersion, List[Chunk], IngestionJob]:
        """
        Parse, hash, version, and chunk a local file.
        Returns (document, version, chunks, job).
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Source file does not exist: {file_path}")

        parser = self.parsers.get_parser(file_path)
        text, metadata = await parser.parse(file_path)

        content_hash = self.compute_content_hash(text)
        doc_id = existing_doc.id if existing_doc else hashlib.sha256(str(file_path).encode("utf-8")).hexdigest()[:16]

        job = self.jobs.create_job(document_id=doc_id)
        self.jobs.update_job(job.id, JobStatus.PROCESSING)

        try:
            # Check versioning
            version_num = (existing_doc.active_version + 1) if (existing_doc and existing_doc.content_hash != content_hash) else 1
            version_id = f"{doc_id}_v{version_num}"

            doc_meta = DocumentMetadata(
                filename=file_path.name,
                file_type=file_path.suffix.lstrip(".").lower() or "txt",
                size_bytes=file_path.stat().st_size,
                source_path=str(file_path.resolve()),
                extra=metadata,
            )

            document = Document(
                id=doc_id,
                content_hash=content_hash,
                metadata=doc_meta,
                active_version=version_num,
                status=JobStatus.COMPLETED,
            )

            # Chunk document
            chunks = self.chunker.chunk(
                text=text,
                document_id=doc_id,
                version_id=version_id,
                base_metadata={"filename": file_path.name, "source": str(file_path)},
            )

            version = DocumentVersion(
                id=version_id,
                document_id=doc_id,
                version_num=version_num,
                content_hash=content_hash,
                parser_name=parser.__class__.__name__,
                chunk_count=len(chunks),
                is_active=True,
            )

            self.jobs.update_job(job.id, JobStatus.COMPLETED)
            return document, version, chunks, job

        except Exception as exc:
            logger.error(f"Failed processing file {file_path}: {exc}", exc_info=True)
            self.jobs.update_job(job.id, JobStatus.FAILED, error=str(exc))
            raise

    async def process_raw_text(
        self,
        text: str,
        filename: str = "snippet.txt",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Document, DocumentVersion, List[Chunk], IngestionJob]:
        """
        Ingest raw text or clipboard payload into structured versioned chunks.
        """
        content_hash = self.compute_content_hash(text)
        doc_id = hashlib.sha256(content_hash.encode("utf-8")).hexdigest()[:16]
        version_id = f"{doc_id}_v1"

        job = self.jobs.create_job(document_id=doc_id)
        self.jobs.update_job(job.id, JobStatus.PROCESSING)

        try:
            doc_meta = DocumentMetadata(
                filename=filename,
                file_type="txt",
                size_bytes=len(text.encode("utf-8")),
                source_path=None,
                extra=extra_metadata or {},
            )

            document = Document(
                id=doc_id,
                content_hash=content_hash,
                metadata=doc_meta,
                active_version=1,
                status=JobStatus.COMPLETED,
            )

            chunks = self.chunker.chunk(
                text=text,
                document_id=doc_id,
                version_id=version_id,
                base_metadata={"filename": filename},
            )

            version = DocumentVersion(
                id=version_id,
                document_id=doc_id,
                version_num=1,
                content_hash=content_hash,
                parser_name="RawText",
                chunk_count=len(chunks),
                is_active=True,
            )

            self.jobs.update_job(job.id, JobStatus.COMPLETED)
            return document, version, chunks, job

        except Exception as exc:
            self.jobs.update_job(job.id, JobStatus.FAILED, error=str(exc))
            raise
