"""
Core architecture contracts and abstract interfaces for Oprel Knowledge & RAG.
Independent of ChromaDB, SQLite, or specific embedding engines.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from oprel.knowledge.models import (
    Chunk,
    ContextBudget,
    Document,
    DocumentMetadata,
    DocumentVersion,
    Evidence,
    GroundedPrompt,
    IngestionJob,
    JobStatus,
    RetrievalQuery,
    RetrievalResult,
)


class DocumentParser(ABC):
    """Extract normalized text and structural metadata from a file."""

    @abstractmethod
    def can_parse(self, file_path: Path) -> bool:
        """Return True if this parser supports the given file path/extension."""
        pass

    @abstractmethod
    async def parse(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        """
        Parse file into text content and metadata dict.
        Returns (content, metadata).
        """
        pass


class Chunker(ABC):
    """Split text into deterministic, traceable chunks."""

    @abstractmethod
    def chunk(
        self,
        text: str,
        document_id: str,
        version_id: str,
        base_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """Convert raw text into traceable chunks with offsets and deterministic IDs."""
        pass


class EmbeddingProvider(ABC):
    """Generate dense vectors through local or configured embedding engines."""

    @abstractmethod
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate normalized embedding vectors for a list of text strings."""
        pass

    @abstractmethod
    async def embed_single(self, text: str) -> List[float]:
        """Generate embedding vector for a single string."""
        pass


class VectorIndex(ABC):
    """Store, retrieve, update, and delete vector records."""

    @abstractmethod
    async def upsert(self, chunks: List[Chunk]) -> None:
        """Upsert chunks with their embeddings into the vector store."""
        pass

    @abstractmethod
    async def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Search nearest neighbors.
        Returns list of (chunk_id, similarity_score).
        """
        pass

    @abstractmethod
    async def delete(self, chunk_ids: List[str]) -> None:
        """Delete chunks by IDs from vector store."""
        pass


class LexicalIndex(ABC):
    """Search exact terms and keyword matches using BM25 or SQLite FTS5."""

    @abstractmethod
    async def upsert(self, chunks: List[Chunk]) -> None:
        """Index chunks for full-text search."""
        pass

    @abstractmethod
    async def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Search keyword index.
        Returns list of (chunk_id, lexical_score).
        """
        pass

    @abstractmethod
    async def delete(self, chunk_ids: List[str]) -> None:
        """Delete chunks from lexical index."""
        pass


class KnowledgeRepository(ABC):
    """Transactional catalog managing documents, versions, chunks, and ingestion jobs."""

    @abstractmethod
    async def save_document(self, document: Document) -> None:
        pass

    @abstractmethod
    async def get_document(self, document_id: str) -> Optional[Document]:
        pass

    @abstractmethod
    async def get_document_by_hash(self, content_hash: str) -> Optional[Document]:
        pass

    @abstractmethod
    async def list_documents(self) -> List[Document]:
        pass

    @abstractmethod
    async def delete_document(self, document_id: str) -> bool:
        pass

    @abstractmethod
    async def save_version(self, version: DocumentVersion) -> None:
        pass

    @abstractmethod
    async def save_chunks(self, chunks: List[Chunk]) -> None:
        pass

    @abstractmethod
    async def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[Chunk]:
        pass

    @abstractmethod
    async def create_job(self, document_id: str) -> IngestionJob:
        pass

    @abstractmethod
    async def update_job_status(
        self, job_id: str, status: JobStatus, error: Optional[str] = None
    ) -> None:
        pass


class RetrievalService(ABC):
    """Run hybrid search, fuse rankings (RRF), and return ranked evidence."""

    @abstractmethod
    async def search(self, query: RetrievalQuery) -> RetrievalResult:
        """Execute hybrid search pipeline and return ranked Evidence items."""
        pass


class ContextBuilder(ABC):
    """Enforce token budgets and construct grounded prompts with citations."""

    @abstractmethod
    def build_grounded_prompt(
        self,
        query: str,
        evidence: List[Evidence],
        budget: ContextBudget,
        system_instruction: Optional[str] = None,
    ) -> GroundedPrompt:
        """Construct prompt with citations and strict token budget enforcement."""
        pass
