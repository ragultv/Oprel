"""
Core domain models and data structures for Oprel Knowledge & RAG.
Storage-agnostic, strictly typed, production-ready.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class DocumentMetadata:
    filename: str
    file_type: str
    size_bytes: int
    source_path: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Document:
    id: str
    content_hash: str
    metadata: DocumentMetadata
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    active_version: int = 1
    status: JobStatus = JobStatus.PENDING


@dataclass
class DocumentVersion:
    id: str
    document_id: str
    version_num: int
    content_hash: str
    parser_name: str
    chunk_count: int
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    is_active: bool = True


@dataclass
class Chunk:
    id: str
    document_id: str
    version_id: str
    chunk_index: int
    text: str
    start_char: int = 0
    end_char: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    token_count: Optional[int] = None


@dataclass
class IngestionJob:
    id: str
    document_id: str
    status: JobStatus = JobStatus.PENDING
    attempts: int = 0
    max_attempts: int = 3
    last_error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class Evidence:
    """Represents a single retrieved and ranked piece of evidence."""
    chunk_id: str
    document_id: str
    filename: str
    text: str
    score: float
    dense_score: Optional[float] = None
    lexical_score: Optional[float] = None
    rank: int = 0
    source_label: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_citation_label(self, index: int) -> str:
        self.source_label = f"[{index}]"
        return self.source_label


@dataclass
class RetrievalQuery:
    text: str
    top_k: int = 5
    score_threshold: float = 0.0
    dense_weight: float = 0.7
    lexical_weight: float = 0.3
    filters: Dict[str, Any] = field(default_factory=dict)
    include_memory: bool = False
    conversation_id: Optional[str] = None


@dataclass
class RetrievalResult:
    query: str
    evidence: List[Evidence]
    dense_count: int = 0
    lexical_count: int = 0
    latency_ms: float = 0.0


@dataclass
class ContextBudget:
    model_ctx_size: int = 8192
    reserve_generation_tokens: int = 2048
    reserve_system_tokens: int = 512
    max_rag_tokens: int = 4096
    chars_per_token: float = 3.8

    @property
    def max_rag_chars(self) -> int:
        return int(self.max_rag_tokens * self.chars_per_token)


@dataclass
class GroundedPrompt:
    prompt_text: str
    evidence: List[Evidence]
    total_chunks_injected: int
    estimated_context_tokens: int
    truncated: bool = False
