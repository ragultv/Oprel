"""
Unit tests for Oprel Knowledge baseline components, models, and interfaces.
"""

import hashlib
import pytest
from pathlib import Path
from oprel.knowledge.models import (
    Document,
    DocumentMetadata,
    DocumentVersion,
    Chunk,
    IngestionJob,
    JobStatus,
    Evidence,
    RetrievalQuery,
    RetrievalResult,
    ContextBudget,
    GroundedPrompt,
)
from oprel.knowledge.interfaces import (
    DocumentParser,
    Chunker,
    EmbeddingProvider,
    VectorIndex,
    LexicalIndex,
    KnowledgeRepository,
    RetrievalService,
    ContextBuilder,
)
from oprel.knowledge.sync_engine import Chunker as LegacyChunker, DedupStore


def test_models_instantiation_and_defaults():
    doc = Document(
        id="doc_123",
        content_hash="hash_abc",
        metadata=DocumentMetadata(
            filename="test.txt",
            file_type="txt",
            size_bytes=1024,
            source_path="/path/test.txt"
        )
    )
    assert doc.id == "doc_123"
    assert doc.status == JobStatus.PENDING
    assert doc.active_version == 1

    chunk = Chunk(
        id="chunk_0",
        document_id="doc_123",
        version_id="v1",
        chunk_index=0,
        text="Sample chunk content"
    )
    assert chunk.chunk_index == 0
    assert chunk.text == "Sample chunk content"

    evidence = Evidence(
        chunk_id="chunk_0",
        document_id="doc_123",
        filename="test.txt",
        text="Sample chunk content",
        score=0.88,
        dense_score=0.9,
        lexical_score=0.7,
        rank=1
    )
    label = evidence.to_citation_label(1)
    assert label == "[1]"
    assert evidence.source_label == "[1]"


def test_context_budget_calculations():
    budget = ContextBudget(
        model_ctx_size=8192,
        reserve_generation_tokens=2048,
        reserve_system_tokens=512,
        max_rag_tokens=2048,
        chars_per_token=4.0
    )
    assert budget.max_rag_chars == 8192


def test_legacy_chunker_deterministic_output():
    chunker = LegacyChunker(size=100, overlap=20)
    sample_text = "A" * 250
    chunks = chunker.chunk(sample_text)
    assert len(chunks) == 4
    assert len(chunks[0]) == 100
    assert len(chunks[1]) == 100
    assert len(chunks[2]) == 90
    assert len(chunks[3]) == 10


def test_dedup_store_idempotency(tmp_path: Path):
    dedup = DedupStore(tmp_path)
    sample_content = "Unique document payload for testing dedup"
    assert dedup.is_new(sample_content) is True
    assert dedup.is_new(sample_content) is False
    assert dedup.is_new(sample_content + " diff") is True


def test_rrf_scoring_invariants():
    # Test RRF formula: 1 / (60 + rank + 1)
    rrf_k = 60
    rank_0_score = 1.0 / (rrf_k + 0 + 1)
    rank_1_score = 1.0 / (rrf_k + 1 + 1)
    assert rank_0_score > rank_1_score
    assert pytest.approx(rank_0_score, 0.0001) == 0.016393
