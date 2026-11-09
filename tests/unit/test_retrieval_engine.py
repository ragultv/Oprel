"""
Unit tests for P3 Hybrid Retrieval, RRF Fusion, and Chunk Deduplication.
"""

from pathlib import Path
import pytest

from oprel.knowledge.models import (
    Chunk,
    Document,
    DocumentMetadata,
    Evidence,
    RetrievalQuery,
)
from oprel.knowledge.retrieval.dedup import ChunkDeduplicator
from oprel.knowledge.retrieval.fusion import ReciprocalRankFusion
from oprel.knowledge.retrieval.hybrid_search import HybridSearchEngine
from oprel.knowledge.storage.repository import UnifiedKnowledgeRepository


def test_rrf_fusion_ordering():
    fusion = ReciprocalRankFusion(k=60)
    vector_results = [("chunk_a", 0.95), ("chunk_b", 0.80)]
    lexical_results = [("chunk_b", 12.5), ("chunk_c", 8.0)]

    # chunk_b appears in both and should receive combined score boost
    fused = fusion.fuse(
        vector_results=vector_results,
        lexical_results=lexical_results,
        vector_weight=0.7,
        lexical_weight=0.3,
        top_k=5,
    )
    assert len(fused) == 3
    # Check that chunk_b is properly scored
    fused_map = {cid: score for cid, score, _, _ in fused}
    assert "chunk_b" in fused_map
    assert fused_map["chunk_b"] > 0


def test_chunk_deduplication():
    dedup = ChunkDeduplicator(jaccard_threshold=0.8)
    e1 = Evidence(chunk_id="1", document_id="d1", filename="f.txt", text="The quick brown fox jumps over the lazy dog.", score=0.9)
    # Very similar chunk (only punctuation/minor difference)
    e2 = Evidence(chunk_id="2", document_id="d1", filename="f.txt", text="The quick brown fox jumps over the lazy dog!", score=0.85)
    # Distinct chunk
    e3 = Evidence(chunk_id="3", document_id="d1", filename="f.txt", text="Completely different topic regarding database transactions.", score=0.8)

    filtered = dedup.deduplicate([e1, e2, e3])
    assert len(filtered) == 2
    assert filtered[0].chunk_id == "1"
    assert filtered[1].chunk_id == "3"


@pytest.mark.asyncio
async def test_hybrid_search_engine_e2e(tmp_path: Path):
    repo = UnifiedKnowledgeRepository(base_dir=tmp_path / "hybrid_repo")

    # Seed repository
    doc = Document(
        id="doc_perf",
        content_hash="h1",
        metadata=DocumentMetadata(filename="perf.md", file_type="md", size_bytes=200),
    )
    await repo.save_document(doc)

    chunks = [
        Chunk(
            id="c_llama",
            document_id="doc_perf",
            version_id="v1",
            chunk_index=0,
            text="Llama 3.2 1B runs efficiently on Apple Silicon and modern CPUs with AVX2.",
            metadata={"filename": "perf.md"},
        ),
        Chunk(
            id="c_bench",
            document_id="doc_perf",
            version_id="v1",
            chunk_index=1,
            text="Benchmarking memory bandwidth shows high throughput during prompt processing.",
            metadata={"filename": "perf.md"},
        ),
    ]
    await repo.save_chunks(chunks)

    engine = HybridSearchEngine(repository=repo)
    query = RetrievalQuery(text="Llama Apple Silicon CPUs", top_k=2)

    result = await engine.search(query)
    assert len(result.evidence) > 0
    top_hit = result.evidence[0]
    assert top_hit.chunk_id == "c_llama"
    assert top_hit.source_label == "[1]"
    assert top_hit.filename == "perf.md"
    assert result.latency_ms >= 0.0
