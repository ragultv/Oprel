"""
Hybrid search engine combining Chroma vector search and SQLite FTS5 lexical search.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from oprel.knowledge.interfaces import (
    EmbeddingProvider,
    KnowledgeRepository,
    RetrievalService,
)
from oprel.knowledge.models import (
    Evidence,
    RetrievalQuery,
    RetrievalResult,
)
from oprel.knowledge.retrieval.dedup import ChunkDeduplicator
from oprel.knowledge.retrieval.fusion import ReciprocalRankFusion

logger = logging.getLogger("oprel.knowledge.retrieval.hybrid_search")


class HybridSearchEngine(RetrievalService):
    """
    Executes hybrid retrieval:
    1. Concurrent dense vector query + FTS5 lexical query
    2. Reciprocal Rank Fusion (RRF)
    3. Metadata filtering & threshold verification
    4. Token-level chunk deduplication
    5. Returns ranked Evidence list with provenance metadata
    """

    def __init__(
        self,
        repository: KnowledgeRepository,
        embedder: Optional[EmbeddingProvider] = None,
        fusion: Optional[ReciprocalRankFusion] = None,
        deduplicator: Optional[ChunkDeduplicator] = None,
    ):
        self.repository = repository
        self.embedder = embedder
        self.fusion = fusion or ReciprocalRankFusion(k=60)
        self.deduplicator = deduplicator or ChunkDeduplicator(jaccard_threshold=0.85)

    async def search(self, query: RetrievalQuery) -> RetrievalResult:
        start_time = time.perf_counter()
        if not query.text or not query.text.strip():
            return RetrievalResult(query=query.text, evidence=[], latency_ms=0.0)

        # 1. Dense search
        vector_candidates: List[tuple[str, float]] = []
        if self.embedder is not None and hasattr(self.repository, "vector_index"):
            try:
                query_vector = await self.embedder.embed_single(query.text)
                if query_vector:
                    vector_candidates = await self.repository.vector_index.search(
                        query_vector=query_vector,
                        top_k=query.top_k * 2,
                        filters=query.filters if query.filters else None,
                    )
            except Exception as e:
                logger.warning(f"Vector search failed in hybrid search: {e}")

        # 2. Lexical search via SQLite FTS5
        lexical_candidates: List[tuple[str, float]] = []
        if hasattr(self.repository, "fts"):
            try:
                lexical_candidates = await self.repository.fts.search(
                    query=query.text,
                    top_k=query.top_k * 2,
                )
            except Exception as e:
                logger.warning(f"FTS5 lexical search failed in hybrid search: {e}")

        # 3. Reciprocal Rank Fusion
        fused = self.fusion.fuse(
            vector_results=vector_candidates,
            lexical_results=lexical_candidates,
            vector_weight=query.dense_weight,
            lexical_weight=query.lexical_weight,
            top_k=query.top_k * 2,
        )

        candidate_ids = [cid for cid, _, _, _ in fused]
        chunks = await self.repository.get_chunks_by_ids(candidate_ids)
        chunk_map = {c.id: c for c in chunks}

        evidence_list: List[Evidence] = []
        rank_idx = 1
        for cid, score, dense_sc, lex_sc in fused:
            if score < query.score_threshold:
                continue

            chunk = chunk_map.get(cid)
            if not chunk:
                continue

            doc_meta = chunk.metadata
            filename = doc_meta.get("filename", "Unknown")

            evidence = Evidence(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                filename=filename,
                text=chunk.text,
                score=score,
                dense_score=dense_sc,
                lexical_score=lex_sc,
                rank=rank_idx,
                source_label=f"[{rank_idx}]",
                metadata=chunk.metadata,
            )
            evidence_list.append(evidence)
            rank_idx += 1

        # 4. Deduplicate overlapping or near-identical evidence chunks
        final_evidence = self.deduplicator.deduplicate(evidence_list)[:query.top_k]

        # Re-index citation labels after deduplication
        for idx, ev in enumerate(final_evidence):
            ev.to_citation_label(idx + 1)

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return RetrievalResult(
            query=query.text,
            evidence=final_evidence,
            dense_count=len(vector_candidates),
            lexical_count=len(lexical_candidates),
            latency_ms=round(latency_ms, 2),
        )
