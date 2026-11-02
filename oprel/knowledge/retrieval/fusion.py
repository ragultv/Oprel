"""
Reciprocal Rank Fusion (RRF) algorithm for fusing dense and lexical search candidates.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple


class ReciprocalRankFusion:
    """
    Ranks candidates by fusing rankings from dense vector search and sparse lexical search:
    score(d) = sum( weight / (k + rank(d) + 1) )
    """

    def __init__(self, k: int = 60):
        self.k = k

    def fuse(
        self,
        vector_results: List[Tuple[str, float]],
        lexical_results: List[Tuple[str, float]],
        vector_weight: float = 0.7,
        lexical_weight: float = 0.3,
        top_k: int = 10,
    ) -> List[Tuple[str, float, Optional[float], Optional[float]]]:
        """
        Returns list of (chunk_id, fused_score, dense_score, lexical_score)
        sorted by fused_score descending.
        """
        scores: Dict[str, float] = {}
        dense_scores: Dict[str, float] = {}
        lexical_scores: Dict[str, float] = {}

        # 1. Process vector results
        for rank, (chunk_id, raw_score) in enumerate(vector_results):
            dense_scores[chunk_id] = raw_score
            rrf_score = vector_weight * (1.0 / (self.k + rank + 1))
            scores[chunk_id] = scores.get(chunk_id, 0.0) + rrf_score

        # 2. Process lexical results
        for rank, (chunk_id, raw_score) in enumerate(lexical_results):
            lexical_scores[chunk_id] = raw_score
            rrf_score = lexical_weight * (1.0 / (self.k + rank + 1))
            scores[chunk_id] = scores.get(chunk_id, 0.0) + rrf_score

        # 3. Sort by fused RRF score
        sorted_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)

        return [
            (
                cid,
                scores[cid],
                dense_scores.get(cid),
                lexical_scores.get(cid),
            )
            for cid in sorted_ids[:top_k]
        ]
