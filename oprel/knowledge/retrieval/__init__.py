"""
Retrieval subsystem for Oprel Knowledge & RAG.
"""

from oprel.knowledge.retrieval.fusion import ReciprocalRankFusion
from oprel.knowledge.retrieval.dedup import ChunkDeduplicator
from oprel.knowledge.retrieval.hybrid_search import HybridSearchEngine

__all__ = [
    "ReciprocalRankFusion",
    "ChunkDeduplicator",
    "HybridSearchEngine",
]
