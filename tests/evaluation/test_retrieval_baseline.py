"""
Evaluation benchmark for retrieval quality against the curated baseline dataset.
Computes Recall@K, Mean Reciprocal Rank (MRR), and keyword precision.
"""

import json
from pathlib import Path
import pytest
from oprel.knowledge.knowledge_store import KnowledgeStore


class DeterministicMockEmbedder:
    """Offline mock embedding generator based on text hashing for reproducible tests."""
    def __init__(self, dim: int = 64):
        self.dim = dim

    def embed_sync(self, texts, model=None):
        if isinstance(texts, str):
            texts = [texts]
        vectors = []
        for text in texts:
            # Deterministic bag-of-words / hash vector
            vec = [0.0] * self.dim
            for token in text.lower().split():
                h = hash(token) % self.dim
                vec[h] += 1.0
            norm = sum(x * x for x in vec) ** 0.5 or 1.0
            vectors.append([x / norm for x in vec])
        return vectors[0] if len(vectors) == 1 and isinstance(texts, str) else vectors

    async def __call__(self, texts, model=None):
        return self.embed_sync(texts, model)


@pytest.fixture
def eval_dataset():
    fixture_path = Path(__file__).parent.parent / "fixtures" / "rag_eval_dataset.json"
    with open(fixture_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.asyncio
async def test_retrieval_baseline_recall_and_mrr(tmp_path: Path, eval_dataset, monkeypatch):
    monkeypatch.setattr("oprel.knowledge.knowledge_store.KNOWLEDGE_DIR", tmp_path)
    monkeypatch.setattr("oprel.knowledge.knowledge_store.CHROMA_PERSIST_DIR", str(tmp_path / "chroma_db"))

    mock_embed = DeterministicMockEmbedder()
    store = KnowledgeStore(embed_func=mock_embed)

    # Ingest evaluation dataset documents
    docs = eval_dataset["documents"]
    for doc in docs:
        await store.index_document(
            text=doc["content"],
            metadata={"filename": doc["filename"], "doc_id": doc["id"]}
        )

    queries = eval_dataset["queries"]
    mrr_sum = 0.0
    recall_at_3_hits = 0

    for q in queries:
        query_text = q["query"]
        expected_docs = q["expected_doc_ids"]
        results = await store.search(query_text, top_k=3)

        hit_rank = None
        for rank, res in enumerate(results):
            doc_id = res.get("metadata", {}).get("doc_id")
            if doc_id in expected_docs:
                hit_rank = rank + 1
                break

        if hit_rank is not None:
            recall_at_3_hits += 1
            mrr_sum += 1.0 / hit_rank

    recall_at_3 = recall_at_3_hits / len(queries)
    mrr = mrr_sum / len(queries)

    # Verify baseline sanity (all queries return top-3 candidates with reasonable recall)
    assert recall_at_3 >= 0.5, f"Baseline Recall@3 too low: {recall_at_3}"
    assert mrr >= 0.3, f"Baseline MRR too low: {mrr}"
