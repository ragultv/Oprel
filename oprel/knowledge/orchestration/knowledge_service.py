"""
Unified KnowledgeService orchestrator for Oprel Knowledge, RAG, and Supermemory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from oprel.knowledge.config import KNOWLEDGE_DIR
from oprel.knowledge.ingestion.pipeline import IngestionPipeline
from oprel.knowledge.models import (
    ContextBudget,
    Document,
    Evidence,
    GroundedPrompt,
    RetrievalQuery,
    RetrievalResult,
)
from oprel.knowledge.orchestration.context_builder import DynamicContextBuilder
from oprel.knowledge.orchestration.query_router import QueryRouter, QueryRouteType
from oprel.knowledge.retrieval.hybrid_search import HybridSearchEngine
from oprel.knowledge.storage.repository import UnifiedKnowledgeRepository

logger = logging.getLogger("oprel.knowledge.orchestration.service")


class KnowledgeService:
    """
    Unified high-level facade for all knowledge operations:
    Ingestion, Storage Catalog, Hybrid Search, Query Routing, and Context Grounding.
    """

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        embed_func: Optional[Any] = None,
    ):
        self.base_dir = base_dir or KNOWLEDGE_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.repository = UnifiedKnowledgeRepository(base_dir=self.base_dir)
        self.pipeline = IngestionPipeline(storage_dir=self.base_dir)

        # Wrap embed_func into an adapter if provided
        class _EmbedAdapter:
            def __init__(self, func):
                self.func = func
            async def embed(self, texts: list[str]) -> list[list[float]]:
                import inspect
                if inspect.iscoroutinefunction(self.func):
                    return await self.func(texts)
                return self.func(texts)
            async def embed_single(self, text: str) -> list[float]:
                res = await self.embed([text])
                return res[0] if res else []

        embedder = _EmbedAdapter(embed_func) if embed_func else None
        self.search_engine = HybridSearchEngine(repository=self.repository, embedder=embedder)
        self.router = QueryRouter()
        self.context_builder = DynamicContextBuilder()

    async def ingest_file(self, file_path: Path) -> Document:
        """Parse, version, and index a file into the knowledge repository."""
        doc, version, chunks, job = await self.pipeline.process_file(file_path)
        await self.repository.save_document(doc)
        await self.repository.save_version(version)
        await self.repository.save_chunks(chunks)
        return doc

    async def ingest_text(
        self,
        text: str,
        filename: str = "snippet.txt",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Document:
        """Ingest raw text into structured versioned chunks."""
        doc, version, chunks, job = await self.pipeline.process_raw_text(
            text=text, filename=filename, extra_metadata=extra_metadata
        )
        await self.repository.save_document(doc)
        await self.repository.save_version(version)
        await self.repository.save_chunks(chunks)
        return doc

    async def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> RetrievalResult:
        """Run hybrid search across FTS5 and vector stores."""
        retrieval_query = RetrievalQuery(
            text=query,
            top_k=top_k,
            filters=filters or {},
        )
        return await self.search_engine.search(retrieval_query)

    async def ground_query(
        self,
        query: str,
        model_ctx_size: int = 8192,
        conv_id: Optional[str] = None,
        context_hint: Optional[str] = None,
        force_rag: Optional[bool] = None,
    ) -> GroundedPrompt:
        """
        Orchestrate retrieval, query expansion, and token-aware prompt grounding.
        """
        route = self.router.route(query, has_active_conversation=bool(conv_id), force_rag=force_rag)

        # 1. Expand query if needed
        expanded_q = self.router.expand_query(query, context_hint=context_hint)

        # 2. Retrieve document evidence if routed to DOCUMENT or HYBRID
        evidence: List[Evidence] = []
        if route in (QueryRouteType.DOCUMENT, QueryRouteType.HYBRID):
            search_res = await self.search(expanded_q, top_k=5)
            evidence = search_res.evidence

        # 3. Retrieve Supermemory if routed to MEMORY or HYBRID
        memory_context: Optional[str] = None
        if route in (QueryRouteType.MEMORY, QueryRouteType.HYBRID) and conv_id:
            try:
                from oprel.knowledge.supermemory import SuperMemoryStore
                store = SuperMemoryStore()
                turns = await store.recall_relevant_turns(conv_id, expanded_q, top_k=3)
                if turns:
                    memory_context = "\n".join(
                        f"[Turn {t['turn_index']} - {t['role']}]: {t['content']}" for t in turns
                    )
            except Exception as e:
                logger.debug(f"Supermemory recall ignored in grounding: {e}")

        # 4. Build budgeted grounded prompt
        budget = ContextBudget(model_ctx_size=model_ctx_size)
        return self.context_builder.build_grounded_prompt(
            query=query,
            evidence=evidence,
            budget=budget,
            memory_context=memory_context,
        )

    async def list_documents(self) -> List[Document]:
        return await self.repository.list_documents()

    async def delete_document(self, document_id: str) -> bool:
        return await self.repository.delete_document(document_id)

    async def delete_all_documents(self) -> None:
        await self.repository.delete_all()
