"""
Unit tests for P4 Knowledge Orchestration, Query Routing, and Dynamic Context Building.
"""

from pathlib import Path
import pytest

from oprel.knowledge.models import ContextBudget, Evidence
from oprel.knowledge.orchestration.context_builder import DynamicContextBuilder
from oprel.knowledge.orchestration.knowledge_service import KnowledgeService
from oprel.knowledge.orchestration.query_router import QueryRouter, QueryRouteType


def test_query_router_intent_classification():
    router = QueryRouter()

    # Document-intent queries
    assert router.route("What is the architecture according to the docs?") == QueryRouteType.DOCUMENT
    assert router.route("How to configure the daemon on Windows?") == QueryRouteType.DOCUMENT

    # Memory-intent queries
    assert router.route("What did we discuss earlier?") == QueryRouteType.MEMORY
    assert router.route("Remember what my preferred quantization was?") == QueryRouteType.MEMORY

    # Forced RAG
    assert router.route("Hello world", force_rag=False) == QueryRouteType.DIRECT
    assert router.route("Tell me a story", force_rag=True) == QueryRouteType.DOCUMENT


def test_dynamic_context_builder_budget_and_citations():
    builder = DynamicContextBuilder()
    evidence = [
        Evidence(
            chunk_id="c1",
            document_id="d1",
            filename="arch.md",
            text="The daemon utilizes SQLite for state storage.",
            score=0.9,
            metadata={"heading": "Storage Engine"}
        ),
        Evidence(
            chunk_id="c2",
            document_id="d2",
            filename="runtime.md",
            text="GPU offloading layer count can be set via --n-gpu-layers.",
            score=0.8,
            metadata={"heading": "GPU Acceleration"}
        ),
    ]

    budget = ContextBudget(model_ctx_size=4096, max_rag_tokens=1000)
    grounded = builder.build_grounded_prompt(
        query="How does storage and GPU configuration work?",
        evidence=evidence,
        budget=budget,
    )

    assert "Source [1] (arch.md, Storage Engine):" in grounded.prompt_text
    assert "Source [2] (runtime.md, GPU Acceleration):" in grounded.prompt_text
    assert "The daemon utilizes SQLite for state storage." in grounded.prompt_text
    assert "QUESTION: How does storage and GPU configuration work?" in grounded.prompt_text
    assert "Cite source labels [1], [2]" in grounded.prompt_text
    assert grounded.total_chunks_injected == 2
    assert grounded.truncated is False


@pytest.mark.asyncio
async def test_knowledge_service_end_to_end(tmp_path: Path):
    service = KnowledgeService(base_dir=tmp_path / "orchestrator_test")

    # Ingest document
    doc = await service.ingest_text(
        text=(
            "# Server Networking\n\n"
            "Oprel exposes an HTTP daemon at port 11435 for Windows compatibility.\n\n"
            "## Socket Mode\n\n"
            "On macOS and Linux, the server uses Unix domain sockets for near-zero IPC overhead."
        ),
        filename="networking.md"
    )
    assert doc.id is not None

    # Search evidence
    search_res = await service.search("Unix domain sockets networking", top_k=2)
    assert len(search_res.evidence) > 0
    assert search_res.evidence[0].filename == "networking.md"

    # Ground query
    grounded = await service.ground_query(
        query="What IPC mechanism does Oprel use on Linux?",
        model_ctx_size=8192,
    )
    assert "Source [1] (networking.md" in grounded.prompt_text
    assert "Unix domain sockets" in grounded.prompt_text
