"""
Unit tests for Oprel Ingestion Pipeline, Parser Registry, Semantic Chunker, and Durable Jobs.
"""

from pathlib import Path
import pytest
from oprel.knowledge.ingestion.parser_registry import ParserRegistry, DefaultFileParser
from oprel.knowledge.ingestion.chunker import SemanticChunker
from oprel.knowledge.ingestion.jobs import JobManager
from oprel.knowledge.ingestion.pipeline import IngestionPipeline
from oprel.knowledge.models import JobStatus


@pytest.mark.asyncio
async def test_parser_registry_and_extraction(tmp_path: Path):
    registry = ParserRegistry()
    md_file = tmp_path / "test_doc.md"
    md_file.write_text("# Guide to Architecture\n\nThis is paragraph one.\n\n## Section 2\nDetails here.", encoding="utf-8")

    parser = registry.get_parser(md_file)
    assert isinstance(parser, DefaultFileParser)

    content, metadata = await parser.parse(md_file)
    assert "Guide to Architecture" in content
    assert metadata["filename"] == "test_doc.md"
    assert metadata["title"] == "Guide to Architecture"


def test_semantic_chunker_structure_and_offsets():
    chunker = SemanticChunker(chunk_size=150, chunk_overlap=30, min_chunk_len=20)
    sample_text = (
        "# Overview\n\n"
        "The Oprel SDK provides high-performance inference on local hardware.\n\n"
        "## Performance\n\n"
        "It supports CPU and GPU layer offloading with minimal latency overhead."
    )

    chunks = chunker.chunk(sample_text, document_id="doc1", version_id="v1")
    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.document_id == "doc1"
        assert chunk.version_id == "v1"
        assert chunk.text in sample_text
        assert chunk.start_char >= 0
        assert chunk.end_char > chunk.start_char
        # Deterministic IDs
        assert len(chunk.id) == 64


def test_job_manager_persistence_and_recovery(tmp_path: Path):
    manager = JobManager(tmp_path)
    job = manager.create_job("doc_abc")
    assert job.status == JobStatus.PENDING

    manager.update_job(job.id, JobStatus.PROCESSING)
    assert manager.get_job(job.id).status == JobStatus.PROCESSING

    # Simulate process restart by reloading from disk
    recovered_manager = JobManager(tmp_path)
    recovered_job = recovered_manager.get_job(job.id)
    assert recovered_job is not None
    assert recovered_job.status == JobStatus.PROCESSING

    uncompleted = recovered_manager.get_uncompleted_jobs()
    assert len(uncompleted) == 1
    assert uncompleted[0].id == job.id


@pytest.mark.asyncio
async def test_pipeline_file_versioning(tmp_path: Path):
    pipeline = IngestionPipeline(storage_dir=tmp_path)
    doc_file = tmp_path / "changelog.txt"
    doc_file.write_text("Initial release version 1.0.0 with core features.", encoding="utf-8")

    doc1, v1, chunks1, job1 = await pipeline.process_file(doc_file)
    assert doc1.active_version == 1
    assert v1.version_num == 1
    assert job1.status == JobStatus.COMPLETED

    # Modify file and process again -> active_version becomes 2
    doc_file.write_text("Updated release version 1.1.0 with advanced RAG indexing.", encoding="utf-8")
    doc2, v2, chunks2, job2 = await pipeline.process_file(doc_file, existing_doc=doc1)
    assert doc2.active_version == 2
    assert v2.version_num == 2
    assert doc2.content_hash != doc1.content_hash
