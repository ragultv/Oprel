"""
Unit tests for P2 Storage Catalog, FTS5 Lexical Index, and Migration.
"""

import json
from pathlib import Path
import pytest

from oprel.knowledge.models import (
    Chunk,
    Document,
    DocumentMetadata,
    DocumentVersion,
    JobStatus,
)
from oprel.knowledge.storage.sqlite_catalog import SqliteCatalog
from oprel.knowledge.storage.fts_index import FtsIndex
from oprel.knowledge.storage.repository import UnifiedKnowledgeRepository
from oprel.knowledge.storage.migration import LegacyDataMigrator


def test_sqlite_catalog_crud(tmp_path: Path):
    db_path = tmp_path / "catalog.db"
    catalog = SqliteCatalog(db_path)

    doc = Document(
        id="doc_1",
        content_hash="hash_1",
        metadata=DocumentMetadata(
            filename="intro.md",
            file_type="md",
            size_bytes=500,
        ),
        active_version=1,
        status=JobStatus.COMPLETED
    )
    catalog.save_document(doc)

    retrieved = catalog.get_document("doc_1")
    assert retrieved is not None
    assert retrieved.metadata.filename == "intro.md"

    # Add chunks
    chunk = Chunk(
        id="c1",
        document_id="doc_1",
        version_id="doc_1_v1",
        chunk_index=0,
        text="Sample text content for retrieval",
        metadata={"filename": "intro.md"}
    )
    catalog.save_chunks([chunk])

    retrieved_chunks = catalog.get_chunks_by_ids(["c1"])
    assert len(retrieved_chunks) == 1
    assert retrieved_chunks[0].text == "Sample text content for retrieval"

    # Delete document
    assert catalog.delete_document("doc_1") is True
    assert catalog.get_document("doc_1") is None
    assert catalog.get_chunks_by_ids(["c1"]) == []


@pytest.mark.asyncio
async def test_fts5_lexical_search(tmp_path: Path):
    db_path = tmp_path / "fts_test.db"
    fts = FtsIndex(db_path)

    chunks = [
        Chunk(id="c1", document_id="d1", version_id="v1", chunk_index=0, text="Deep learning models use CUDA acceleration on NVIDIA GPUs."),
        Chunk(id="c2", document_id="d2", version_id="v1", chunk_index=0, text="SQLite database stores transactional records with write-ahead logging."),
    ]
    await fts.upsert(chunks)

    # Search for CUDA
    results = await fts.search("CUDA NVIDIA", top_k=5)
    assert len(results) > 0
    assert results[0][0] == "c1"

    # Search for SQLite
    results_sqlite = await fts.search("SQLite logging", top_k=5)
    assert len(results_sqlite) > 0
    assert results_sqlite[0][0] == "c2"

    # Delete chunk
    await fts.delete(["c1"])
    results_after_del = await fts.search("CUDA", top_k=5)
    assert len(results_after_del) == 0


@pytest.mark.asyncio
async def test_legacy_data_migration(tmp_path: Path):
    legacy_json = tmp_path / "bm25_docs.json"
    legacy_data = [
        {
            "id": "legacy_chunk_1",
            "text": "Oprel runs locally on device without cloud dependencies.",
            "metadata": {"filename": "local_guide.md", "doc_id": "guide_01", "chunk_index": 0}
        }
    ]
    legacy_json.write_text(json.dumps(legacy_data), encoding="utf-8")

    repo = UnifiedKnowledgeRepository(base_dir=tmp_path / "repo")
    migrator = LegacyDataMigrator(knowledge_dir=tmp_path, repository=repo)

    count = await migrator.migrate_legacy_bm25()
    assert count == 1

    # Verify searchable via FTS5
    fts_results = await repo.fts.search("Oprel dependencies", top_k=5)
    assert len(fts_results) == 1
    assert fts_results[0][0] == "legacy_chunk_1"
