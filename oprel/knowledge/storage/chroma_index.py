"""
ChromaDB vector store adapter conforming to VectorIndex interface.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from oprel.knowledge.interfaces import VectorIndex
from oprel.knowledge.models import Chunk

logger = logging.getLogger("oprel.knowledge.storage.chroma_index")

try:
    import chromadb
except ImportError:
    chromadb = None


class ChromaIndex(VectorIndex):
    """
    Persistent ChromaDB vector index implementation with cosine distance metrics.
    """

    def __init__(
        self,
        persist_dir: Path,
        collection_name: str = "oprel_knowledge",
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self._init_chroma()

    def _init_chroma(self) -> None:
        if chromadb is None:
            logger.warning("chromadb package not installed. Vector search will be disabled.")
            return

        try:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(self.persist_dir))
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"ChromaIndex connected at {self.persist_dir} (collection: {self.collection_name})")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB at {self.persist_dir}: {e}")
            self.client = None
            self.collection = None

    async def upsert(self, chunks: List[Chunk]) -> None:
        if not self.collection or not chunks:
            return

        valid = [c for c in chunks if c.embedding is not None]
        if not valid:
            return

        ids = [c.id for c in valid]
        embeddings = [c.embedding for c in valid]
        documents = [c.text for c in valid]
        metadatas = []
        for c in valid:
            meta = c.metadata.copy()
            meta["doc_id"] = c.document_id
            meta["version_id"] = c.version_id
            meta["chunk_index"] = c.chunk_index
            metadatas.append(meta)

        try:
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
        except Exception as e:
            logger.error(f"ChromaIndex upsert failed: {e}")
            raise

    async def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float]]:
        if not self.collection or not query_vector:
            return []

        try:
            query_kwargs: Dict[str, Any] = {
                "query_embeddings": [query_vector],
                "n_results": top_k,
                "include": ["distances"],
            }
            if filters:
                query_kwargs["where"] = filters

            results = self.collection.query(**query_kwargs)

            output: List[Tuple[str, float]] = []
            if results["ids"] and results["ids"][0]:
                for idx, chunk_id in enumerate(results["ids"][0]):
                    dist = results["distances"][0][idx] if results.get("distances") else 0.0
                    similarity = max(0.0, 1.0 - dist)
                    output.append((chunk_id, similarity))
            return output
        except Exception as e:
            logger.error(f"ChromaIndex search error: {e}")
            return []

    async def delete(self, chunk_ids: List[str]) -> None:
        if not self.collection or not chunk_ids:
            return
        try:
            self.collection.delete(ids=chunk_ids)
        except Exception as e:
            logger.error(f"ChromaIndex deletion failed for IDs: {e}")
