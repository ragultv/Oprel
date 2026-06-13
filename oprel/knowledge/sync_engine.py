import hashlib
import logging
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from oprel.knowledge.config import (
    KNOWLEDGE_DIR, CHUNK_SIZE, CHUNK_OVERLAP,
    RSS_FEEDS, WIKIPEDIA_TOPICS, LOCAL_WATCH_DIR
)
from oprel.knowledge.knowledge_store import KnowledgeStore

logger = logging.getLogger("oprel.knowledge.sync_engine")

class DedupStore:
    """
    Persistent store for SHA256 hashes to prevent re-indexing identical content.
    """
    def __init__(self, storage_path: Path):
        self.path = storage_path / "dedup_store.json"
        self.seen_hashes = self._load()
        
    def _load(self) -> set:
        if self.path.exists():
            try:
                with open(self.path, 'r') as f:
                    return set(json.load(f))
            except Exception as e:
                logger.error(f"Failed to load dedup store: {e}")
        return set()
        
    def _save(self):
        try:
            with open(self.path, 'w') as f:
                json.dump(list(self.seen_hashes), f)
        except Exception as e:
            logger.error(f"Failed to save dedup store: {e}")
            
    def is_new(self, content: str) -> bool:
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        if content_hash in self.seen_hashes:
            return False
        self.seen_hashes.add(content_hash)
        self._save()
        return True

class Chunker:
    """
    Splits text into fixed-size chunks with overlap.
    """
    def __init__(self, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        self.size = size
        self.overlap = overlap
        
    def chunk(self, text: str) -> List[str]:
        if not text:
            return []
            
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.size
            chunks.append(text[start:end])
            start += self.size - self.overlap
            
            # Avoid infinite loop if overlap >= size
            if self.overlap >= self.size:
                break
                
        return chunks

class FailureQueue:
    """
    Tracks failed ingestion tasks for retry with exponential backoff.
    """
    def __init__(self, storage_path: Path):
        self.path = storage_path / "failure_queue.json"
        self.queue = self._load()
        
    def _load(self) -> List[Dict]:
        if self.path.exists():
            try:
                with open(self.path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load failure queue: {e}")
        return []
        
    def add(self, source: str, error: str):
        self.queue.append({
            "source": source,
            "error": error,
            "timestamp": datetime.now().isoformat(),
            "retries": 0
        })
        self._save()
        
    def _save(self):
        try:
            with open(self.path, 'w') as f:
                json.dump(self.queue, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save failure queue: {e}")

class SyncEngine:
    """
    Orchestrates the ingestion process: Source -> Dedup -> Chunk -> Store.
    """
    def __init__(self, store: Optional[KnowledgeStore] = None):
        self.store = store or KnowledgeStore()
        self.dedup = DedupStore(KNOWLEDGE_DIR)
        self.chunker = Chunker()
        self.failures = FailureQueue(KNOWLEDGE_DIR)
        
    async def ingest_text(self, text: str, source_metadata: Dict = None):
        """
        Main entry point for ingesting raw text from any source.
        """
        if not self.dedup.is_new(text):
            logger.debug("Content already indexed, skipping.")
            return
            
        chunks = self.chunker.chunk(text)
        logger.info(f"Ingesting {len(chunks)} chunks from source.")
        
        chunk_texts = []
        chunk_metadatas = []
        
        for i, chunk_text in enumerate(chunks):
            metadata = (source_metadata or {}).copy()
            metadata["chunk_index"] = i
            metadata["total_chunks"] = len(chunks)
            
            chunk_texts.append(chunk_text)
            chunk_metadatas.append(metadata)
            
        try:
            await self.store.index_documents(chunk_texts, chunk_metadatas)
        except Exception as e:
            logger.error(f"Failed to batch index chunks: {e}")
            self.failures.add(str(source_metadata), str(e))

    async def ingest_file(self, file_path: Path):
        """
        Ingest a local file (PDF, TXT, MD).
        """
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return
            
        try:
            # Simple text reading for now. In production, use specialized parsers.
            # (Note: PDF parsing would require PyPDF2 as planned)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            metadata = {
                "source": str(file_path),
                "type": file_path.suffix[1:],
                "filename": file_path.name
            }
            await self.ingest_text(content, metadata)
            logger.info(f"Successfully ingested {file_path.name}")
        except Exception as e:
            logger.error(f"Failed to ingest file {file_path}: {e}")
            self.failures.add(str(file_path), str(e))
    async def sync_all(self):
        """
        Orchestrate a full sync of all configured sources.
        """
        logger.info("Starting global knowledge synchronization...")
        
        # 1. Sync local watch directory
        watch_path = Path(LOCAL_WATCH_DIR)
        if watch_path.exists():
            for file in watch_path.rglob("*"):
                if file.suffix.lower() in [".txt", ".md", ".pdf"]:
                    await self.ingest_file(file)

        # 2. Sync Wikipedia (Stub for month 2)
        # 3. Sync RSS (Stub for month 2)
        logger.info("Global sync completed.")
