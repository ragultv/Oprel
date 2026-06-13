import re
import hashlib
import json
import shutil
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

# Optional imports for RAG specialized parts
try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

import requests
from oprel.knowledge.config import (
    KNOWLEDGE_DIR, CHROMA_PERSIST_DIR, CHROMA_SNAPSHOT_DIR, COLLECTION_NAME,
    CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_LEN,
    TOP_K, SCORE_THRESHOLD, BM25_WEIGHT, VECTOR_WEIGHT,
    OPREL_BASE_URL, OPREL_EMBED_MODEL
)
from oprel import embed

logger = logging.getLogger("oprel.knowledge.knowledge_store")

class KnowledgeStore:
    """
    Hybrid search engine combining Vector (ChromaDB) and Keyword (BM25) search.
    Follows the architecture for Index Layer mapping.
    """
    
    def __init__(self, embed_func=None):
        self.embed_func = embed_func or embed
        self.bm25 = None
        self._initialize_vector_db()
        self._initialize_bm25()
        
    def _initialize_vector_db(self):
        if chromadb is None:
            logger.warning("chromadb not installed. Vector search will be disabled.")
            self.client = None
            self.collection = None
            return
            
        try:
            self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
            self.collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Initialized ChromaDB at {CHROMA_PERSIST_DIR}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            self.client = None
            self.collection = None

    def _initialize_bm25(self):
        self.bm25 = None
        self.bm25_docs = []
        # BM25 is built from current documents during search or initialization
        # In a production system, we'd persist this index.
        # For this implementation, we'll try to load cached documents if any.
        self._load_bm25_cache()

    def _load_bm25_cache(self):
        """Load BM25 docs from disk"""
        cache_path = KNOWLEDGE_DIR / "bm25_docs.json"
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    self.bm25_docs = json.load(f)
                logger.info(f"Loaded {len(self.bm25_docs)} documents into BM25 cache")
            except Exception as e:
                logger.error(f"Failed to load BM25 cache: {e}")

    def _save_bm25_cache(self):
        """Save BM25 docs to disk"""
        cache_path = KNOWLEDGE_DIR / "bm25_docs.json"
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.bm25_docs, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save BM25 cache: {e}")

    async def index_document(self, text: str, metadata: Dict = None):
        """
        Index a single text chunk into both Vector and BM25 stores.
        """
        await self.index_documents([text], [metadata] if metadata else None)

    async def index_documents(self, text_list: List[str], metadata_list: List[Dict] = None):
        """
        Index multiple text chunks into both Vector and BM25 stores.
        This is much more efficient than calling index_document multiple times,
        especially when using the Oprel embedding API.
        """
        if not text_list:
            return

        valid_texts = []
        valid_metadatas = []
        ids = []

        for i, text in enumerate(text_list):
            if not text or len(text) < MIN_CHUNK_LEN:
                continue
                
            metadata = (metadata_list[i] if metadata_list and i < len(metadata_list) else {}) or {}
            metadata["timestamp"] = datetime.now().isoformat()
            
            doc_id = hashlib.sha256(text.encode()).hexdigest()
            
            valid_texts.append(text)
            valid_metadatas.append(metadata)
            ids.append(doc_id)

        if not valid_texts:
            return

        # 1. Index in Vector DB
        if self.collection:
            try:
                # Use Oprel's internal embedding API (Support both sync and async)
                import inspect
                if inspect.iscoroutinefunction(self.embed_func):
                    vectors = await self.embed_func(valid_texts, model=OPREL_EMBED_MODEL)
                else:
                    vectors = self.embed_func(valid_texts, model=OPREL_EMBED_MODEL)
                    
                self.collection.add(
                    ids=ids,
                    embeddings=vectors,
                    documents=valid_texts,
                    metadatas=valid_metadatas
                )
            except Exception as e:
                logger.error(f"Error indexing in vector db: {e}")
                
        # 2. Add to BM25 candidate list
        for i in range(len(valid_texts)):
            self.bm25_docs.append({
                "id": ids[i], 
                "text": valid_texts[i], 
                "metadata": valid_metadatas[i]
            })
            
        self.bm25 = None # Reset BM25 index so it is rebuilt on next search
        self._save_bm25_cache()

    async def search(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        """
        Perform hybrid search using Vector + BM25 with RRF reranking.
        """
        if not query:
            return []
            
        vector_results = await self._vector_search(query, top_k * 2)
        keyword_results = self._keyword_search(query, top_k * 2)
        
        # RRF Reranking
        hybrid_results = self._rerank(vector_results, keyword_results, top_k)
        return hybrid_results

    async def _vector_search(self, query: str, n: int) -> List[Dict]:
        if not self.collection:
            return []
            
        try:
            import inspect
            if inspect.iscoroutinefunction(self.embed_func):
                vector = await self.embed_func(query, model=OPREL_EMBED_MODEL)
            else:
                vector = self.embed_func(query, model=OPREL_EMBED_MODEL)
                
            results = self.collection.query(
                query_embeddings=[vector],
                n_results=n,
                include=["documents", "metadatas", "distances"]
            )
            
            output = []
            if results["ids"] and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    output.append({
                        "id": results["ids"][0][i],
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "score": 1.0 - results["distances"][0][i]  # Convert distance to similarity
                    })
            return output
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def _keyword_search(self, query: str, n: int) -> List[Dict]:
        if not BM25Okapi or not self.bm25_docs:
            return []
            
        if self.bm25 is None:
            logger.info("Building BM25 index...")
            tokenized_corpus = [doc["text"].lower().split() for doc in self.bm25_docs]
            self.bm25 = BM25Okapi(tokenized_corpus)
        
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Sort and take top n
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        
        output = []
        for i in ranked_indices:
            if scores[i] > 0:
                doc = self.bm25_docs[i].copy()
                doc["score"] = scores[i]
                output.append(doc)
        return output

    def _rerank(self, vector_res: List[Dict], keyword_res: List[Dict], top_k: int) -> List[Dict]:
        """
        Reciprocal Rank Fusion (RRF)
        score = sum( 1 / (k + rank) )
        """
        rrf_k = 60
        scores = {}  # doc_id -> score
        docs_map = {} # doc_id -> doc object
        
        # Process vector results
        for rank, res in enumerate(vector_res):
            doc_id = res["id"]
            scores[doc_id] = scores.get(doc_id, 0) + VECTOR_WEIGHT * (1.0 / (rrf_k + rank + 1))
            docs_map[doc_id] = res
            
        # Process keyword results
        for rank, res in enumerate(keyword_res):
            doc_id = res["id"]
            scores[doc_id] = scores.get(doc_id, 0) + BM25_WEIGHT * (1.0 / (rrf_k + rank + 1))
            if doc_id not in docs_map:
                docs_map[doc_id] = res
                
        # Sort and return top_k
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        final_results = []
        for doc_id in sorted_ids:
            doc = docs_map[doc_id].copy()
            doc["hybrid_score"] = scores[doc_id]
            
            # Apply score threshold if defined
            # Note: RRF scores are normalized relative to ranks, 
            # so thresholding might need calibration.
            if len(final_results) < top_k:
                final_results.append(doc)
            
        return final_results
