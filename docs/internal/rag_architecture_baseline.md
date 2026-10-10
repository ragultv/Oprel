# Oprel Knowledge & RAG Architecture Baseline

## 1. Overview and Invariants
This document captures the baseline architecture, storage layout, data flow, and retrieval contracts of the Oprel Knowledge and Retrieval-Augmented Generation (RAG) subsystem prior to the multi-phase production upgrade.

## 2. On-Disk Layout
All knowledge data resides under the configured `OPREL_HOME` directory (defaults to `~/.cache/oprel`):

| Path | Purpose | Format | Persistence |
| :--- | :--- | :--- | :--- |
| `~/.cache/oprel/knowledge/chroma_db/` | Vector store index (`oprel_knowledge` collection) | ChromaDB / DuckDB / Parquet / SQLite | Persistent across daemon restarts |
| `~/.cache/oprel/knowledge/bm25_docs.json` | Keyword search cache for BM25 ranking | JSON array of `{id, text, metadata}` | Disk cache, loaded into RAM |
| `~/.cache/oprel/knowledge/dedup_store.json` | Hash deduplication tracking | JSON array of SHA-256 strings | Disk cache, appended on ingest |
| `~/.cache/oprel/knowledge/failure_queue.json` | Failed ingestion tasks queue | JSON array of `{source, error, timestamp, retries}` | Append-only error tracking |
| `~/.cache/oprel/knowledge/my_docs/` | Default local directory watched for sync | Raw files (`.txt`, `.md`, `.pdf`) | User source directory |
| `~/.cache/oprel/models/chat_history.db` | Daemon SQLite DB (conversations, messages, supermemory) | SQLite 3 | WAL mode, thread-safe |

## 3. Data Flow

### 3.1 Ingestion Flow (`sync_engine.py`)
1. **Source Discovery**: Local watch directory scanned recursively or single file/text received via CLI / API.
2. **Deduplication**: SHA-256 hash calculated over raw text content. If hash exists in `dedup_store.json`, ingestion skips.
3. **Chunking**: Naive fixed-size character slicing (Default `CHUNK_SIZE = 400`, `CHUNK_OVERLAP = 80`). Discards chunks below `MIN_CHUNK_LEN = 60`.
4. **Embedding**: Batched call via `oprel.embed` (or daemon embedding endpoint `nomic-embed-text`).
5. **Storage**:
   - Chunks stored in ChromaDB collection with metadata (`filename`, `type`, `chunk_index`, `timestamp`).
   - Chunks appended to `bm25_docs.json` for lexical search.

### 3.2 Retrieval & Generation Flow (`generation.py`, `knowledge_store.py`)
1. **Query Processing**: User generation prompt intercepted when `params.rag = True`.
2. **Hybrid Search**:
   - Vector Search: Query embedded and searched against ChromaDB (Top-2K).
   - Lexical Search: Query tokenized and scored via `rank_bm25.BM25Okapi` (Top-2K).
3. **Fusion (RRF)**:
   - Reciprocal Rank Fusion: $\text{Score}(d) = \sum \frac{\text{weight}}{60 + \text{rank} + 1}$
   - Vector weight: `0.7`, BM25 weight: `0.3`.
   - Results sorted by fusion score, top $K$ (default 5) selected.
4. **Context Construction**:
   - Chunks concatenated with `Source [i] (filename): \n <text>` up to a hard cap of 4,000 characters.
   - Prepended to prompt under `CONTEXT FROM LOCAL KNOWLEDGE BASE`.

### 3.3 Conversational Recall (`supermemory.py`)
1. Active sliding window (default 6 or 10 turns) stays in primary context.
2. Turns beyond window recorded to SQLite `conversation_memories` and ChromaDB `oprel_supermemory`.
3. Short user queries undergo anaphora resolution / token expansion using previous turn keywords.
4. Historical turns retrieved via dense/lexical matching and injected under `--- RELEVANT CONVERSATION MEMORY (SUPERMEMORY) ---`.

## 4. Key Contracts & Extension Points
The upgrade preserves these core behaviors while introducing:
- Versioned documents and chunk lineage.
- Structured Evidence objects replacing plain text concatenation.
- SQLite catalog with FTS5 replacing `bm25_docs.json`.
- Dynamic token budgeting based on active model context limits.
