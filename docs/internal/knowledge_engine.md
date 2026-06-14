# Knowledge Engine Internals

The `oprel/knowledge/` module provides a fully local vector-database abstraction designed to support the RAG feature.

## 1. The Sync Engine (`sync_engine.py`)
The sync engine acts as the ingestion pipeline.
- **File Parsing**: It recursively crawls directories. It uses libraries like `PyMuPDF` or `unstructured` to crack open PDFs, Word documents, Markdown, and TXT files, extracting raw text strings.
- **Chunking Strategy**: It splits massive books or documents into semantic chunks (usually around 500-1000 tokens) with a sliding overlap to ensure context isn't lost across chunk boundaries.
- **Embedding Push**: It fires these chunks asynchronously against the internal `/v1/embeddings` endpoint, turning text into dense mathematical arrays.

## 2. Vector Database (`knowledge_store.py`)
Oprel abstracts the actual database layer. It typically wraps a high-speed vector engine (like `chromadb` or a lightweight `FAISS` implementation using SQLite bindings).
- **Storage**: Embeddings are stored alongside their original text chunks and crucial metadata (like the source filename and page number).
- **Querying**: When a chat request triggers RAG, `knowledge_store.search()` converts the user's prompt into a vector using the exact same embedding model used during ingestion. It then calculates the Cosine Distance between the prompt vector and the database vectors.
- **Top-K Retrieval**: The module sorts the vectors by distance (nearest neighbors) and returns the Top-K (default: 5) most relevant text chunks back to the generation service to be injected into the LLM's system prompt.
