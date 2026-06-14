# Embeddings Runtime

The generation of vector embeddings (`oprel/server/services/generation.py`) is decoupled from standard text generation to ensure that high-throughput RAG indexing tasks do not block conversational chats.

## Architecture & Flow

1. **Independent Process**: The embedding model (e.g., `nomic-embed-text.gguf`) is loaded into its own `llama.cpp` server instance, listening on a distinct internal port.
2. **Batch Processing (`embed_chunk`)**:
   - The Python daemon accepts a potentially massive string or an array of strings.
   - It iterates through the text. If an HTTP `500` error is returned by the backend (usually indicating that the text exceeds the model's maximum context length—often 8192 or 2048 tokens), Oprel's `embed_text` handler gracefully degrades into **chunked mode**.
3. **Automatic Chunking & Pooling**:
   - Oprel slices the giant text into 150-word chunks with a 20-word sliding overlap.
   - It fires off parallel embedding requests to the backend for every chunk.
   - Once all chunks return their mathematical vectors, Oprel pools them using an unweighted average.
4. **L2 Normalization**: The resulting pooled vector is mathematically squashed into a normalized unit sphere (L2 normalization). This makes downstream Cosine Similarity math exponentially faster and more accurate.
