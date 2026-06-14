# Embeddings

Embeddings are mathematical vectors that represent the semantic meaning of text. Oprel provides a highly optimized pipeline for generating text embeddings using local models like `nomic-embed-text` or `bge-m3`.

## Architecture
The embedding execution flow (`oprel/server/services/generation.py`) is designed to prevent blocking the main LLM processes:
1. **Model Loading**: Embedding models are loaded via the `llama.cpp` backend on a separate port.
2. **Chunking**: If the input text exceeds the model's context window, Oprel automatically splits the text into smaller chunks (with a sliding window overlap).
3. **Pooling**: It generates embeddings for each chunk and pools them together using an averaging technique.
4. **L2 Normalization**: The resulting vector is normalized to a magnitude of 1, allowing for ultra-fast Cosine Similarity comparisons using simple dot products.

## CLI Usage

Generate embeddings from the command line:
```bash
oprel embed "The quick brown fox" --model nomic-embed-text
```

### Batch Processing
You can process entire files or lists of texts using the CLI, and output the result directly to a JSON file.
```bash
# Extract and embed a PDF
oprel embed --model nomic-embed-text --files ./document.pdf -o embeddings.json

# Embed a list of strings from a text file
oprel embed --model nomic-embed-text --batch ./sentences.txt
```

## API Usage

The embeddings endpoint follows the standard OpenAI signature.

**Endpoint:** `POST /v1/embeddings`

**Payload:**
```json
{
  "model": "nomic-embed-text",
  "input": ["sentence 1", "sentence 2"]
}
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.0023, -0.0192, ...],
      "index": 0
    }
  ],
  "model": "nomic-embed-text",
  "usage": {
    "prompt_tokens": 12,
    "total_tokens": 12
  }
}
```
