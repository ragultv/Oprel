# Knowledge Base & Retrieval-Augmented Generation (RAG)

Oprel includes a native, fully-local Retrieval-Augmented Generation pipeline. This system allows you to feed your personal documents (PDFs, Markdown files, Word documents) into a vector database, enabling the LLM to read and reference them when answering your questions.

## The Sync Engine

The `sync_engine.py` is responsible for parsing files on your hard drive, chunking them, generating embeddings, and storing them in the local Knowledge Store.

### Indexing Files via CLI
To add files to your local knowledge base:
```bash
oprel index add /path/to/my_documents/
```
The sync engine will recursively scan the directory, extract text using the `unstructured` library, and compute vectors.

### Manual Synchronization
If you modify the original files, you can tell Oprel to rescan and update the vectors:
```bash
oprel index sync
```

## Searching the Knowledge Base

Before passing context to the LLM, you can test the retrieval quality directly from the CLI. This performs a Cosine Similarity search over your local vector store.
```bash
oprel index search "What is Oprel architecture?" --top-k 5
```

## RAG in Generation

When executing a generation request, you can enable the RAG flag. 
```bash
oprel chat llama3-8b --rag
```

**How it works (inside `generation.py`):**
1. Oprel intercepts your prompt.
2. It calls the `KnowledgeStore.search()` function to retrieve the Top-K relevant document chunks.
3. It constructs a massive prompt containing a `CONTEXT FROM LOCAL KNOWLEDGE BASE` section.
4. It instructs the LLM to answer *only* using the provided context and to cite the sources (e.g., "According to [1]...").

By integrating RAG directly into the daemon layer, frontends (like Oprel Studio) do not have to implement complex vector search logic themselves.
