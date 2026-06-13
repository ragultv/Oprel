"""
Embedding Model Usage Examples

This demonstrates how to use embedding models with Oprel SDK
for semantic search, RAG, and text similarity tasks.
"""

# ============================================================================
# Example 1: Simple embedding generation
# ============================================================================

from oprel import embed

# Generate embedding for a single text
text = "Oprel SDK makes AI inference simple and fast"
embedding = embed(text)

print(f"Text: {text}")
print(f"Embedding dimensions: {len(embedding)}")
print(f"First 10 values: {embedding[:10]}")


# ============================================================================
# Example 2: Batch embedding generation
# ============================================================================

texts = [
    "Machine learning is a subset of artificial intelligence",
    "Deep learning uses neural networks with many layers",
    "Natural language processing helps computers understand text",
]

embeddings = embed(texts, model="nomic-embed-text")

for i, (text, emb) in enumerate(zip(texts, embeddings)):
    print(f"\n[{i}] {text}")
    print(f"    Dimensions: {len(emb)}, Magnitude: {sum(x**2 for x in emb)**0.5:.4f}")


# ============================================================================
# Example 3: Semantic similarity search
# ============================================================================

import math
from oprel import Client

client = Client()

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = math.sqrt(sum(x * x for x in a))
    magnitude_b = math.sqrt(sum(x * x for x in b))
    return dot_product / (magnitude_a * magnitude_b)

# Knowledge base
docs = [
    "Python is a high-level programming language",
    "JavaScript runs in web browsers",
    "Machine learning requires large datasets",
    "Neural networks are inspired by the human brain",
]

# Generate embeddings for knowledge base
doc_embeddings = client.embed(docs, model="nomic-embed-text")

# Query
query = "What is neural network architecture?"
query_embedding = client.embed(query, model="nomic-embed-text")

# Find most similar document
similarities = [
    cosine_similarity(query_embedding, doc_emb) 
    for doc_emb in doc_embeddings
]

best_match_idx = similarities.index(max(similarities))
print(f"\nQuery: {query}")
print(f"Best match: {docs[best_match_idx]}")
print(f"Similarity: {similarities[best_match_idx]:.4f}")


# ============================================================================
# Example 4: Different embedding models
# ============================================================================

text = "Embedding models convert text into numerical vectors"

# Nomic Embed (768 dimensions, great for general use)
nomic_emb = client.embed(text, model="nomic-embed-text")
print(f"\nNomic: {len(nomic_emb)} dimensions")

# Snowflake Arctic (1024 dimensions, optimized for RAG)
try:
    arctic_emb = client.embed(text, model="snowflake-arctic")
    print(f"Arctic: {len(arctic_emb)} dimensions")
except Exception as e:
    print(f"Arctic: {e} (model may need to be downloaded)")


# ============================================================================
# Example 5: Retrieval Augmented Generation (RAG) Pattern
# ============================================================================

from oprel import embed, generate

# Step 1: Build knowledge base with embeddings
knowledge_base = [
    "Oprel SDK is a local-first AI runtime",
    "It supports llama.cpp for fast CPU/GPU inference",
    "Models are cached locally for instant reuse",
    "Zero configuration required - works out of the box",
]

kb_embeddings = embed(knowledge_base, model="nomic-embed-text")

# Step 2: User asks a question
question = "How does Oprel handle model caching?"
question_embedding = embed(question, model="nomic-embed-text")

# Step 3: Find most relevant context
similarities = [
    sum(a * b for a, b in zip(question_embedding, kb_emb))
    for kb_emb in kb_embeddings
]
best_idx = similarities.index(max(similarities))
context = knowledge_base[best_idx]

# Step 4: Generate answer using retrieved context
prompt = f"""Context: {context}

Question: {question}

Answer:"""

answer = generate("qwen3-1.6b", prompt)
print(f"\nQuestion: {question}")
print(f"Retrieved Context: {context}")
print(f"Answer: {answer}")


# ============================================================================
# Example 6: CLI usage
# ============================================================================

"""
# Single text embedding
oprel embed nomic-embed-text "Hello world"

# Embedding with JSON output
oprel embed nomic-embed-text "Hello world" --format json

# Process files (PDF, DOCX, TXT, JSON) - outputs to embeddings.json
oprel embed nomic-embed-text --files document.pdf data.json notes.txt

# Custom output file for files
oprel embed nomic-embed-text --files report.pdf --output my_embeddings.json

# Batch processing from text file (one text per line)
cat > texts.txt << EOF
First document
Second document
Third document
EOF

oprel embed nomic-embed-text --batch texts.txt --output embeddings.json

# Different output formats
oprel embed nomic-embed-text --batch texts.txt --format jsonl > embeddings.jsonl
oprel embed nomic-embed-text --batch texts.txt --format json | jq '.embeddings[0] | length'

# Save without text content (just vectors)
oprel embed nomic-embed-text --batch texts.txt --output vectors.json --no-texts
"""
