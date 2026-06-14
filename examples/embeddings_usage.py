"""
Embeddings Usage Example for Oprel SDK.

Demonstrates how to generate vector embeddings for semantic search or clustering.
"""

from oprel.client import OprelClient

def main():
    client = OprelClient()
    
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast auburn fox leaps over a sleepy hound."
    ]
    
    for text in texts:
        emb = client.embed(model="nomic-embed-text", input=text)
        print(f"Text: {text}\nEmbedding dimension: {len(emb['embedding'])}\n")

if __name__ == "__main__":
    main()
