# knowledge/config_knowledge.py
 
import os
from pathlib import Path

# Base paths
OPREL_HOME = Path(os.environ.get("OPREL_HOME", Path.home() / ".cache" / "oprel"))
KNOWLEDGE_DIR = OPREL_HOME / "knowledge"
 
# ── Oprel connection ──────────────────────────────────
OPREL_BASE_URL    = os.getenv("OPREL_BASE_URL", "http://localhost:11435")
OPREL_LLM_MODEL   = os.getenv("OPREL_LLM_MODEL",   "llama3.2-1b") # Updated to match user's preferred model
OPREL_EMBED_MODEL = os.getenv("OPREL_EMBED_MODEL",  "nomic-embed-text")
 
# ── Vector DB ─────────────────────────────────────────
CHROMA_PERSIST_DIR  = os.getenv("CHROMA_DIR", str(KNOWLEDGE_DIR / "chroma_db"))
CHROMA_SNAPSHOT_DIR = os.getenv("CHROMA_SNAP", str(KNOWLEDGE_DIR / "chroma_snapshots"))
COLLECTION_NAME     = "oprel_knowledge"
 
# ── Chunking ──────────────────────────────────────────
CHUNK_SIZE          = 400   # characters per chunk
CHUNK_OVERLAP       = 80    # overlap between chunks
MIN_CHUNK_LEN       = 60    # discard chunks shorter than this
 
# ── Retrieval ─────────────────────────────────────────
TOP_K               = 3     # chunks to inject per query
SCORE_THRESHOLD     = 0.25  # below this = fallback mode
BM25_WEIGHT         = 0.3   # RRF weight for keyword results
VECTOR_WEIGHT       = 0.7   # RRF weight for semantic results
 
# ── Sync schedule ─────────────────────────────────────
SYNC_INTERVAL_HOURS = 24
MAX_RETRY_ATTEMPTS  = 5
RETRY_DELAYS_SEC    = [60, 300, 1800, 21600, 86400]
 
# ── Sources ───────────────────────────────────────────
RSS_FEEDS = [
    "https://feeds.bbci.co.uk/news/rss.xml",
    "https://feeds.arstechnica.com/arstechnica/index",
    "https://hnrss.org/frontpage",
]
 
WIKIPEDIA_TOPICS = [
    "Artificial intelligence",
    "Large language model",
    "Cybersecurity",
    "Python (programming language)",
]
 
LOCAL_WATCH_DIR = os.getenv("LOCAL_WATCH_DIR", str(KNOWLEDGE_DIR / "my_docs"))

# Ensure directories exist
os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
os.makedirs(CHROMA_SNAPSHOT_DIR, exist_ok=True)
os.makedirs(LOCAL_WATCH_DIR, exist_ok=True)
