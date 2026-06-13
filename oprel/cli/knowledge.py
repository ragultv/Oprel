import argparse
import sys
import logging
from pathlib import Path
from oprel.utils.logging import get_logger

logger = get_logger(__name__)

def cmd_index(args: argparse.Namespace) -> int:
    """Index files or text into the knowledge store"""
    from oprel.knowledge.sync_engine import SyncEngine
    
    engine = SyncEngine()
    path = Path(args.path)
    
    if not path.exists():
        print(f"Error: Path not found: {path}")
        return 1
        
    print(f"Indexing {path.name}...")
    import asyncio
    try:
        if path.is_dir():
            # Index all supported files in directory
            count = 0
            for file in path.rglob("*"):
                if file.suffix.lower() in [".txt", ".md", ".pdf"]:
                    asyncio.run(engine.ingest_file(file))
                    count += 1
            print(f"✓ Indexed {count} files from {path}")
        else:
            asyncio.run(engine.ingest_file(path))
            print(f"✓ Indexed {path}")
        return 0
    except Exception as e:
        print(f"❌ Indexing failed: {e}")
        return 1

def cmd_knowledge_search(args: argparse.Namespace) -> int:
    """Search the knowledge store"""
    from oprel.knowledge.knowledge_store import KnowledgeStore
    
    import asyncio
    store = KnowledgeStore()
    print(f"Searching knowledge for: '{args.query}'...\n")
    
    try:
        results = asyncio.run(store.search(args.query, top_k=args.top_k))
        
        if not results:
            print("No relevant results found.")
            return 0
            
        for i, res in enumerate(results):
            score = res.get('score', 0)
            source = res.get('metadata', {}).get('filename', 'Unknown')
            # Handle RRF scores which are small
            score_text = f"Score: {score:.4f}"
            
            print(f"[{i+1}] {source} ({score_text})")
            print("-" * 40)
            print(res['text'].strip())
            print("-" * 40)
            print()
            
        return 0
    except Exception as e:
        print(f"❌ Search failed: {e}")
        return 1

def cmd_knowledge_sync(args: argparse.Namespace) -> int:
    """Run a full sync of all configured knowledge sources"""
    from oprel.knowledge.sync_engine import SyncEngine
    import asyncio
    
    engine = SyncEngine()
    print("🔄 Running global knowledge sync...")
    try:
        asyncio.run(engine.sync_all())
        print("✅ Sync completed successfully.")
        return 0
    except Exception as e:
        print(f"❌ Sync failed: {e}")
        return 1
