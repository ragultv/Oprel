from __future__ import annotations

from pathlib import Path
from typing import Any, BinaryIO

from oprel.server.services.context import CONFIG, logger


def ingest_text_or_file(text: str | None, file_path: str | None, metadata: dict[str, Any] | None) -> dict[str, Any]:
    from oprel.knowledge.sync_engine import SyncEngine

    engine = SyncEngine()

    if file_path:
        engine.ingest_file(Path(file_path))
        return {"success": True, "message": f"Ingested file: {file_path}"}
    if text:
        engine.ingest_text(text, metadata)
        return {"success": True, "message": "Ingested raw text"}

    raise ValueError("Either 'text' or 'file_path' must be provided")


async def search_knowledge(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    from oprel.knowledge.knowledge_store import KnowledgeStore

    async def internal_embed(text: str, model: str | None = None) -> list[float]:
        from oprel.downloader.aliases import resolve_model_id
        from oprel.server.services.generation import get_embeddings, EmbeddingParams

        res = await get_embeddings(EmbeddingParams(input=text, model=resolve_model_id(model or "nomic-embed-text")))
        return res.embedding or []

    store = KnowledgeStore(embed_func=internal_embed)
    return await store.search(query, top_k=top_k)


def reset_knowledge() -> dict[str, Any]:
    from oprel.knowledge.config import KNOWLEDGE_DIR
    import shutil

    if KNOWLEDGE_DIR.exists():
        shutil.rmtree(KNOWLEDGE_DIR)
        KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    return {"success": True, "message": "Knowledge store reset"}


async def upload_document(filename: str, file_obj: BinaryIO) -> dict[str, Any]:
    from oprel.knowledge.sync_engine import SyncEngine
    from oprel.knowledge.knowledge_store import KnowledgeStore
    import shutil

    knowledge_dir = CONFIG.cache_dir / "knowledge_files"
    knowledge_dir.mkdir(parents=True, exist_ok=True)

    file_path = knowledge_dir / filename

    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file_obj, buffer)

        async def internal_embed(text: str, model: str | None = None) -> list[float]:
            from oprel.downloader.aliases import resolve_model_id
            from oprel.server.services.generation import get_embeddings, EmbeddingParams

            res = await get_embeddings(EmbeddingParams(input=text, model=resolve_model_id(model or "nomic-embed-text")))
            return res.embedding or []

        store = KnowledgeStore(embed_func=internal_embed)
        engine = SyncEngine(store)

        await engine.ingest_file(file_path)

        return {
            "success": True,
            "filename": filename,
            "message": "Document indexed successfully",
        }
    except Exception as exc:
        logger.error(f"Failed to ingest file via API: {exc}")
        raise


async def chat_extract_file(filename: str, file_obj: BinaryIO, model_id: str | None = None, reply_reserve: int | None = None) -> dict[str, Any]:
    """Extract text from an uploaded file for chat-only ingestion and estimate token usage.

    Returns a dict with extracted text, character count, estimated tokens, context limits,
    and a suggested truncated excerpt that fits within the model's context (conservative).
    """
    import shutil
    from oprel.utils.file_parser import extract_text

    tmp_dir = CONFIG.cache_dir / "tmp_uploads"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    file_path = tmp_dir / filename

    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file_obj, buffer)

        # Extract text using existing utilities
        text = extract_text(file_path)

        if not isinstance(text, str):
            text = str(text or "")

        chars = len(text)

        # Try to use tiktoken for accurate token counting and truncation when available
        estimated_tokens = None
        suggested_text = text
        try:
            import tiktoken

            try:
                enc = tiktoken.encoding_for_model(model_id) if model_id else tiktoken.get_encoding('cl100k_base')
            except Exception:
                # Fallback to cl100k_base
                enc = tiktoken.get_encoding('cl100k_base')

            token_ids = enc.encode(text)
            estimated_tokens = len(token_ids)

            # Determine context limit (tokens) from model_id hints, fallback 4096
            ctx_limit = 4096
            if model_id:
                mid = model_id.lower()
                if '8192' in mid or '8k' in mid:
                    ctx_limit = 8192
                elif '16384' in mid or '16k' in mid or '16kb' in mid:
                    ctx_limit = 16384

            reply_reserve = reply_reserve or 1024
            available_tokens = max(0, ctx_limit - reply_reserve)

            if estimated_tokens > available_tokens and available_tokens > 50:
                # Truncate tokens: keep prefix+suffix split to preserve context
                keep = available_tokens
                half = keep // 2
                first_tokens = token_ids[:half]
                last_tokens = token_ids[-(keep - half):]
                # decode back to text and join with ellipsis
                first_text = enc.decode(first_tokens)
                last_text = enc.decode(last_tokens)
                suggested_text = first_text + "\n\n... (truncated) ...\n\n" + last_text
            else:
                suggested_text = text

        except Exception:
            # Fallback heuristic: 4 chars per token
            chars_per_token = 4
            estimated_tokens = max(1, chars // chars_per_token)
            ctx_limit = 4096
            if model_id:
                mid = model_id.lower()
                if "8192" in mid or "8k" in mid:
                    ctx_limit = 8192
                elif "16384" in mid or "16k" in mid or "16kb" in mid:
                    ctx_limit = 16384
            reply_reserve = reply_reserve or 1024
            available_tokens = max(0, ctx_limit - reply_reserve)
            available_chars = available_tokens * chars_per_token
            needs_truncation = chars > available_chars
            if needs_truncation and available_chars > 200:
                first_part = text[: max(0, available_chars // 2 - 4)]
                last_part = text[-max(0, available_chars // 2 - 4) :]
                suggested_text = first_part + "\n\n... (truncated) ...\n\n" + last_part
            elif needs_truncation:
                suggested_text = text[: max(0, available_chars - 4)] + " ..."

        return {
            "success": True,
            "filename": filename,
            "extracted_text": text,
            "chars": chars,
            "estimated_tokens": estimated_tokens,
            "context_limit_tokens": ctx_limit,
            "reply_reserve": reply_reserve,
            "available_context_tokens": available_tokens,
            "available_context_chars": available_chars,
            "needs_truncation": needs_truncation,
            "suggested_text": suggested_text,
        }
    except Exception as exc:
        logger.error(f"Failed to extract file for chat upload: {exc}")
        raise


def list_documents() -> list[dict[str, Any]]:
    from oprel.knowledge.knowledge_store import KnowledgeStore

    store = KnowledgeStore()

    docs_map: dict[str, dict[str, Any]] = {}
    for doc in store.bm25_docs:
        meta = doc.get("metadata", {})
        fname = meta.get("filename", "Unknown")
        if fname not in docs_map:
            docs_map[fname] = {
                "id": doc["id"],
                "filename": fname,
                "size_bytes": 0,
                "indexed_at": meta.get("timestamp", ""),
                "chunks": 0,
            }
        docs_map[fname]["chunks"] += 1

    return list(docs_map.values())


def delete_document(filename: str) -> dict[str, Any]:
    return {"success": False, "message": "Deletion via API not yet implemented"}
