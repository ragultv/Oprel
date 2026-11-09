"""
Semantic, recursive structure-aware chunker for Oprel Knowledge Ingestion.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional

from oprel.knowledge.interfaces import Chunker
from oprel.knowledge.models import Chunk


class SemanticChunker(Chunker):
    """
    Recursively splits documents respecting markdown structure, code blocks,
    paragraphs, and sentence boundaries with deterministic ID generation.
    """

    def __init__(
        self,
        chunk_size: int = 400,
        chunk_overlap: int = 80,
        min_chunk_len: int = 60,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_len = min_chunk_len

    def _split_text(self, text: str) -> List[str]:
        """Split text recursively into chunks not exceeding chunk_size."""
        if len(text) <= self.chunk_size:
            return [text]

        # Splitting hierarchy
        separators = ["\n## ", "\n### ", "\n\n", "\n", ". ", " "]

        for sep in separators:
            parts = text.split(sep)
            if len(parts) > 1:
                chunks: List[str] = []
                current = ""
                for part in parts:
                    candidate = f"{current}{sep}{part}" if current else part
                    if len(candidate) <= self.chunk_size:
                        current = candidate
                    else:
                        if current:
                            chunks.append(current)
                        current = part
                if current:
                    chunks.append(current)
                
                # Check if all chunks comply
                if all(len(c) <= self.chunk_size * 1.5 for c in chunks):
                    return chunks

        # Fallback to character slicing with overlap
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start += max(1, self.chunk_size - self.chunk_overlap)
        return chunks

    def chunk(
        self,
        text: str,
        document_id: str,
        version_id: str,
        base_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        if not text:
            return []

        base_meta = base_metadata or {}
        raw_chunks = self._split_text(text)
        result: List[Chunk] = []

        current_heading = base_meta.get("title", "")
        current_offset = 0

        for idx, raw_chunk in enumerate(raw_chunks):
            cleaned = raw_chunk.strip()
            if len(cleaned) < self.min_chunk_len:
                continue

            # Update heading if chunk contains a Markdown header
            header_match = re.search(r"^(#{1,3}\s+[^\n]+)", cleaned, re.MULTILINE)
            if header_match:
                current_heading = header_match.group(1).lstrip("#").strip()

            start_char = text.find(raw_chunk, current_offset)
            if start_char == -1:
                start_char = current_offset
            end_char = start_char + len(raw_chunk)
            current_offset = max(current_offset, start_char + 1)

            # Deterministic chunk ID
            content_hash = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:16]
            chunk_id = hashlib.sha256(
                f"{document_id}:{version_id}:{idx}:{content_hash}".encode("utf-8")
            ).hexdigest()

            metadata = base_meta.copy()
            metadata.update({
                "chunk_index": idx,
                "heading": current_heading,
                "content_hash": content_hash,
                "char_length": len(cleaned),
            })

            result.append(
                Chunk(
                    id=chunk_id,
                    document_id=document_id,
                    version_id=version_id,
                    chunk_index=idx,
                    text=cleaned,
                    start_char=start_char,
                    end_char=end_char,
                    metadata=metadata,
                    token_count=max(1, len(cleaned) // 4),
                )
            )

        return result
