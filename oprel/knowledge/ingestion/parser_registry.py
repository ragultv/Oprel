"""
Parser registry and file format extractors for Oprel Knowledge Ingestion.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from oprel.knowledge.interfaces import DocumentParser
from oprel.utils.file_parser import extract_text

logger = logging.getLogger("oprel.knowledge.ingestion.parser")


class DefaultFileParser(DocumentParser):
    """
    General file parser wrapping Oprel's unified file extractor
    supporting PDF, DOCX, Markdown, plain text, and source code.
    """

    SUPPORTED_EXTENSIONS = {
        ".txt", ".md", ".pdf", ".docx",
        ".py", ".js", ".ts", ".tsx", ".jsx",
        ".html", ".css", ".json", ".yaml", ".yml",
        ".csv", ".log"
    }

    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    async def parse(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        text = extract_text(file_path)
        stat = file_path.stat()

        metadata: Dict[str, Any] = {
            "filename": file_path.name,
            "extension": file_path.suffix.lower(),
            "size_bytes": stat.st_size,
            "char_count": len(text),
            "line_count": text.count("\n") + 1 if text else 0,
        }

        # Extract title from first markdown heading or filename
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if lines and lines[0].startswith("# "):
            metadata["title"] = lines[0][2:].strip()
        else:
            metadata["title"] = file_path.stem

        return text, metadata


class ParserRegistry:
    """
    Registry for format-specific parsers with fallback to DefaultFileParser.
    """

    def __init__(self):
        self._parsers: List[DocumentParser] = []
        self._default_parser = DefaultFileParser()

    def register(self, parser: DocumentParser) -> None:
        self._parsers.insert(0, parser)

    def get_parser(self, file_path: Path) -> DocumentParser:
        for parser in self._parsers:
            if parser.can_parse(file_path):
                return parser
        return self._default_parser
