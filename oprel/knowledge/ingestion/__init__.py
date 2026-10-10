"""
Ingestion subsystem for Oprel Knowledge & RAG.
"""

from oprel.knowledge.ingestion.parser_registry import ParserRegistry, DefaultFileParser
from oprel.knowledge.ingestion.chunker import SemanticChunker
from oprel.knowledge.ingestion.jobs import JobManager
from oprel.knowledge.ingestion.pipeline import IngestionPipeline

__all__ = [
    "ParserRegistry",
    "DefaultFileParser",
    "SemanticChunker",
    "JobManager",
    "IngestionPipeline",
]
