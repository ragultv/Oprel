"""
Intelligent query routing between Document Knowledge, Conversational Supermemory, and Direct generation.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Optional

from oprel.knowledge.supermemory import expand_query_with_context


class QueryRouteType(str, Enum):
    DOCUMENT = "document"       # Prioritize user files and technical documents
    MEMORY = "memory"           # Prioritize historical conversation turns (Supermemory)
    HYBRID = "hybrid"           # Retrieve from both documents and conversation memory
    DIRECT = "direct"           # No retrieval required (general conversational greeting/math)


class QueryRouter:
    """
    Classifies query intent and determines the optimal retrieval source and query expansion.
    """

    MEMORY_TRIGGERS = [
        r"\bwhat did we (talk|discuss|say)\b",
        r"\bremember (when|what|how|my)\b",
        r"\bearlier (you|we|I) (mentioned|said|discussed)\b",
        r"\bour (previous|last) conversation\b",
        r"\bmy (name|preference|favorite|goal)\b",
    ]

    DOC_TRIGGERS = [
        r"\baccording to (the|my)? (doc|file|document|manual|guide|spec)\b",
        r"\bhow to\b",
        r"\bwhat is the architecture\b",
        r"\bin (the|my) code\b",
        r"\bdocumentation\b",
        r"\binstall(ation)?\b",
        r"\bconfiguration\b",
    ]

    def route(
        self,
        query: str,
        has_active_conversation: bool = False,
        force_rag: Optional[bool] = None,
    ) -> QueryRouteType:
        if force_rag is False:
            return QueryRouteType.DIRECT

        lower_q = query.lower()

        # Check memory triggers
        for pattern in self.MEMORY_TRIGGERS:
            if re.search(pattern, lower_q):
                return QueryRouteType.MEMORY if not force_rag else QueryRouteType.HYBRID

        # Check doc triggers
        for pattern in self.DOC_TRIGGERS:
            if re.search(pattern, lower_q):
                return QueryRouteType.DOCUMENT

        # If RAG is explicitly requested, default to document or hybrid
        if force_rag is True:
            return QueryRouteType.HYBRID if has_active_conversation else QueryRouteType.DOCUMENT

        return QueryRouteType.HYBRID if has_active_conversation else QueryRouteType.DOCUMENT

    def expand_query(self, query: str, context_hint: Optional[str] = None) -> str:
        """Enrich short or pronoun-heavy follow-up questions with recent context."""
        return expand_query_with_context(query, context_hint)
