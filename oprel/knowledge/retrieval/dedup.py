"""
Evidence deduplication and near-duplicate filtering based on token Jaccard similarity.
"""

from __future__ import annotations

import re
from typing import List

from oprel.knowledge.models import Evidence


class ChunkDeduplicator:
    """
    Filters near-identical retrieved chunks to prevent redundant prompt context.
    """

    def __init__(self, jaccard_threshold: float = 0.85):
        self.jaccard_threshold = jaccard_threshold

    def _tokenize(self, text: str) -> set[str]:
        return set(re.findall(r"\w+", text.lower()))

    def deduplicate(self, evidence_list: List[Evidence]) -> List[Evidence]:
        if not evidence_list:
            return []

        deduped: List[Evidence] = []
        token_sets: List[set[str]] = []

        for item in evidence_list:
            item_tokens = self._tokenize(item.text)
            if not item_tokens:
                continue

            is_duplicate = False
            for existing_tokens in token_sets:
                intersection = len(item_tokens.intersection(existing_tokens))
                union = len(item_tokens.union(existing_tokens))
                similarity = intersection / union if union > 0 else 0.0

                if similarity >= self.jaccard_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                deduped.append(item)
                token_sets.append(item_tokens)

        return deduped
