"""
Dynamic token-aware context builder with citation formatting and budget enforcement.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from oprel.knowledge.interfaces import ContextBuilder
from oprel.knowledge.models import (
    ContextBudget,
    Evidence,
    GroundedPrompt,
)

logger = logging.getLogger("oprel.knowledge.orchestration.context_builder")


class DynamicContextBuilder(ContextBuilder):
    """
    Constructs prompt context with source-attributed evidence, strictly enforcing
    model context window token budgets and reserving headroom for LLM response generation.
    """

    def build_grounded_prompt(
        self,
        query: str,
        evidence: List[Evidence],
        budget: Optional[ContextBudget] = None,
        system_instruction: Optional[str] = None,
        memory_context: Optional[str] = None,
    ) -> GroundedPrompt:
        ctx_budget = budget or ContextBudget()
        max_chars = ctx_budget.max_rag_chars

        injected_evidence: List[Evidence] = []
        context_parts: List[str] = []
        current_chars = 0
        truncated = False

        # 1. Format document evidence
        for idx, item in enumerate(evidence):
            heading = item.metadata.get("heading")
            header_str = f"Source [{idx + 1}] ({item.filename}" + (f", {heading}" if heading else "") + "):"
            chunk_text = item.text.strip()

            item_block = f"{header_str}\n{chunk_text}"
            block_len = len(item_block) + 2  # account for newlines

            if current_chars + block_len > max_chars:
                remaining_chars = max_chars - current_chars
                if remaining_chars > 120:
                    truncated_text = chunk_text[:remaining_chars - len(header_str) - 10] + "..."
                    context_parts.append(f"{header_str}\n{truncated_text}")
                    injected_evidence.append(item)
                truncated = True
                break

            context_parts.append(item_block)
            injected_evidence.append(item)
            current_chars += block_len

        evidence_block = "\n\n".join(context_parts)

        # 2. Assemble prompt sections
        prompt_lines: List[str] = []

        if memory_context:
            prompt_lines.append(f"--- RELEVANT CONVERSATION MEMORY (SUPERMEMORY) ---\n{memory_context}\n")

        if evidence_block:
            prompt_lines.append("CONTEXT FROM LOCAL KNOWLEDGE BASE:")
            prompt_lines.append("----------------------------------------")
            prompt_lines.append(evidence_block)
            prompt_lines.append("----------------------------------------\n")

        prompt_lines.append(f"QUESTION: {query}\n")

        default_instruction = (
            "INSTRUCTION: Use the provided context above to answer the question accurately. "
            "Cite source labels [1], [2], etc. when referencing specific facts from the sources. "
            "If the answer cannot be determined from the context, state that clearly."
        )
        prompt_lines.append(system_instruction or default_instruction)

        full_prompt = "\n".join(prompt_lines)
        estimated_tokens = int(len(full_prompt) / ctx_budget.chars_per_token)

        return GroundedPrompt(
            prompt_text=full_prompt,
            evidence=injected_evidence,
            total_chunks_injected=len(injected_evidence),
            estimated_context_tokens=estimated_tokens,
            truncated=truncated,
        )
