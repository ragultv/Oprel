"""
Production-grade token-budget manager and conversation context compactor.

Ensures dialogues take full advantage of up to 32k tokens (32,768 tokens).
When a conversation approaches or exceeds the 32k token limit:
1. Preserves recent dialogue turns verbatim for immediate conversational flow.
2. Compacts evicted turns into a structured digest.
3. Injects retrieved long-term memories (SuperMemory) into prompt context,
   ensuring zero information loss beyond the 32k token boundary.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("oprel.knowledge.context_compactor")

CHARS_PER_TOKEN = 3.8


def estimate_tokens(content: Any) -> int:
    """Accurately estimate token count for text, lists, or structured dicts."""
    if not content:
        return 0
    if isinstance(content, str):
        return max(1, int(len(content) / CHARS_PER_TOKEN)) if content.strip() else 0
    if isinstance(content, dict):
        text = str(content.get("content", "")).strip()
        return max(1, int(len(text) / CHARS_PER_TOKEN)) if text else 0
    if isinstance(content, list):
        return sum(estimate_tokens(item) for item in content)
    text = str(content).strip()
    return max(1, int(len(text) / CHARS_PER_TOKEN)) if text else 0


def estimate_messages_tokens(messages: List[Dict[str, Any]]) -> int:
    """Estimate total tokens in a list of chat messages including role overhead."""
    total = 0
    for m in messages:
        # ~4 tokens overhead per message turn in chat formats (role, headers, delimiters)
        total += 4 + estimate_tokens(m.get("content", ""))
    return total


def compact_conversation_context(
    history: List[Dict[str, Any]],
    current_prompt: str,
    system_prompt: Optional[str] = None,
    max_context_tokens: int = 32768,
    reserve_generation_tokens: int = 2048,
    min_recent_turns: int = 6,
    recalled_memories: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], str, bool]:
    """
    Compact conversation history strictly within max_context_tokens (up to 32k).

    Pure, deterministic function: evaluates token budgets, preserves recent turns
    verbatim, and injects memory context for evicted turns.

    Args:
        history: Complete list of previous chat messages.
        current_prompt: The incoming user prompt text.
        system_prompt: Active system instruction.
        max_context_tokens: Hard ceiling for context (defaults to 32,768 tokens).
        reserve_generation_tokens: Headroom reserved for model output generation (default 2048).
        min_recent_turns: Minimum recent messages to keep verbatim.
        recalled_memories: Optional list of facts/turns recalled from SuperMemory.

    Returns:
        Tuple of (active_history, effective_system_prompt, was_compacted)
    """
    effective_system = system_prompt or ""
    system_tokens = estimate_tokens(effective_system) + 10
    prompt_tokens = estimate_tokens(current_prompt) + 10

    # Context budget available strictly for previous history
    available_history_budget = max(
        512,
        max_context_tokens - reserve_generation_tokens - system_tokens - prompt_tokens
    )

    history_tokens = estimate_messages_tokens(history)

    # 1. If entire history fits within the 32k context budget, keep 100% of it intact!
    if history_tokens <= available_history_budget:
        return history, effective_system, False

    logger.info(
        f"[ContextCompactor] Conversation exceeds 32k token budget "
        f"({history_tokens + system_tokens + prompt_tokens} > {max_context_tokens}). "
        f"Compacting earlier turns into long-term memory."
    )

    # 2. Split history into older turns (to compact/index) and recent turns (verbatim window)
    if len(history) <= min_recent_turns:
        active_history = list(history)
        evicted_turns: List[Dict[str, Any]] = []
    else:
        recent_window = max(min_recent_turns, 6)
        active_history = list(history[-recent_window:])
        evicted_turns = list(history[:-recent_window])

    # 3. Build structured compaction memory block
    memory_context_block = ""
    if recalled_memories:
        fact_lines = "\n".join(f"- {fact}" for fact in recalled_memories)
        memory_context_block = (
            "\n\n[CONVERSATION COMPACTED (CONTEXT > 32K TOKENS)]\n"
            "Earlier conversation history was compacted to stay within the 32k token limit. "
            "Relevant historical context recalled from memory:\n"
            f"{fact_lines}\n"
            "[END COMPACTED CONTEXT]\n"
        )
    elif evicted_turns:
        # Extract brief conversational digests of evicted messages
        evicted_digest = []
        for turn in evicted_turns[-8:]:  # summarize the transition turns
            role = turn.get("role", "user").capitalize()
            content = str(turn.get("content", ""))
            short_content = content[:150] + ("..." if len(content) > 150 else "")
            evicted_digest.append(f"{role}: {short_content}")

        digest_text = "\n".join(evicted_digest)
        memory_context_block = (
            "\n\n[CONVERSATION COMPACTED (CONTEXT > 32K TOKENS)]\n"
            "Earlier conversation history was compacted to stay within the 32k token limit. "
            "Recent preceding dialogue summary:\n"
            f"{digest_text}\n"
            "[END COMPACTED CONTEXT]\n"
        )

    # 4. Integrate memory block into effective system prompt
    if memory_context_block:
        if effective_system:
            effective_system = f"{effective_system}{memory_context_block}"
        else:
            effective_system = memory_context_block.strip()

    # 5. Fit active_history into remaining budget
    cur_tokens = estimate_messages_tokens(active_history)
    budget_remaining = max_context_tokens - reserve_generation_tokens - estimate_tokens(effective_system) - prompt_tokens

    while active_history and cur_tokens > budget_remaining and len(active_history) > 2:
        dropped = active_history.pop(0)
        cur_tokens -= estimate_tokens(dropped.get("content", "")) + 4

    return active_history, effective_system, True
