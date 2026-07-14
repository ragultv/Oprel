"""
skill_matcher.py

Analyses a user's prompt and returns the best matching enabled skill
from the database, or None if no skill is a confident match.

Matching is intentionally done WITHOUT a second LLM call to keep latency low.
It uses a two-stage approach:
  Stage 1 — Fast keyword pre-filter (zero latency)
  Stage 2 — Semantic scoring using token overlap + category heuristics

A skill only matches if its confidence score exceeds MATCH_THRESHOLD.
"""

from __future__ import annotations

import re
from typing import Optional

from oprel.server import db

# Tune this. Lower = more aggressive matching. Raise to reduce false positives.
# Set to 0.30 because default skills have no keywords yet (Component 1 always 0).
# Once skills have keywords populated, raise this back to 0.40+ for precision.
MATCH_THRESHOLD = 0.30

# Maps common user intent signals to skill categories.
# This is the primary fast-path for routing.
CATEGORY_SIGNALS: dict[str, list[str]] = {
    # ── Coding / Development ─────────────────────────────────────────────────
    "coding": [
        "code", "write", "function", "class", "bug", "fix", "debug", "error",
        "implement", "refactor", "script", "python", "javascript", "typescript",
        "java", "cpp", "rust", "sql", "api", "program", "algorithm", "test",
        "unit test", "optimize", "performance", "snippet", "module", "library",
    ],
    "development": [
        "code", "write", "function", "class", "bug", "fix", "debug", "error",
        "implement", "refactor", "script", "python", "javascript", "typescript",
        "java", "cpp", "rust", "sql", "api", "program", "algorithm", "test",
        "unit test", "optimize", "performance", "snippet", "module", "library",
        "diagram", "mermaid", "html", "css", "devops", "docker", "kubernetes",
        "cloud", "deploy", "pipeline", "ci", "cd", "infrastructure",
    ],
    # ── Writing ───────────────────────────────────────────────────────────────
    "writing": [
        "write", "essay", "blog", "article", "draft", "story", "poem",
        "paragraph", "summarize", "summary", "rewrite", "proofread", "edit",
        "letter", "email", "cover letter", "report", "content", "copywriting",
        "translate", "translation",
    ],
    # Alias for DB rows that store category as "Writing" (capitalised)
    "writer": [
        "write", "essay", "blog", "article", "draft", "story", "poem",
        "paragraph", "summarize", "summary", "rewrite", "proofread", "edit",
        "letter", "email", "cover letter", "report", "content", "copywriting",
        "translate", "translation",
    ],
    # ── Analysis ─────────────────────────────────────────────────────────────
    "analysis": [
        "analyse", "analyze", "explain", "compare", "evaluate", "review",
        "assess", "what is", "how does", "why does", "difference between",
        "pros and cons", "breakdown", "insight", "understand",
    ],
    # ── Research ─────────────────────────────────────────────────────────────
    "research": [
        "research", "find", "look up", "information about", "details on",
        "history of", "what happened", "who is", "when did", "where is",
        "analyse", "analyze", "compare", "data",
    ],
    # ── Math ─────────────────────────────────────────────────────────────────
    "math": [
        "calculate", "solve", "equation", "math", "formula", "compute",
        "integral", "derivative", "statistics", "probability",
    ],
    # ── Translation ──────────────────────────────────────────────────────────
    "translation": [
        "translate", "in french", "in spanish", "in german", "in japanese",
        "in chinese", "in arabic", "in hindi", "language",
    ],
    # ── Documents ────────────────────────────────────────────────────────────
    "documents": [
        "pdf", "document", "file", "extract", "presentation", "slides",
        "summarize", "analyze",
    ],
    # ── Media ────────────────────────────────────────────────────────────────
    "media": [
        "image", "picture", "photo", "generate image", "create image",
        "draw", "illustration", "visual",
    ],
}
# Lowercase-alias map: many DB skills store category with mixed case.
# Build aliases automatically so "Writing" maps to "writing", etc.
_CATEGORY_SIGNAL_ALIASES: dict[str, list[str]] = {
    k.lower(): v for k, v in CATEGORY_SIGNALS.items()
}


def _tokenize(text: str) -> set[str]:
    """Lowercase, remove punctuation, split into word tokens."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return set(text.split())


def _score_skill(prompt_tokens: set[str], prompt_lower: str, skill: dict) -> float:
    """
    Returns a confidence score between 0.0 and 1.0 for how well the skill
    matches the user's prompt.

    Scoring components:
      - Keyword overlap between prompt and skill keywords field        (weight 0.5)
      - Keyword overlap between prompt and skill name/description      (weight 0.3)
      - Category signal match (checks CATEGORY_SIGNALS for category)  (weight 0.3)
    """
    score = 0.0

    # --- Component 1: Match against skill's explicit keywords ---
    skill_keywords_raw = skill.get("keywords", "") or ""
    if skill_keywords_raw.strip():
        skill_kw_tokens = {
            kw.strip().lower()
            for kw in skill_keywords_raw.split(",")
            if kw.strip()
        }
        overlap = prompt_tokens & skill_kw_tokens
        if skill_kw_tokens:
            score += 0.5 * (len(overlap) / len(skill_kw_tokens))

    # --- Component 2: Match against skill name + description tokens ---
    skill_text = f"{skill.get('name', '')} {skill.get('description', '')}"
    skill_text_tokens = _tokenize(skill_text)
    # Remove very short/common tokens (articles, prepositions, etc.)
    skill_text_tokens = {t for t in skill_text_tokens if len(t) > 2}
    if skill_text_tokens:
        overlap2 = prompt_tokens & skill_text_tokens
        score += 0.3 * (len(overlap2) / len(skill_text_tokens))

    # --- Component 3: Category signal match ---
    # Weight raised to 0.30 so a single strong intent word (essay, translate,
    # debug, calculate) alone reaches the MATCH_THRESHOLD of 0.30.
    category = (skill.get("category") or "").lower()
    # Look up in the combined alias map (handles mixed-case DB values like "Writing")
    category_signals = _CATEGORY_SIGNAL_ALIASES.get(category, [])
    for signal in category_signals:
        if signal in prompt_lower:
            score += 0.30
            break  # Only count once per category

    return min(score, 1.0)  # Cap at 1.0


def _is_overridable_system_prompt(system_prompt: str | None) -> bool:
    """
    Returns True if the system_prompt is safe for skill matching to override.

    We allow override when:
      - system_prompt is None or empty (no instruction set)
      - system_prompt exactly matches the user's global system_instruction
        stored in user_settings (it's just the generic fallback, not a
        functional prompt like a title generator or explicit skill selection)

    We DO NOT override when the system_prompt is something specific and
    functional (e.g. 'You are a concise title generator...').
    """
    if not system_prompt or not system_prompt.strip():
        return True

    try:
        settings = db.get_user_settings()
        if settings:
            global_instruction = (settings.get("system_instruction") or "").strip()
            if global_instruction and system_prompt.strip() == global_instruction:
                return True  # It's just the user's generic global setting — safe to override
    except Exception:
        pass

    return False  # Unknown / functional system_prompt — preserve it


def match_skill_for_prompt(prompt: str) -> Optional[dict]:
    """
    Given a user prompt string, returns the best matching enabled skill dict
    from the database, or None if no skill exceeds MATCH_THRESHOLD.

    Returns a dict with at least: system_prompt, temperature, max_tokens,
    name, id, category, keywords.

    This function is synchronous and safe to call from within generate_text().
    It does NOT make any LLM calls.
    """
    if not prompt or not isinstance(prompt, str):
        return None

    # Fetch all skills from the DB
    try:
        skills = db.list_skills()
    except Exception:
        return None

    if not skills:
        return None

    # Only keep enabled skills
    enabled_skills = [s for s in skills if s.get("enabled", True)]
    if not enabled_skills:
        return None

    prompt_lower = prompt.lower().strip()
    prompt_tokens = _tokenize(prompt_lower)

    best_skill: Optional[dict] = None
    best_score = 0.0

    for skill in enabled_skills:
        score = _score_skill(prompt_tokens, prompt_lower, skill)
        if score > best_score:
            best_score = score
            best_skill = skill

    if best_score >= MATCH_THRESHOLD:
        return best_skill

    return None
