"""
Tidy the planner's text before the tool step draws it.

The plan is written by a small local model and drawn "exactly", so its slips become geometry.
Seen live with Qwen3-8B (casting: sprue well, sprue, pouring cup, "make everything one solid"):

    6. pouring_cup | cone | radius 0.015 height 0.1 | on top of sprue at its front end
    7. pouring_cup | cone | radius_bottom 0.015 radius_top 0.05 height 0.1 | on top of pouring_cup at its front end
    8. merge everything into one solid

Line 7 corrects line 6 (a pointed cone) but was drawn as a second cup standing on the first one's
tip; touching at a point, the two never fused, so "make everything one solid" kept failing. And the
merge, numbered instead of "after:", ran in the same call and merged only the new parts.
"""

from __future__ import annotations

import re

_LINE = re.compile(r"^\s*(\d+)\s*[.)]\s*(.*)$")
_AFTER = re.compile(r"^\s*after\s*:", re.IGNORECASE)
_STEP_WORDS = re.compile(r"^\s*(merge|fuse|combine|weld|make (everything|it all|all)|delete|remove|move)\b",
                         re.IGNORECASE)
# "at the front end of runner" on a box has no open end to join: it means flush against that end.
_AT_END_OF = re.compile(r"^\s*at the (left|right|front|back) end of\s+(\S+)(.*)$", re.IGNORECASE)
_BESIDE = {"left": "left of", "right": "right of", "front": "in front of", "back": "behind"}
_PARAM = re.compile(r"([A-Za-z_]+)\s+(\S+)")


def _params(text: str) -> dict[str, str]:
    return {k: v for k, v in _PARAM.findall(text)}


def _mentions(where: str, oid: str) -> bool:
    return re.search(rf"(?<![\w]){re.escape(oid)}(?![\w])", where, re.IGNORECASE) is not None


def clean_plan(plan: str) -> str:
    """
    - an object listed twice under one id is one object: the later line's shape and params win (it is
      the correction), its "where" too unless that names the object itself ("on top of pouring_cup");
    - numbered steps on what is drawn ("8. merge everything") become "after:" lines at the end;
    - "at the front end of X" (beside an object, not a pipe's open end) becomes "in front of X".
    Lines it does not understand are kept as they are.
    """
    # an object: [id, shape, params, where, *rest]; a numbered line it can't read: str; other text: (str,)
    rows: list[list[str] | str | tuple[str]] = []
    by_id: dict[str, list[str]] = {}
    after: list[str] = []
    for raw in plan.splitlines():
        m = _LINE.match(raw)
        if _AFTER.match(raw):
            after.append(raw.strip())
            continue
        if not m:
            if raw.strip():
                rows.append((raw.rstrip(),))
            continue
        body = m.group(2).strip()
        cells = [c.strip() for c in body.split("|")]
        if len(cells) < 4:
            if _STEP_WORDS.match(body):
                after.append(f"after: {body}")
            else:
                rows.append(body)
            continue
        oid, shape, params, where, *rest = cells
        end = _AT_END_OF.match(where)
        if end:
            where = f"{_BESIDE[end.group(1).lower()]} {end.group(2)}{end.group(3)}"
        key = oid.lower()
        if key in by_id and not key.startswith("hole in"):
            prev = by_id[key]
            merged = {**_params(prev[2]), **_params(params)}
            prev[1] = shape or prev[1]
            prev[2] = " ".join(f"{k} {v}" for k, v in merged.items())
            if where and not _mentions(where, oid):
                prev[3] = where
            continue
        row = [oid, shape, params, where, *rest]
        by_id[key] = row
        rows.append(row)

    out, n = [], 0
    for row in rows:
        if isinstance(row, tuple):
            out.append(row[0])
            continue
        n += 1
        out.append(f"{n}. " + (" | ".join(row).replace("|  |", "| |") if isinstance(row, list) else row))
    return "\n".join(out + after)
