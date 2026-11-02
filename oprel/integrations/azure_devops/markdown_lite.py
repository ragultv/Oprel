"""
Tiny Markdown -> HTML converter for Azure DevOps rich-text fields.

Models write Markdown; ADO's Description/Comment fields are HTML. All input is
HTML-escaped first, so model output can never inject markup — only the
constructs below are turned into tags.

Supported: headings, paragraphs, bullet/numbered lists, task-list checkboxes,
fenced code blocks, inline code, bold, italic, and http(s) links.
"""

from __future__ import annotations

import html
import re

_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*|__(.+?)__")
_ITALIC_RE = re.compile(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])|(?<![\w_])_(?!\s)(.+?)(?<!\s)_(?![\w_])")
_CODE_RE = re.compile(r"`([^`]+)`")
_UL_RE = re.compile(r"^\s*[-*+]\s+(.*)$")
_OL_RE = re.compile(r"^\s*\d+[.)]\s+(.*)$")
_H_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_TASK_RE = re.compile(r"^\[( |x|X)\]\s+(.*)$")


def _inline(text: str) -> str:
    # Pull out code spans first so their contents are not formatted.
    codes: list[str] = []

    def stash(m: re.Match[str]) -> str:
        codes.append(m.group(1))
        return f"\x00{len(codes) - 1}\x00"

    text = _CODE_RE.sub(stash, html.escape(text, quote=False))
    # The URL was already escaped with the rest of the text; re-escape for an attribute context.
    text = _LINK_RE.sub(
        lambda m: f'<a href="{html.escape(html.unescape(m.group(2)), quote=True)}">{m.group(1)}</a>',
        text,
    )
    text = _BOLD_RE.sub(lambda m: f"<strong>{m.group(1) or m.group(2)}</strong>", text)
    text = _ITALIC_RE.sub(lambda m: f"<em>{m.group(1) or m.group(2)}</em>", text)
    return re.sub(r"\x00(\d+)\x00", lambda m: f"<code>{codes[int(m.group(1))]}</code>", text)


def _list_item(content: str) -> str:
    task = _TASK_RE.match(content)
    if task:
        box = "&#9745;" if task.group(1).lower() == "x" else "&#9744;"
        return f"<li>{box} {_inline(task.group(2))}</li>"
    return f"<li>{_inline(content)}</li>"


def markdown_to_html(markdown: str) -> str:
    if not markdown:
        return ""
    lines = markdown.replace("\r\n", "\n").split("\n")
    out: list[str] = []
    paragraph: list[str] = []
    list_tag: str | None = None
    in_code = False
    code_lines: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            out.append("<p>" + "<br>".join(_inline(p) for p in paragraph) + "</p>")
            paragraph.clear()

    def close_list() -> None:
        nonlocal list_tag
        if list_tag:
            out.append(f"</{list_tag}>")
            list_tag = None

    for line in lines:
        if line.strip().startswith("```"):
            if in_code:
                out.append("<pre><code>" + html.escape("\n".join(code_lines), quote=False) + "</code></pre>")
                code_lines.clear()
                in_code = False
            else:
                flush_paragraph()
                close_list()
                in_code = True
            continue
        if in_code:
            code_lines.append(line)
            continue

        if not line.strip():
            flush_paragraph()
            close_list()
            continue

        heading = _H_RE.match(line)
        ul = _UL_RE.match(line)
        ol = _OL_RE.match(line)

        if heading:
            flush_paragraph()
            close_list()
            level = len(heading.group(1))
            out.append(f"<h{level}>{_inline(heading.group(2))}</h{level}>")
        elif ul or ol:
            flush_paragraph()
            tag = "ul" if ul else "ol"
            if list_tag != tag:
                close_list()
                out.append(f"<{tag}>")
                list_tag = tag
            out.append(_list_item((ul or ol).group(1)))
        else:
            close_list()
            paragraph.append(line.strip())

    if in_code:
        out.append("<pre><code>" + html.escape("\n".join(code_lines), quote=False) + "</code></pre>")
    flush_paragraph()
    close_list()
    return "".join(out)


_TAG_RE = re.compile(r"<[^>]+>")
_BLOCK_END_RE = re.compile(r"(?i)</(p|div|li|h[1-6]|tr)>|<br\s*/?>")


def html_to_text(value: str | None, limit: int = 2000) -> str:
    """Flatten ADO HTML to plain text for the model."""
    if not value:
        return ""
    text = _BLOCK_END_RE.sub("\n", value)
    text = html.unescape(_TAG_RE.sub("", text))
    text = re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", text)).strip()
    if len(text) > limit:
        text = text[:limit].rstrip() + " …[truncated]"
    return text
