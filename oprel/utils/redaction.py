"""
Secret redaction for logs, error messages and tool output.

Two layers:
1. Exact-value redaction: every secret loaded through SecretStore is registered
   here, so the literal value (and its Basic-auth encoding) never survives.
2. Pattern redaction: catches credential-shaped strings we were never told about
   (Authorization headers, "api_key": "...", etc.).
"""

from __future__ import annotations

import base64
import logging
import re
import threading

REDACTED = "[REDACTED]"

_MIN_SECRET_LEN = 8
_lock = threading.Lock()
_known_secrets: set[str] = set()

_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Authorization: Basic xxx / Bearer xxx (headers, curl dumps, repr of dicts)
    (
        re.compile(r"(?i)(authorization['\"]?\s*[:=]\s*['\"]?(?:basic|bearer|token)\s+)[A-Za-z0-9+/=._~\-]+"),
        r"\1" + REDACTED,
    ),
    # key/value pairs whose key looks like a credential
    (
        re.compile(
            r"(?i)(['\"]?\b(?:pat|personal_access_token|access_token|refresh_token|api[_-]?key|"
            r"secret|client_secret|password|x-oprel-token)\b['\"]?\s*[:=]\s*)(['\"]?)[^'\"\s,}&]+\2"
        ),
        r"\1\2" + REDACTED + r"\2",
    ),
]


def _variants(secret: str) -> set[str]:
    """The secret plus the encodings it is commonly transmitted in."""
    out = {secret}
    for prefix in ("", ":"):
        out.add(base64.b64encode(f"{prefix}{secret}".encode()).decode())
    return out


def register_secret(value: str | None) -> None:
    """Make sure `value` is scrubbed from anything passed through redact()."""
    if not value or len(value) < _MIN_SECRET_LEN:
        return
    with _lock:
        _known_secrets.update(_variants(value))


def unregister_secret(value: str | None) -> None:
    if not value:
        return
    with _lock:
        _known_secrets.difference_update(_variants(value))


def redact(text: object) -> str:
    """Return `text` as a string with every known or credential-shaped secret removed."""
    s = text if isinstance(text, str) else str(text)
    if not s:
        return s
    with _lock:
        secrets = sorted(_known_secrets, key=len, reverse=True)
    for secret in secrets:
        if secret in s:
            s = s.replace(secret, REDACTED)
    for pattern, repl in _PATTERNS:
        s = pattern.sub(repl, s)
    return s


class RedactingFilter(logging.Filter):
    """Logging filter that scrubs the formatted message and any traceback text."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
            cleaned = redact(message)
            if cleaned != message:
                record.msg = cleaned
                record.args = None
            if record.exc_info and not record.exc_text:
                record.exc_text = logging.Formatter().formatException(record.exc_info)
            if record.exc_text:
                record.exc_text = redact(record.exc_text)
        except Exception:
            pass  # never let redaction break logging
        return True


_FILTER = RedactingFilter()


def attach_filter(handler: logging.Handler) -> None:
    if _FILTER not in handler.filters:
        handler.addFilter(_FILTER)


def install_log_redaction() -> None:
    """Attach the redacting filter to every handler that exists now, and quiet chatty HTTP libs."""
    loggers = [logging.getLogger()] + [
        logging.getLogger(name) for name in list(logging.Logger.manager.loggerDict)
    ]
    for lg in loggers:
        for handler in getattr(lg, "handlers", []):
            attach_filter(handler)
    # httpx/httpcore log full request lines (and headers at DEBUG) — keep them quiet.
    for name in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)
