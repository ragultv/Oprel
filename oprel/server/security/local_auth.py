"""
Local request authentication for privileged endpoints.

The app-wide LocalGuardMiddleware (guard.py) already blocks foreign Host headers and
foreign origins, and requires the per-install bearer token from non-browser clients.
It deliberately lets *any* loopback origin through, so a browser request from some other
local web app is treated like the Oprel UI.

Endpoints that act with the user's stored credentials (integrations, tool execution,
approvals) are stricter, via `require_local_token`:

1. Browser requests must come from the daemon's own origin or the Next.js dev server
   (plus OPREL_CORS_ORIGINS), not just any loopback origin.
2. A token is always required: the per-process `X-Oprel-Token` (the web UI gets it from
   GET /auth/session) or the per-install bearer token (CLI/SDK, see oprel.utils.auth).
   Custom headers force a CORS preflight, which untrusted origins fail.
3. When bound to a non-loopback interface, an explicit opt-in is needed
   (OPREL_ALLOW_REMOTE_INTEGRATIONS=1).
"""

from __future__ import annotations

import hmac
import ipaddress
import os
import secrets
import threading

from fastapi import HTTPException, Request

from oprel.core.config import Config
from oprel.utils.auth import get_api_token
from oprel.utils.redaction import register_secret

TOKEN_HEADER = "X-Oprel-Token"

DEFAULT_TRUSTED_ORIGINS = (
    # Next.js dev server for the web UI (see webui-react/services/api.ts)
    "http://localhost:3000",
    "http://127.0.0.1:3000",
)

_LOOPBACK_NAMES = {"localhost", "127.0.0.1", "::1"}

_lock = threading.Lock()
_token: str | None = None
_bind_host: str = "127.0.0.1"


# ---------------------------------------------------------------------------
# Bind host
# ---------------------------------------------------------------------------

def set_bind_host(host: str) -> None:
    global _bind_host
    _bind_host = host or "127.0.0.1"


def get_bind_host() -> str:
    return _bind_host


def _is_loopback(host: str) -> bool:
    host = (host or "").strip("[]").lower()
    if host in _LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def is_loopback_bind() -> bool:
    return _is_loopback(_bind_host)


def remote_integrations_allowed() -> bool:
    return os.environ.get("OPREL_ALLOW_REMOTE_INTEGRATIONS", "").strip() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Origins (strict check for privileged endpoints)
# ---------------------------------------------------------------------------

def _env_list(name: str) -> list[str]:
    return [v.strip().rstrip("/") for v in os.environ.get(name, "").split(",") if v.strip()]


def _strict_trusted_origins() -> set[str]:
    """Never includes '*': privileged endpoints ignore a wildcard setting."""
    return set(DEFAULT_TRUSTED_ORIGINS) | {o for o in _env_list("OPREL_CORS_ORIGINS") if o != "*"}


def _same_origin(origin: str, request: Request) -> bool:
    host = request.headers.get("host", "")
    return origin in (f"http://{host}", f"https://{host}")


def is_trusted_browser_request(request: Request) -> bool:
    """
    True for non-browser clients (no Origin, no cross-site fetch metadata) and for
    browser requests from the daemon's own origin or an explicitly trusted origin.
    """
    origin = (request.headers.get("origin") or "").rstrip("/")
    if origin:
        return _same_origin(origin, request) or origin in _strict_trusted_origins()
    return (request.headers.get("sec-fetch-site") or "none") in ("none", "same-origin")


# ---------------------------------------------------------------------------
# Token
# ---------------------------------------------------------------------------

def _token_file():
    return Config().cache_dir / "daemon.token"


def get_local_token() -> str:
    """Per-process token. Also written to an owner-only file so local CLI tools can use it."""
    global _token
    with _lock:
        if _token is None:
            _token = secrets.token_urlsafe(32)
            register_secret(_token)
            try:
                path = _token_file()
                path.parent.mkdir(parents=True, exist_ok=True)
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                with os.fdopen(fd, "w") as fh:
                    fh.write(_token)
                try:
                    os.chmod(path, 0o600)
                except OSError:
                    pass
            except OSError:
                pass  # the token still works for the web UI via /auth/session
        return _token


async def require_local_token(request: Request) -> None:
    """FastAPI dependency for endpoints that act with the user's stored credentials."""
    if not is_loopback_bind() and not remote_integrations_allowed():
        raise HTTPException(
            status_code=403,
            detail=(
                "Integrations are disabled while the server is bound to a non-loopback address. "
                "Set OPREL_ALLOW_REMOTE_INTEGRATIONS=1 to enable them (not recommended)."
            ),
        )
    if not is_trusted_browser_request(request):
        raise HTTPException(status_code=403, detail="Untrusted origin")
    if not (_session_token_ok(request) or _bearer_token_ok(request)):
        raise HTTPException(status_code=401, detail="Missing or invalid local API token")


def _session_token_ok(request: Request) -> bool:
    supplied = request.headers.get(TOKEN_HEADER, "")
    return bool(supplied) and hmac.compare_digest(supplied, get_local_token())


def _bearer_token_ok(request: Request) -> bool:
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        return False
    expected = get_api_token(create=False)
    return bool(expected) and hmac.compare_digest(auth[7:].strip(), expected)

