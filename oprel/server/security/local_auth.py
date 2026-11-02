"""
Local request authentication for privileged endpoints.

Threat model: the daemon listens on localhost, and any web page the user visits can
fire requests at it. CORS stops a page from *reading* responses, but not from
*sending* simple requests. So endpoints that can act on the user's behalf
(integrations, tool execution, approvals) additionally require:

1. A per-process random token in the `X-Oprel-Token` header. Custom headers force
   a CORS preflight, which untrusted origins fail.
2. A Host header naming a loopback address when bound to loopback (blocks DNS rebinding).
3. When bound to a non-loopback interface, an explicit opt-in
   (OPREL_ALLOW_REMOTE_INTEGRATIONS=1), since a LAN-reachable token-bearing daemon
   is a much larger attack surface.

The web UI obtains the token from GET /auth/session, which only answers trusted origins.
"""

from __future__ import annotations

import hmac
import ipaddress
import json
import os
import secrets
import threading
from urllib.parse import urlsplit

from fastapi import HTTPException, Request

from oprel.core.config import Config
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
# Origins (CORS + strict checks)
# ---------------------------------------------------------------------------

def _env_list(name: str) -> list[str]:
    return [v.strip().rstrip("/") for v in os.environ.get(name, "").split(",") if v.strip()]


def cors_allowed_origins() -> list[str]:
    """Origins for CORSMiddleware. OPREL_CORS_ORIGINS extends (or with '*' opens) the list."""
    extra = _env_list("OPREL_CORS_ORIGINS")
    if "*" in extra:
        return ["*"]
    return list(DEFAULT_TRUSTED_ORIGINS) + extra


def _strict_trusted_origins() -> set[str]:
    """Never includes '*': privileged endpoints ignore a wildcard CORS setting."""
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
    return request.headers.get("sec-fetch-site", "none") in ("none", "same-origin")


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


def _host_is_allowed(host_header: str) -> bool:
    if not is_loopback_bind():
        return True  # user deliberately exposed the server; privileged routes are gated separately
    if not host_header:
        return True  # HTTP/1.0-style clients; DNS-rebinding attacks always carry a Host
    hostname = urlsplit(f"//{host_header}").hostname or ""
    if _is_loopback(hostname):
        return True
    return hostname.lower() in {h.lower() for h in _env_list("OPREL_ALLOWED_HOSTS")}


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
    supplied = request.headers.get(TOKEN_HEADER, "")
    if not supplied or not hmac.compare_digest(supplied, get_local_token()):
        raise HTTPException(status_code=401, detail="Missing or invalid local API token")


# ---------------------------------------------------------------------------
# ASGI middleware: Host header guard (DNS-rebinding protection)
# ---------------------------------------------------------------------------

class HostGuardMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            host = ""
            for key, value in scope.get("headers", []):
                if key == b"host":
                    host = value.decode("latin-1")
                    break
            if not _host_is_allowed(host):
                if scope["type"] == "websocket":
                    await send({"type": "websocket.close", "code": 1008})
                    return
                body = json.dumps({"detail": "Host not allowed"}).encode()
                await send({
                    "type": "http.response.start",
                    "status": 403,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
                })
                await send({"type": "http.response.body", "body": body})
                return
        await self.app(scope, receive, send)
