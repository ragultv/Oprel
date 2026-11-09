"""
Request guard for the local daemon.

Threat: any web page the user visits can send requests to 127.0.0.1 (drive-by
calls and DNS rebinding).  Three checks stop that:

1. The Host header must be a loopback name (defeats DNS rebinding).
2. Browser requests must be same-site (``Sec-Fetch-Site``) or come from a
   loopback ``Origin``.  Browsers set these headers; page scripts cannot forge them.
3. Everything else (SDK, CLI, curl, OpenAI clients) must send the per-install
   bearer token from ``oprel.utils.auth``.
"""

import os
import re
import secrets
from urllib.parse import urlsplit

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from oprel.utils.auth import AUTH_DISABLED_ENV, get_api_token

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]"}
# Matches http(s)://localhost, 127.0.0.1 or [::1] on any port.
LOOPBACK_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$"
_ORIGIN_RE = re.compile(LOOPBACK_ORIGIN_REGEX)

# Reachable without a token: health probe and the static UI shell.
_OPEN_PATHS = ("/health", "/gui", "/favicon.ico")


def _extra_hosts() -> set:
    raw = os.environ.get("OPREL_ALLOWED_HOSTS", "")
    return {h.strip().lower() for h in raw.split(",") if h.strip()}


def _host_ok(request: Request) -> bool:
    host_header = request.headers.get("host", "")
    host = urlsplit(f"//{host_header}").hostname or ""
    return host.lower() in LOOPBACK_HOSTS or host.lower() in _extra_hosts()


def _bearer_ok(request: Request) -> bool:
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        return False
    return secrets.compare_digest(auth[7:].strip(), get_api_token())


class LocalGuardMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method == "OPTIONS":
            return await call_next(request)

        if not _host_ok(request):
            return JSONResponse({"detail": "Forbidden host"}, status_code=403)

        origin = request.headers.get("origin")
        fetch_site = request.headers.get("sec-fetch-site")

        if origin and not _ORIGIN_RE.match(origin):
            return JSONResponse({"detail": "Forbidden origin"}, status_code=403)
        if fetch_site == "cross-site" and not (origin and _ORIGIN_RE.match(origin)):
            return JSONResponse({"detail": "Cross-site request blocked"}, status_code=403)

        # A browser request that passed the checks above is the local UI.
        browser_ok = bool(origin) or fetch_site in ("same-origin", "same-site", "none")
        if browser_ok or os.environ.get(AUTH_DISABLED_ENV) == "0":
            return await call_next(request)

        path = request.scope.get("path", "")
        if path.startswith("/api/"):
            path = path[4:]
        if any(path == p or path.startswith(p + "/") for p in _OPEN_PATHS):
            return await call_next(request)

        if not _bearer_ok(request):
            return JSONResponse(
                {"detail": "Missing or invalid API token. Use the token in ~/.oprel/api_token."},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
        return await call_next(request)
