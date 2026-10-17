"""
Local API token for the Oprel daemon.

The daemon only listens on loopback, but any web page in the user's browser
can still reach loopback.  Browsers are stopped by an Origin/Host check; every
other client (SDK, CLI, scripts) must present this per-install bearer token.
The token lives in a file readable only by the current user.
"""

import os
import secrets
from pathlib import Path
from typing import Dict

import requests

TOKEN_ENV = "OPREL_API_TOKEN"
AUTH_DISABLED_ENV = "OPREL_API_AUTH"  # set to "0" to disable token enforcement


def _token_path() -> Path:
    return Path.home() / ".oprel" / "api_token"


def get_api_token(create: bool = True) -> str:
    """Return the per-install API token, creating it on first use."""
    env = os.environ.get(TOKEN_ENV)
    if env:
        return env.strip()
    path = _token_path()
    try:
        if path.exists():
            token = path.read_text(encoding="utf-8").strip()
            if token:
                return token
    except OSError:
        pass
    if not create:
        return ""
    token = secrets.token_urlsafe(32)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(token, encoding="utf-8")
    try:
        os.chmod(path, 0o600)  # no-op on Windows, where the user profile ACL applies
    except OSError:
        pass
    return token


def auth_headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {get_api_token()}"}


class _LocalHttp:
    """`requests` wrapper that adds the daemon token to every call."""

    def _call(self, method: str, url: str, **kwargs):
        headers = dict(kwargs.pop("headers", None) or {})
        headers.update(auth_headers())
        return requests.request(method, url, headers=headers, **kwargs)

    def get(self, url, **kw):
        return self._call("GET", url, **kw)

    def post(self, url, **kw):
        return self._call("POST", url, **kw)

    def put(self, url, **kw):
        return self._call("PUT", url, **kw)

    def delete(self, url, **kw):
        return self._call("DELETE", url, **kw)


local_http = _LocalHttp()
