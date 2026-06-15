from __future__ import annotations

import base64
import hashlib
import os
import secrets
import time
from typing import Any

import httpx

from oprel.server import db

# In-memory store for pending OAuth states (code_verifier lives here until callback)
# { state_token: { connector_id, code_verifier, redirect_uri, created_at } }
_pending: dict[str, dict] = {}


# ── PKCE helpers ──────────────────────────────────────────────────────────────

def _generate_pkce() -> tuple[str, str]:
    """Returns (code_verifier, code_challenge)."""
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return verifier, challenge


# ── Step 1: build the authorization URL ──────────────────────────────────────

def build_auth_url(
    connector_id: str,
    oauth_config: dict,
    redirect_uri: str,
) -> tuple[str, str]:
    """
    Returns (authorization_url, state_token).
    Store state_token server-side; send authorization_url to the browser.
    """
    state = secrets.token_urlsafe(32)
    verifier, challenge = _generate_pkce()

    _pending[state] = {
        "connector_id": connector_id,
        "code_verifier": verifier,
        "redirect_uri": redirect_uri,
        "created_at": time.time(),
    }

    client_id = os.environ.get(oauth_config.get("client_id_key", ""), "")
    if not client_id:
        raise ValueError(
            f"Missing env var '{oauth_config.get('client_id_key', '')}'. "
            "Set it in your .env file or environment before starting Oprel."
        )

    scopes = " ".join(oauth_config.get("scopes", []))

    params = {
        "response_type":         "code",
        "client_id":             client_id,
        "redirect_uri":          redirect_uri,
        "scope":                 scopes,
        "state":                 state,
        "code_challenge":        challenge,
        "code_challenge_method": "S256",
        **oauth_config.get("extra_params", {}),
    }

    from urllib.parse import urlencode
    url = oauth_config["authorize_url"] + "?" + urlencode(params)
    return url, state


# ── Step 2: exchange code for tokens ─────────────────────────────────────────

async def exchange_code(
    state: str,
    code: str,
    oauth_config: dict,
) -> dict:
    """
    Exchanges the authorization code for access + refresh tokens.
    Returns the token response dict.
    """
    pending = _pending.pop(state, None)
    if not pending:
        raise ValueError("Invalid or expired OAuth state token.")
    if time.time() - pending["created_at"] > 600:
        raise ValueError("OAuth state expired (10 min limit).")

    client_id     = os.environ.get(oauth_config.get("client_id_key", ""), "")
    client_secret = os.environ.get(
        oauth_config.get("client_secret_key", ""),
        ""
    )

    body = {
        "grant_type":    "authorization_code",
        "code":          code,
        "redirect_uri":  pending["redirect_uri"],
        "client_id":     client_id,
        "code_verifier": pending["code_verifier"],  # PKCE verification
    }
    if client_secret:
        body["client_secret"] = client_secret

    async with httpx.AsyncClient() as hc:
        resp = await hc.post(
            oauth_config["token_url"],
            data=body,
            headers={"Accept": "application/json"},
            timeout=15.0,
        )
        resp.raise_for_status()
        tokens = resp.json()

    # Compute absolute expiry time
    tokens["_expires_at"] = time.time() + tokens.get("expires_in", 3600)
    tokens["_connector_id"] = pending["connector_id"]
    return tokens


# ── Step 3: refresh an expired token ─────────────────────────────────────────

async def refresh_token(connector_id: str, oauth_config: dict) -> str:
    """
    Refreshes the access token if expired. Returns the current access token.
    """
    connector = db.get_mcp_connector(connector_id)
    if not connector:
        raise RuntimeError(f"Connector '{connector_id}' not found")

    cfg = connector.get("config", {})
    access_token  = cfg.get("oauth_token", "")
    refresh_tok   = cfg.get("oauth_refresh_token", "")
    expires_at    = cfg.get("oauth_expires_at", 0)

    # Still valid (with 60s buffer)
    if access_token and time.time() < expires_at - 60:
        return access_token

    if not access_token and not refresh_tok:
        raise RuntimeError(f"OAuth token missing. Please authenticate via the UI.")

    if not refresh_tok:
        raise RuntimeError(f"No refresh token stored for '{connector_id}'. Re-connect required.")

    client_id     = os.environ.get(oauth_config.get("client_id_key", ""), "")
    client_secret = os.environ.get(oauth_config.get("client_secret_key", ""), "")

    body = {
        "grant_type":    "refresh_token",
        "refresh_token": refresh_tok,
        "client_id":     client_id,
    }
    if client_secret:
        body["client_secret"] = client_secret

    async with httpx.AsyncClient() as hc:
        resp = await hc.post(
            oauth_config["token_url"],
            data=body,
            headers={"Accept": "application/json"},
            timeout=15.0,
        )
        resp.raise_for_status()
        tokens = resp.json()

    new_access  = tokens["access_token"]
    new_refresh = tokens.get("refresh_token", refresh_tok)  # some providers rotate it
    new_expiry  = time.time() + tokens.get("expires_in", 3600)

    # Persist updated tokens
    updated_cfg = {
        **cfg,
        "oauth_token":         new_access,
        "oauth_refresh_token": new_refresh,
        "oauth_expires_at":    new_expiry,
    }
    db.upsert_mcp_connector({**connector, "config": updated_cfg})
    return new_access


# ── cleanup stale pending states ──────────────────────────────────────────────

def cleanup_pending() -> None:
    cutoff = time.time() - 600
    stale = [s for s, p in _pending.items() if p["created_at"] < cutoff]
    for s in stale:
        _pending.pop(s, None)
