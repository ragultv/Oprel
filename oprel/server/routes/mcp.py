from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from oprel.server import db
from oprel.mcp import manager as mcp_manager
from oprel.mcp import oauth as mcp_oauth
from oprel.mcp.registry import get_builtin_connector, list_builtin_connectors

router = APIRouter(prefix="/mcp", tags=["mcp"])


# ── request schemas ───────────────────────────────────────────────────────────

class CreateConnectorRequest(BaseModel):
    builtin_id: str
    name: str | None = None
    config: dict[str, Any] = {}
    enabled: bool = True


class UpdateConnectorRequest(BaseModel):
    name: str | None = None
    config: dict[str, Any] | None = None
    enabled: bool | None = None


# ── catalog ───────────────────────────────────────────────────────────────────

@router.get("/catalog")
async def get_catalog():
    return {"connectors": list_builtin_connectors()}


@router.get("/catalog/{builtin_id}")
async def get_catalog_item(builtin_id: str):
    builtin = get_builtin_connector(builtin_id)
    if not builtin:
        raise HTTPException(404, f"No built-in connector '{builtin_id}'")
    return builtin


# ── connector instances ───────────────────────────────────────────────────────

@router.get("/connectors")
async def list_connectors():
    instances = db.list_mcp_connectors()
    live = set(mcp_manager.get_all_live_connector_ids())
    for inst in instances:
        if inst["id"] in live and inst.get("status") != "connected":
            inst["status"] = "connected"
    return {"connectors": instances}


@router.post("/connectors", status_code=201)
async def add_connector(body: CreateConnectorRequest, background_tasks: BackgroundTasks):
    builtin = get_builtin_connector(body.builtin_id)
    if not builtin and body.builtin_id != "__custom__":
        raise HTTPException(400, f"Unknown built-in connector id '{body.builtin_id}'")

    connector_id = body.builtin_id if body.builtin_id != "__custom__" else f"custom_{uuid.uuid4().hex[:8]}"
    if db.get_mcp_connector(connector_id):
        connector_id = f"{connector_id}_{uuid.uuid4().hex[:6]}"

    template = builtin or {}
    data = {
        "id": connector_id,
        "builtin_id": body.builtin_id,
        "name": body.name or template.get("name", connector_id),
        "transport": template.get("transport", body.config.get("transport", "stdio")),
        "config": body.config,
        "enabled": body.enabled,
        "status": "disconnected",
    }
    connector = db.upsert_mcp_connector(data)

    if body.enabled:
        background_tasks.add_task(mcp_manager.connect, connector_id)

    return {"connector": connector}


@router.get("/connectors/{connector_id}")
async def get_connector(connector_id: str):
    connector = db.get_mcp_connector(connector_id)
    if not connector:
        raise HTTPException(404, f"Connector '{connector_id}' not found")
    return {"connector": connector}


@router.patch("/connectors/{connector_id}")
async def update_connector(
    connector_id: str, body: UpdateConnectorRequest, background_tasks: BackgroundTasks
):
    existing = db.get_mcp_connector(connector_id)
    if not existing:
        raise HTTPException(404, f"Connector '{connector_id}' not found")

    updated = dict(existing)
    if body.name is not None:
        updated["name"] = body.name
    if body.config is not None:
        updated["config"] = {**existing.get("config", {}), **body.config}

    was_enabled = existing.get("enabled", True)
    if body.enabled is not None:
        updated["enabled"] = body.enabled

    connector = db.upsert_mcp_connector(updated)

    if body.enabled is True and not was_enabled:
        background_tasks.add_task(mcp_manager.connect, connector_id)
    elif body.enabled is False and was_enabled:
        background_tasks.add_task(mcp_manager.disconnect, connector_id)
    elif body.config is not None:
        background_tasks.add_task(mcp_manager.reconnect, connector_id)

    return {"connector": connector}


@router.delete("/connectors/{connector_id}", status_code=204)
async def delete_connector(connector_id: str, background_tasks: BackgroundTasks):
    if not db.get_mcp_connector(connector_id):
        raise HTTPException(404, f"Connector '{connector_id}' not found")
    background_tasks.add_task(mcp_manager.disconnect, connector_id)
    db.delete_mcp_connector(connector_id)
    return None


# ── connection control ────────────────────────────────────────────────────────

@router.post("/connectors/{connector_id}/connect")
async def connect_connector(connector_id: str, background_tasks: BackgroundTasks):
    if not db.get_mcp_connector(connector_id):
        raise HTTPException(404, f"Connector '{connector_id}' not found")
    background_tasks.add_task(mcp_manager.connect, connector_id)
    return {"status": "connecting", "connector_id": connector_id}


@router.post("/connectors/{connector_id}/disconnect")
async def disconnect_connector(connector_id: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(mcp_manager.disconnect, connector_id)
    return {"status": "disconnecting", "connector_id": connector_id}


@router.post("/connectors/{connector_id}/reconnect")
async def reconnect_connector(connector_id: str, background_tasks: BackgroundTasks):
    if not db.get_mcp_connector(connector_id):
        raise HTTPException(404, f"Connector '{connector_id}' not found")
    background_tasks.add_task(mcp_manager.reconnect, connector_id)
    return {"status": "reconnecting", "connector_id": connector_id}


@router.post("/connectors/{connector_id}/test")
async def test_connector(connector_id: str):
    if not db.get_mcp_connector(connector_id):
        raise HTTPException(404, f"Connector '{connector_id}' not found")
    return await mcp_manager.test_connection(connector_id)


# ── tools ─────────────────────────────────────────────────────────────────────

@router.get("/connectors/{connector_id}/tools")
async def get_connector_tools(connector_id: str):
    if not db.get_mcp_connector(connector_id):
        raise HTTPException(404, f"Connector '{connector_id}' not found")
    tools = db.get_mcp_tools(connector_id)
    return {"connector_id": connector_id, "tools": tools, "count": len(tools)}


@router.get("/tools")
async def get_all_tools():
    tools = db.get_all_mcp_enabled_tools()
    return {"tools": tools, "count": len(tools)}


# ── audit log & status ────────────────────────────────────────────────────────

@router.get("/logs")
async def get_call_logs(limit: int = 50, conversation_id: str | None = None):
    logs = db.list_mcp_call_logs(limit=limit, conversation_id=conversation_id)
    return {"logs": logs, "count": len(logs)}


@router.get("/status")
async def get_mcp_status():
    connectors = db.list_mcp_connectors()
    live_ids = set(mcp_manager.get_all_live_connector_ids())
    tools = db.get_all_mcp_enabled_tools()
    return {
        "total_connectors": len(connectors),
        "connected": sum(1 for c in connectors if c["id"] in live_ids),
        "disconnected": sum(1 for c in connectors if c["id"] not in live_ids),
        "total_tools": len(tools),
        "live_connector_ids": list(live_ids),
    }


# ── oauth ─────────────────────────────────────────────────────────────────────

@router.get("/connectors/{connector_id}/oauth/start")
async def oauth_start(connector_id: str, request: Request):
    """
    Returns the authorization URL. Frontend opens this in a popup/redirect.
    """
    connector = db.get_mcp_connector(connector_id)
    if not connector:
        raise HTTPException(404, "Connector not found")

    builtin = get_builtin_connector(connector["builtin_id"]) or {}
    oauth_cfg = builtin.get("oauth")
    if not oauth_cfg:
        raise HTTPException(400, "This connector does not use OAuth")

    redirect_uri = str(request.base_url) + "api/mcp/oauth/callback"
    auth_url, state = mcp_oauth.build_auth_url(connector_id, oauth_cfg, redirect_uri)

    return {"auth_url": auth_url, "state": state}


@router.get("/oauth/callback")
async def oauth_callback(
    code: str,
    state: str,
    background_tasks: BackgroundTasks,
):
    """
    Provider redirects here after user approves.
    Exchanges code for tokens, stores them, reconnects the session.
    """
    # Recover which connector this belongs to
    pending = mcp_oauth._pending.get(state)
    if not pending:
        raise HTTPException(400, "Invalid or expired OAuth state")

    connector_id = pending["connector_id"]
    connector    = db.get_mcp_connector(connector_id)
    builtin      = get_builtin_connector(connector["builtin_id"]) or {}
    oauth_cfg    = builtin.get("oauth", {})

    try:
        tokens = await mcp_oauth.exchange_code(state, code, oauth_cfg)
    except Exception as exc:
        raise HTTPException(400, f"Token exchange failed: {exc}")

    # Store tokens in connector config
    updated_cfg = {
        **connector.get("config", {}),
        "oauth_token":         tokens["access_token"],
        "oauth_refresh_token": tokens.get("refresh_token", ""),
        "oauth_expires_at":    tokens["_expires_at"],
    }
    db.upsert_mcp_connector({**connector, "config": updated_cfg})

    # Reconnect the live session with the new token
    background_tasks.add_task(mcp_manager.reconnect, connector_id)

    # Close the popup / redirect to settings page
    return HTMLResponse("""
        <script>
            window.opener?.postMessage({type:'oauth_success'}, '*');
            window.close();
        </script>
        <p>Connected! You can close this tab.</p>
    """)
