from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from oprel.mcp import approvals, manager, store
from oprel.mcp.registry import CATALOG, get_catalog_entry, get_preview

router = APIRouter(prefix="/mcp")

# Arbitrary commands are an advanced, opt-in feature: a connector is a program
# that runs with the user's rights.
ALLOW_CUSTOM_ENV = "OPREL_MCP_ALLOW_CUSTOM"


class InstallRequest(BaseModel):
    catalog_id: str
    paths: list[str] = Field(default_factory=list)
    version: str | None = None  # optional override; otherwise the latest is looked up and pinned


class CustomRequest(BaseModel):
    name: str
    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)


class PatchRequest(BaseModel):
    enabled: bool | None = None
    use_in_chat: bool | None = None


class PermissionRequest(BaseModel):
    mode: str  # allow | ask | deny
    tool: str | None = None  # one tool ...
    group: str | None = None  # ... or "read" (read-only tools) / "write" (write and delete tools)


class DecisionRequest(BaseModel):
    decision: str  # once | always | deny


def _validated_paths(paths: list[str]) -> list[str]:
    out = []
    for p in paths:
        resolved = Path(p).expanduser().resolve()
        if not resolved.is_dir():
            raise HTTPException(400, f"Folder not found: {p}")
        if resolved == Path(resolved.anchor):
            raise HTTPException(400, "Choose a specific folder, not a whole drive.")
        out.append(str(resolved))
    return out


def _tool_views(connector: dict[str, Any]) -> list[dict[str, Any]]:
    """Cached tools of a connector with the permission that applies to each."""
    stored = store.stored_modes(connector["id"])
    return [
        {
            "name": t["name"],
            "description": t["description"],
            "read_only": t["read_only"],
            "mode": stored.get(t["name"]) or store.default_mode(connector, t["read_only"]),
            "customized": t["name"] in stored,
        }
        for t in store.get_tools(connector["id"])
    ]


@router.get("/catalog")
async def catalog() -> list[dict[str, Any]]:
    installed = {c["catalog_id"]: c["id"] for c in store.list_connectors() if c.get("catalog_id")}
    out = []
    for entry in CATALOG:
        preview = get_preview(entry["id"])
        out.append({
            **entry,
            "installed_id": installed.get(entry["id"]),
            "tools": preview["tools"],
            "prompts": preview["prompts"],
            "checked_version": preview["checked_version"],
        })
    return out


@router.get("/connectors")
async def list_connectors() -> list[dict[str, Any]]:
    out = []
    for c in store.list_connectors():
        entry = get_catalog_entry(c.get("catalog_id") or "") or {}
        out.append({
            **store.public(c),
            "description": entry.get("description", ""),
            "tool_count": len(store.get_tools(c["id"])),
            "tools": [t["name"] for t in manager.get_tools(c["id"])],
        })
    return out


@router.get("/connectors/{cid}")
async def connector_detail(cid: str) -> dict[str, Any]:
    connector = store.get_connector(cid)
    if not connector:
        raise HTTPException(404, "Connector not found.")
    entry = get_catalog_entry(connector.get("catalog_id") or "") or {}
    return {
        **store.public(connector),
        "description": entry.get("description", ""),
        "risk": entry.get("risk"),
        "tools": _tool_views(connector),
        "prompts": store.get_prompts(cid),
    }


@router.post("/connectors")
async def install(req: InstallRequest) -> dict[str, Any]:
    entry = get_catalog_entry(req.catalog_id)
    if not entry:
        raise HTTPException(404, "Unknown connector.")
    if any(c.get("catalog_id") == entry["id"] for c in store.list_connectors()):
        raise HTTPException(409, f"{entry['name']} is already installed.")
    extra_args: list[str] = []
    if entry.get("needs_paths"):
        if not req.paths:
            raise HTTPException(400, "Choose at least one folder this connector may use.")
        extra_args = _validated_paths(req.paths)
    try:
        version = req.version or await asyncio.to_thread(manager.resolve_npm_version, entry["package"])
        command, args = manager.build_command(entry, version, extra_args)
    except Exception as exc:
        raise HTTPException(502, str(exc))
    connector = store.add_connector(
        entry["name"], command, args, catalog_id=entry["id"], package=entry["package"],
        version=version, network=entry["network"],
    )
    try:
        await manager.connect(connector["id"])
    except Exception as exc:
        # Do not leave a broken connector behind; the user can fix the details and retry.
        store.delete_connector(connector["id"])
        raise HTTPException(502, f"The connector could not start, so it was not added: {exc}")
    return store.public(store.get_connector(connector["id"]))


@router.post("/connectors/custom")
async def install_custom(req: CustomRequest) -> dict[str, Any]:
    if os.environ.get(ALLOW_CUSTOM_ENV) != "1":
        raise HTTPException(
            403,
            f"Custom connectors run any program on your computer, so they are off by default. "
            f"Set {ALLOW_CUSTOM_ENV}=1 and restart Oprel to enable them.",
        )
    connector = store.add_connector(req.name, req.command, req.args, env=req.env, network="unknown")
    try:
        await manager.connect(connector["id"])
    except Exception as exc:
        store.delete_connector(connector["id"])
        raise HTTPException(502, f"The connector could not start, so it was not added: {exc}")
    return store.public(store.get_connector(connector["id"]))


@router.patch("/connectors/{cid}")
async def patch_connector(cid: str, req: PatchRequest) -> dict[str, Any]:
    if not store.get_connector(cid):
        raise HTTPException(404, "Connector not found.")
    fields = {k: v for k, v in req.model_dump().items() if v is not None}
    store.update_connector(cid, **fields)
    if req.enabled is True:
        try:
            await manager.connect(cid)
        except Exception as exc:
            raise HTTPException(502, f"The connector did not start: {exc}")
    elif req.enabled is False:
        await manager.disconnect(cid)
    return store.public(store.get_connector(cid))


@router.put("/connectors/{cid}/permissions")
async def set_permissions(cid: str, req: PermissionRequest) -> dict[str, Any]:
    """Set what the assistant may do: for one tool, or for the read-only / write-and-delete group."""
    connector = store.get_connector(cid)
    if not connector:
        raise HTTPException(404, "Connector not found.")
    if req.mode not in store.MODES:
        raise HTTPException(400, "Mode must be allow, ask or deny.")
    if req.tool:
        if req.tool not in {t["name"] for t in store.get_tools(cid)}:
            raise HTTPException(404, "Unknown tool.")
        store.set_tool_mode(cid, req.tool, req.mode)
    elif req.group in ("read", "write"):
        store.set_group_mode(cid, req.group == "read", req.mode)
    else:
        raise HTTPException(400, "Give a tool, or a group of read or write.")
    return {"tools": _tool_views(connector)}


@router.delete("/connectors/{cid}")
async def remove_connector(cid: str) -> dict[str, bool]:
    await manager.disconnect(cid)
    store.delete_connector(cid)
    return {"success": True}


@router.get("/approvals")
async def pending_approvals() -> list[dict[str, Any]]:
    return approvals.list_pending()


@router.post("/approvals/{approval_id}")
async def decide(approval_id: str, req: DecisionRequest) -> dict[str, bool]:
    if not approvals.resolve(approval_id, req.decision):
        raise HTTPException(404, "This request is no longer waiting.")
    return {"success": True}


@router.get("/logs")
async def logs(limit: int = 100) -> list[dict[str, Any]]:
    return store.list_logs(min(max(limit, 1), 500))
