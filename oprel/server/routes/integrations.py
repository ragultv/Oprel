from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from oprel.server.security.local_auth import require_local_token
from oprel.server.tools import connections, store

# Every route here can act with the user's stored credentials: token required.
router = APIRouter(prefix="/integrations", dependencies=[Depends(require_local_token)])


class ConnectRequest(BaseModel):
    # Field values keyed by AuthField.key. Secret fields are write-only.
    values: dict[str, Any] = Field(default_factory=dict)


class ConnectionPatch(BaseModel):
    enabled: bool


class ToolPolicyUpdate(BaseModel):
    name: str
    enabled: bool | None = None
    approval_mode: Literal["manual", "auto"] | None = None


class ToolPoliciesRequest(BaseModel):
    tools: list[ToolPolicyUpdate]


def _raise(exc: connections.ConnectionError_) -> None:
    raise HTTPException(status_code=exc.status_code, detail=str(exc))


@router.get("/catalog")
async def get_catalog():
    return connections.catalog()


@router.get("")
async def list_connections():
    return [connections.public_connection(c) for c in store.list_connections()]


@router.post("/{integration_type}/connect")
async def connect(integration_type: str, body: ConnectRequest):
    try:
        return await connections.connect(integration_type, body.values)
    except connections.ConnectionError_ as exc:
        _raise(exc)


@router.post("/{connection_id}/test")
async def test_connection(connection_id: str):
    try:
        return await connections.test(connection_id)
    except connections.ConnectionError_ as exc:
        _raise(exc)


@router.patch("/{connection_id}")
async def patch_connection(connection_id: str, body: ConnectionPatch):
    try:
        return connections.set_enabled(connection_id, body.enabled)
    except connections.ConnectionError_ as exc:
        _raise(exc)


@router.put("/{connection_id}/tools")
async def update_tools(connection_id: str, body: ToolPoliciesRequest):
    updates = [u.model_dump(exclude_none=True) for u in body.tools]
    try:
        return connections.update_tool_policies(connection_id, updates)
    except connections.ConnectionError_ as exc:
        _raise(exc)


@router.delete("/{connection_id}")
async def disconnect(connection_id: str):
    try:
        connections.disconnect(connection_id)
    except connections.ConnectionError_ as exc:
        _raise(exc)
    return {"success": True}


@router.get("/audit/log")
async def audit_log(limit: int = 100):
    return store.list_audit_entries(limit=max(1, min(limit, 500)))
