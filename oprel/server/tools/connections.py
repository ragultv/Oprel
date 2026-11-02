"""
Connection lifecycle (connect / test / rotate / disconnect) and tool binding.

This is the only server module that moves secrets between the API layer and the
SecretStore. Public representations never include secret values or refs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from oprel.integrations.base import Integration, IntegrationConfigError
from oprel.integrations.registry import get_integration, list_integrations
from oprel.server.security.secrets import SecretStoreError, get_secret_store
from oprel.server.tools import store
from oprel.tools.base import Tool

APPROVAL_MODES = ("manual", "auto")


class ConnectionError_(ValueError):
    """User-facing error with an HTTP status hint."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

def default_policy(tool: Tool) -> dict[str, Any]:
    # Reads run automatically; anything that changes data asks first.
    return {"enabled": True, "approval_mode": "auto" if tool.kind == "read" else "manual"}


def effective_policies(integration: Integration, connection_id: str) -> dict[str, dict[str, Any]]:
    stored = store.get_policies(connection_id)
    return {t.name: {**default_policy(t), **stored.get(t.name, {})} for t in integration.tools()}


# ---------------------------------------------------------------------------
# Public views
# ---------------------------------------------------------------------------

def public_connection(connection: dict[str, Any]) -> dict[str, Any]:
    integration = get_integration(connection["integration_type"])
    policies = effective_policies(integration, connection["id"]) if integration else {}
    return {
        "id": connection["id"],
        "integration_type": connection["integration_type"],
        "integration_name": integration.name if integration else connection["integration_type"],
        "display_name": connection["display_name"],
        "config": connection["config"],
        "has_secret": bool(connection.get("secret_ref")),
        "secret_hint": connection.get("secret_hint") or "",
        "status": connection["status"],
        "status_message": connection.get("status_message") or "",
        "enabled": connection["enabled"],
        "last_verified_at": connection.get("last_verified_at"),
        "created_at": connection.get("created_at"),
        "updated_at": connection.get("updated_at"),
        "tools": [
            {
                "name": t.name,
                "title": t.title or t.name,
                "description": t.description,
                "kind": t.kind,
                **policies.get(t.name, default_policy(t)),
            }
            for t in (integration.tools() if integration else [])
        ],
    }


def catalog(user_id: int = store.DEFAULT_USER_ID) -> list[dict[str, Any]]:
    connected = {c["integration_type"]: c["id"] for c in store.list_connections(user_id)}
    secret_store = get_secret_store()
    return [
        {
            **i.manifest(),
            "connection_id": connected.get(i.id),
            "secure_storage": secret_store.backend_name,
            "secure_storage_available": secret_store.available,
        }
        for i in list_integrations()
    ]


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def _require_integration(integration_type: str) -> Integration:
    integration = get_integration(integration_type)
    if not integration:
        raise ConnectionError_(f"Unknown integration '{integration_type}'", 404)
    return integration


def _require_connection(connection_id: str, user_id: int) -> dict[str, Any]:
    connection = store.get_connection(connection_id, user_id)
    if not connection:
        raise ConnectionError_("Connection not found", 404)
    return connection


def load_secrets(connection: dict[str, Any]) -> dict[str, str]:
    ref = connection.get("secret_ref")
    if not ref:
        return {}
    try:
        return get_secret_store().get(ref)
    except SecretStoreError as exc:
        raise ConnectionError_(str(exc), 500) from exc


async def connect(integration_type: str, values: dict[str, Any], user_id: int = store.DEFAULT_USER_ID) -> dict[str, Any]:
    """
    Create or update the user's connection. Credentials are verified *before* anything
    is stored, so a typo'd or wrong-scope PAT never lands in the keychain.
    Omitting the secret on an existing connection keeps the stored one (edit org/project only).
    """
    integration = _require_integration(integration_type)
    try:
        config, new_secrets = integration.split_fields(values)
    except IntegrationConfigError as exc:
        raise ConnectionError_(str(exc)) from exc

    existing = store.get_connection_by_type(integration_type, user_id)
    secrets = dict(new_secrets)
    if existing and not new_secrets:
        secrets = load_secrets(existing)
    missing = [k for k in integration.secret_keys if not secrets.get(k)]
    if missing:
        labels = [f.label for f in integration.auth_fields if f.key in missing]
        raise ConnectionError_(f"{', '.join(labels)} is required")

    result = await integration.test_connection(config, secrets)
    if not result.ok:
        raise ConnectionError_(result.message, 400)
    config = integration.normalize_config({**config, **result.config_updates})

    secret_store = get_secret_store()
    secret_ref = existing.get("secret_ref") if existing else None
    if new_secrets or not secret_ref:
        try:
            secret_ref = secret_store.put(secrets, ref=secret_ref)
        except SecretStoreError as exc:
            raise ConnectionError_(str(exc), 500) from exc

    display_name = f"{integration.name} · {config.get('organization', '')}/{config.get('project', '')}".rstrip("/ ·")
    connection = store.upsert_connection(
        integration_type=integration_type,
        display_name=display_name,
        config=config,
        secret_ref=secret_ref,
        secret_hint=integration.secret_hint(secrets) if new_secrets else (existing or {}).get("secret_hint", ""),
        status="connected",
        status_message=result.message,
        user_id=user_id,
    )
    return {**public_connection(connection), "details": result.details}


async def test(connection_id: str, user_id: int = store.DEFAULT_USER_ID) -> dict[str, Any]:
    connection = _require_connection(connection_id, user_id)
    integration = _require_integration(connection["integration_type"])
    secrets = load_secrets(connection)
    result = await integration.test_connection(connection["config"], secrets)
    config = None
    if result.ok and result.config_updates:
        config = integration.normalize_config({**connection["config"], **result.config_updates})
    store.update_connection_status(
        connection_id, "connected" if result.ok else "error", result.message, config
    )
    return {
        "ok": result.ok,
        "message": result.message,
        "details": result.details,
        "connection": public_connection(_require_connection(connection_id, user_id)),
    }


def disconnect(connection_id: str, user_id: int = store.DEFAULT_USER_ID) -> None:
    connection = _require_connection(connection_id, user_id)
    if connection.get("secret_ref"):
        try:
            get_secret_store().delete(connection["secret_ref"])
        except SecretStoreError:
            pass  # still remove the row; an orphaned keychain entry is harmless
    store.delete_connection(connection_id)


def set_enabled(connection_id: str, enabled: bool, user_id: int = store.DEFAULT_USER_ID) -> dict[str, Any]:
    _require_connection(connection_id, user_id)
    store.set_connection_enabled(connection_id, enabled)
    return public_connection(_require_connection(connection_id, user_id))


def update_tool_policies(
    connection_id: str, updates: list[dict[str, Any]], user_id: int = store.DEFAULT_USER_ID
) -> dict[str, Any]:
    connection = _require_connection(connection_id, user_id)
    integration = _require_integration(connection["integration_type"])
    current = effective_policies(integration, connection_id)
    for update in updates:
        name = update.get("name")
        if name not in current:
            raise ConnectionError_(f"Unknown tool '{name}' for {integration.name}")
        mode = update.get("approval_mode", current[name]["approval_mode"])
        if mode not in APPROVAL_MODES:
            raise ConnectionError_(f"approval_mode must be one of {APPROVAL_MODES}")
        enabled = bool(update.get("enabled", current[name]["enabled"]))
        store.set_policy(connection_id, name, enabled, mode)
    return public_connection(_require_connection(connection_id, user_id))


# ---------------------------------------------------------------------------
# Binding tools for a chat request
# ---------------------------------------------------------------------------

@dataclass
class BoundTool:
    tool: Tool
    integration: Integration
    connection: dict[str, Any]
    approval_mode: str

    @property
    def name(self) -> str:
        return self.tool.name


def resolve_active_tools(
    integration_ids: list[str] | None, user_id: int = store.DEFAULT_USER_ID
) -> list[BoundTool]:
    """
    Tools the model may call for this request: integrations the caller asked for
    (None/["*"] = all), connected, enabled, and with the tool itself enabled.
    """
    want_all = not integration_ids or "*" in integration_ids
    bound: list[BoundTool] = []
    for connection in store.list_connections(user_id):
        if not connection["enabled"] or connection["status"] not in ("connected", "error"):
            continue
        if not want_all and connection["integration_type"] not in integration_ids:
            continue
        integration = get_integration(connection["integration_type"])
        if not integration:
            continue
        policies = effective_policies(integration, connection["id"])
        for tool in integration.tools():
            policy = policies[tool.name]
            if policy["enabled"]:
                bound.append(BoundTool(tool, integration, connection, policy["approval_mode"]))
    return bound
