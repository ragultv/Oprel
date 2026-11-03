"""
Where an MCP server gets its credentials from.

StoredConnection (default)
    Uses the connection saved in Oprel (Settings -> Connectors). Secrets are read from the
    OS keychain on every call and never written to the MCP client's config file. Tool on/off
    switches set in Oprel apply, token rotation takes effect without a restart, every call is
    written to Oprel's tool audit log, and a rejected token marks the connection expired.

EnvCredentials (--env)
    Reads <INTEGRATION_ID>_<FIELD_KEY> environment variables, e.g. AZURE_DEVOPS_PAT, for
    machines where Oprel itself isn't set up. The secret then lives in whatever config file
    sets the variable, so prefer the stored connection.
"""

from __future__ import annotations

import os
import sqlite3
from abc import ABC, abstractmethod
from typing import Any

from oprel.integrations.base import Integration, IntegrationConfigError
from oprel.utils.redaction import register_secret

DEFAULT_USER_ID = 1


class CredentialError(RuntimeError):
    """Credentials are missing or unusable. The message is shown to the MCP client."""


class CredentialSource(ABC):
    def __init__(self, integration: Integration):
        self.integration = integration

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable origin for logs, e.g. 'Oprel connection (contoso/Web)'."""

    @abstractmethod
    def load(self) -> tuple[str, dict[str, Any], dict[str, str]]:
        """(connection_id, config, secrets) for one call. Raises CredentialError."""

    def tool_enabled(self, tool_name: str) -> bool:
        return True

    def record(self, *, connection_id: str, tool_name: str, arguments: dict[str, Any],
               status: str, duration_ms: float, summary: str) -> None:
        """Audit hook, called after every tool call."""

    def auth_failed(self, connection_id: str, message: str) -> None:
        """Called when the remote service rejected the credentials (HTTP 401)."""


def env_var_name(integration: Integration, key: str) -> str:
    return f"{integration.id}_{key}".upper()


class EnvCredentials(CredentialSource):
    def __init__(self, integration: Integration, environ: dict[str, str] | None = None):
        super().__init__(integration)
        env = os.environ if environ is None else environ
        values = {f.key: env.get(env_var_name(integration, f.key), "") for f in integration.auth_fields}
        missing = [
            env_var_name(integration, f.key)
            for f in integration.auth_fields
            if f.required and not values[f.key].strip()
        ]
        if missing:
            raise CredentialError(f"Missing environment variables: {', '.join(missing)}")
        try:
            self._config, self._secrets = integration.split_fields(values)
        except IntegrationConfigError as exc:
            raise CredentialError(str(exc)) from exc
        for value in self._secrets.values():
            register_secret(value)

    @property
    def description(self) -> str:
        return "environment variables"

    def load(self) -> tuple[str, dict[str, Any], dict[str, str]]:
        return "env", dict(self._config), dict(self._secrets)


class StoredConnection(CredentialSource):
    def __init__(self, integration: Integration, user_id: int = DEFAULT_USER_ID):
        super().__init__(integration)
        self.user_id = user_id
        self._connection()  # fail fast at startup if nothing is connected

    def _connection(self) -> dict[str, Any]:
        from oprel.server.tools import store

        try:
            connection = store.get_connection_by_type(self.integration.id, self.user_id)
        except sqlite3.Error as exc:
            raise CredentialError(f"Could not read Oprel's database: {exc}") from exc
        if not connection:
            raise CredentialError(
                f"{self.integration.name} is not connected in Oprel. Connect it under "
                "Settings -> Connectors, or start this server with --env."
            )
        if not connection["enabled"]:
            raise CredentialError(f"The {self.integration.name} connection is paused in Oprel.")
        return connection

    @property
    def description(self) -> str:
        config = self._connection()["config"]
        where = "/".join(str(config[f.key]) for f in self.integration.auth_fields if f.type != "secret" and config.get(f.key))
        return f"Oprel connection ({where})" if where else "Oprel connection"

    def load(self) -> tuple[str, dict[str, Any], dict[str, str]]:
        from oprel.server.tools.connections import ConnectionError_, load_secrets

        connection = self._connection()
        try:
            secrets = load_secrets(connection)  # registers values for redaction
        except ConnectionError_ as exc:
            raise CredentialError(str(exc)) from exc
        if not secrets:
            raise CredentialError(f"No credentials are stored for {self.integration.name}. Reconnect it in Oprel.")
        return connection["id"], dict(connection["config"]), secrets

    def tool_enabled(self, tool_name: str) -> bool:
        from oprel.server.tools.connections import effective_policies

        try:
            policies = effective_policies(self.integration, self._connection()["id"])
        except CredentialError:
            return False
        return policies.get(tool_name, {}).get("enabled", True)

    def record(self, *, connection_id: str, tool_name: str, arguments: dict[str, Any],
               status: str, duration_ms: float, summary: str) -> None:
        from oprel.server.tools import store

        store.add_audit_entry(
            conversation_id=None,
            connection_id=connection_id,
            tool_name=tool_name,
            arguments=arguments,
            status=status,
            approval="mcp_client",
            duration_ms=duration_ms,
            result_summary=summary,
            user_id=self.user_id,
        )

    def auth_failed(self, connection_id: str, message: str) -> None:
        from oprel.server.tools import store

        store.update_connection_status(connection_id, "expired", message)
