from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from oprel.tools.base import Tool

AuthFieldType = Literal["text", "url", "secret"]


@dataclass(frozen=True)
class AuthField:
    """
    One input on the connect form. The UI renders the form from these, so a new
    integration needs no frontend code. `secret` fields are write-only: they go to
    the OS keychain and are never returned by the API.
    """

    key: str
    label: str
    type: AuthFieldType = "text"
    required: bool = True
    placeholder: str = ""
    help: str = ""
    help_url: str = ""


@dataclass
class ConnectionTestResult:
    ok: bool
    message: str
    # Non-secret facts worth showing, e.g. {"user": "Jane Doe", "project": "Contoso"}
    details: dict[str, Any] = field(default_factory=dict)
    # Non-secret values learned during the test to persist in the connection config
    # (e.g. the project's available work item types)
    config_updates: dict[str, Any] = field(default_factory=dict)


class IntegrationConfigError(ValueError):
    pass


class Integration(ABC):
    id: str
    name: str
    description: str
    icon: str = "plug"
    docs_url: str = ""
    auth_fields: tuple[AuthField, ...] = ()
    # Lines appended to the system prompt when this integration's tools are active
    system_hint: str = ""

    # -- config ------------------------------------------------------------

    @property
    def secret_keys(self) -> list[str]:
        return [f.key for f in self.auth_fields if f.type == "secret"]

    def split_fields(self, values: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
        """Separate submitted form values into (non-secret config, secrets)."""
        config: dict[str, Any] = {}
        secrets: dict[str, str] = {}
        for f in self.auth_fields:
            raw = values.get(f.key)
            value = raw.strip() if isinstance(raw, str) else raw
            if f.type == "secret":
                if value:
                    secrets[f.key] = value
            elif value not in (None, ""):
                config[f.key] = value
        return self.normalize_config(config), secrets

    def normalize_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Validate/normalize non-secret config. Override for integration-specific rules."""
        for f in self.auth_fields:
            if f.type != "secret" and f.required and not config.get(f.key):
                raise IntegrationConfigError(f"'{f.label}' is required")
        return config

    def secret_hint(self, secrets: dict[str, str]) -> str:
        """A safe-to-display hint ("…x7Qa") for the primary secret."""
        for key in self.secret_keys:
            value = secrets.get(key, "")
            if len(value) >= 8:
                return f"…{value[-4:]}"
        return ""

    # -- behaviour ---------------------------------------------------------

    @abstractmethod
    async def test_connection(self, config: dict[str, Any], secrets: dict[str, str]) -> ConnectionTestResult:
        """Verify credentials with one cheap API call. Must not raise for auth failures."""

    @abstractmethod
    def tools(self) -> list[Tool]:
        ...

    def manifest(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "icon": self.icon,
            "docs_url": self.docs_url,
            "auth_fields": [asdict(f) for f in self.auth_fields],
            "tools": [
                {"name": t.name, "title": t.title or t.name, "description": t.description, "kind": t.kind}
                for t in self.tools()
            ],
        }
