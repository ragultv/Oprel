"""Fixtures for the parked MCP tests. Run them with:  pytest oprel/mcp/tests"""

import pytest


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch):
    """An in-memory keyring so the real OS credential store is never read or changed."""
    from types import SimpleNamespace

    from oprel.utils import secrets_store

    vault = {}
    fake = SimpleNamespace(
        set_password=lambda service, name, value: vault.__setitem__((service, name), value),
        get_password=lambda service, name: vault.get((service, name)),
        delete_password=lambda service, name: vault.pop((service, name), None),
    )
    monkeypatch.setattr(secrets_store, "_keyring", lambda: fake)
    yield vault
