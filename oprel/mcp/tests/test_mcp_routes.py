from fastapi import FastAPI
from fastapi.testclient import TestClient

from oprel.mcp import store
from oprel.mcp import routes as mcp_routes


def _client(tmp_path, monkeypatch):
    from oprel.server import db

    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    from oprel.utils import secrets_store

    monkeypatch.setattr(secrets_store, "_keyring", lambda: None)  # never touch the real keyring in tests
    store.init_tables()
    app = FastAPI()
    app.include_router(mcp_routes.router)
    return TestClient(app)


def test_custom_commands_are_disabled_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv(mcp_routes.ALLOW_CUSTOM_ENV, raising=False)
    r = _client(tmp_path, monkeypatch).post("/mcp/connectors/custom", json={"name": "x", "command": "calc.exe"})
    assert r.status_code == 403
    assert store.list_connectors() == []


def test_filesystem_requires_a_folder_and_rejects_drive_root(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    assert c.post("/mcp/connectors", json={"catalog_id": "filesystem"}).status_code == 400
    root = tmp_path.anchor
    assert c.post("/mcp/connectors", json={"catalog_id": "filesystem", "paths": [root]}).status_code == 400
    assert c.post("/mcp/connectors", json={"catalog_id": "filesystem", "paths": [str(tmp_path / "nope")]}).status_code == 400


def test_unknown_catalog_entry_and_decision_validation(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    assert c.post("/mcp/connectors", json={"catalog_id": "nope"}).status_code == 404
    assert c.post("/mcp/approvals/missing", json={"decision": "once"}).status_code == 404


def test_catalog_lists_network_and_risk(tmp_path, monkeypatch):
    entries = _client(tmp_path, monkeypatch).get("/mcp/catalog").json()
    assert entries and all("network" in e and "risk" in e for e in entries)


def test_failed_start_does_not_leave_a_broken_connector(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)

    async def boom(cid):
        raise RuntimeError("could not start")

    monkeypatch.setattr(mcp_routes.manager, "connect", boom)
    monkeypatch.setattr(mcp_routes.manager, "resolve_npm_version", lambda p: "1.0.0")
    r = c.post("/mcp/connectors", json={"catalog_id": "memory"})
    assert r.status_code == 502 and "could not start" in r.json()["detail"]
    assert store.list_connectors() == []


def test_successful_install_is_pinned_and_listed(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)

    async def ok(cid):
        return None

    monkeypatch.setattr(mcp_routes.manager, "connect", ok)
    monkeypatch.setattr(mcp_routes.manager, "resolve_npm_version", lambda p: "9.9.9")
    r = c.post("/mcp/connectors", json={"catalog_id": "memory"})
    assert r.status_code == 200 and r.json()["version"] == "9.9.9" and r.json()["network"] == "none"
    assert [x["name"] for x in c.get("/mcp/connectors").json()] == ["Memory"]


def test_custom_connector_secrets_are_never_returned(tmp_path, monkeypatch):
    monkeypatch.setenv(mcp_routes.ALLOW_CUSTOM_ENV, "1")
    c = _client(tmp_path, monkeypatch)

    async def ok(cid):
        return None

    monkeypatch.setattr(mcp_routes.manager, "connect", ok)
    r = c.post("/mcp/connectors/custom", json={"name": "x", "command": "tool", "env": {"API_KEY": "super-secret-value"}})
    assert r.status_code == 200
    assert "super-secret-value" not in r.text and r.json()["env"] == ["API_KEY"]
    assert "super-secret-value" not in str(c.get("/mcp/connectors").json())
