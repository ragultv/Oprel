from fastapi import FastAPI
from fastapi.testclient import TestClient

from oprel.mcp import store
from oprel.mcp import routes as mcp_routes


def _client(tmp_path, monkeypatch):
    from oprel.server import db

    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    store.init_tables()
    app = FastAPI()
    app.include_router(mcp_routes.router)
    return TestClient(app)


def test_catalog_includes_tool_previews_and_install_state(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    entries = {e["id"]: e for e in c.get("/mcp/catalog").json()}
    memory = entries["memory"]
    assert {t["name"] for t in memory["tools"]} >= {"create_entities", "read_graph", "search_nodes"}
    assert any(t["read_only"] for t in memory["tools"]) and any(not t["read_only"] for t in memory["tools"])
    assert memory["installed_id"] is None and memory["checked_version"]
    assert len(entries["filesystem"]["tools"]) == 14


def test_connector_detail_lists_tools_with_modes_and_permissions_can_change(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    conn = store.add_connector("Memory", "npx", [], catalog_id="memory")
    store.save_tools(conn["id"], [
        {"name": "read_graph", "description": "r", "input_schema": {}, "annotations": {"readOnlyHint": True}},
        {"name": "create_entities", "description": "w", "input_schema": {}, "annotations": {}},
    ])
    detail = c.get(f"/mcp/connectors/{conn['id']}").json()
    assert {t["name"]: t["mode"] for t in detail["tools"]} == {"read_graph": "allow", "create_entities": "ask"}

    url = f"/mcp/connectors/{conn['id']}/permissions"
    r = c.put(url, json={"tool": "create_entities", "mode": "deny"})
    assert r.status_code == 200 and {t["name"]: t["mode"] for t in r.json()["tools"]}["create_entities"] == "deny"
    r = c.put(url, json={"group": "read", "mode": "ask"})
    assert {t["name"]: t["mode"] for t in r.json()["tools"]}["read_graph"] == "ask"

    assert c.put(url, json={"tool": "nope", "mode": "allow"}).status_code == 404
    assert c.put(url, json={"tool": "read_graph", "mode": "maybe"}).status_code == 400
    assert c.put(url, json={"mode": "allow"}).status_code == 400
    assert c.get("/mcp/connectors/missing").status_code == 404


def test_installing_twice_is_rejected(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    store.add_connector("Memory", "npx", [], catalog_id="memory")
    assert c.post("/mcp/connectors", json={"catalog_id": "memory"}).status_code == 409
