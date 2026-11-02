"""
The PAT must never leave the server: not in API responses, not in SQLite, not in logs.
Privileged endpoints must reject requests without the local token, from foreign
origins, and via DNS-rebinding Host headers.
"""

import logging
import sqlite3

import pytest

from oprel.integrations.azure_devops import integration as ado
from oprel.integrations.base import ConnectionTestResult

PAT = "zq9-SUPER-SECRET-PAT-77abcdefghijklmnopqrstu"
EVIL = "https://evil.example"


@pytest.fixture
def ado_ok(monkeypatch):
    async def fake_test(config, secrets):
        assert secrets == {"pat": PAT}
        return ConnectionTestResult(ok=True, message="Connected as Jane.", config_updates={"work_item_types": ["Task", "Bug"]})

    monkeypatch.setattr(ado, "test_connection", fake_test)


def _connect(client, token, pat=PAT):
    return client.post(
        "/integrations/azure_devops/connect",
        headers={"X-Oprel-Token": token},
        json={"values": {"organization": "https://dev.azure.com/contoso", "project": "Web", "pat": pat}},
    )


class TestAuthGuards:
    def test_session_token_only_for_trusted_origins(self, client):
        assert client.get("/auth/session").status_code == 200
        assert client.get("/auth/session", headers={"Origin": "http://127.0.0.1:11435"}).status_code == 200
        assert client.get("/auth/session", headers={"Origin": "http://localhost:3000"}).status_code == 200
        assert client.get("/auth/session", headers={"Origin": EVIL}).status_code == 403
        assert client.get("/auth/session", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403

    def test_wildcard_cors_does_not_leak_token(self, client, monkeypatch):
        monkeypatch.setenv("OPREL_CORS_ORIGINS", "*")
        assert client.get("/auth/session", headers={"Origin": EVIL}).status_code == 403

    def test_integrations_require_token(self, client, token):
        assert client.get("/integrations").status_code == 401
        assert client.get("/integrations", headers={"X-Oprel-Token": "wrong"}).status_code == 401
        assert client.get("/integrations", headers={"X-Oprel-Token": token}).status_code == 200

    def test_token_from_foreign_origin_rejected(self, client, token):
        r = client.get("/integrations", headers={"X-Oprel-Token": token, "Origin": EVIL})
        assert r.status_code == 403

    def test_tool_chat_requires_token(self, client):
        r = client.post("/v1/chat/completions", json={
            "model": "x", "messages": [{"role": "user", "content": "hi"}], "enable_tools": ["azure_devops"],
        })
        assert r.status_code == 401

    def test_dns_rebinding_host_blocked(self, client):
        assert client.get("/health", headers={"Host": "attacker.example:11435"}).status_code == 403
        assert client.get("/health", headers={"Host": "localhost:11435"}).status_code != 403

    def test_cors_no_longer_wildcard(self, client):
        r = client.options("/v1/models", headers={"Origin": EVIL, "Access-Control-Request-Method": "GET"})
        assert r.headers.get("access-control-allow-origin") not in ("*", EVIL)

    def test_remote_bind_disables_integrations_by_default(self, client, token):
        from oprel.server.security import local_auth

        local_auth.set_bind_host("0.0.0.0")
        try:
            r = client.get("/integrations", headers={"X-Oprel-Token": token, "Host": "192.168.1.5:11435"})
            assert r.status_code == 403
        finally:
            local_auth.set_bind_host("127.0.0.1")


class TestPatNeverLeaks:
    def test_connect_and_read_back(self, client, token, ado_ok, tool_env, caplog):
        caplog.set_level(logging.DEBUG)
        r = _connect(client, token)
        assert r.status_code == 200, r.text
        body = r.text
        assert PAT not in body
        data = r.json()
        assert data["has_secret"] is True
        assert data["secret_hint"] == f"…{PAT[-4:]}"
        assert data["config"]["organization"] == "contoso"
        assert "secret_ref" not in data

        for path in ("/integrations", "/integrations/catalog", "/integrations/audit/log"):
            assert PAT not in client.get(path, headers={"X-Oprel-Token": token}).text

        # SQLite holds only an opaque reference
        raw = sqlite3.connect(tool_env / "test.db").execute("SELECT * FROM integration_connections").fetchall()
        assert raw and PAT not in repr(raw)

        assert PAT not in caplog.text

    def test_secret_lives_in_secret_store(self, client, token, ado_ok):
        from oprel.server.security.secrets import get_secret_store
        from oprel.server.tools import store

        _connect(client, token)
        conn = store.get_connection_by_type("azure_devops")
        assert get_secret_store().get(conn["secret_ref"]) == {"pat": PAT}

    def test_edit_without_pat_keeps_stored_one(self, client, token, ado_ok):
        _connect(client, token)
        r = client.post(
            "/integrations/azure_devops/connect",
            headers={"X-Oprel-Token": token},
            json={"values": {"organization": "contoso", "project": "Web"}},
        )
        assert r.status_code == 200, r.text

    def test_failed_verification_stores_nothing(self, client, token, monkeypatch):
        async def bad(config, secrets):
            return ConnectionTestResult(ok=False, message="Azure DevOps rejected the Personal Access Token")

        monkeypatch.setattr(ado, "test_connection", bad)
        r = _connect(client, token)
        assert r.status_code == 400 and PAT not in r.text
        assert client.get("/integrations", headers={"X-Oprel-Token": token}).json() == []

    def test_disconnect_deletes_secret(self, client, token, ado_ok):
        from oprel.server.security.secrets import get_secret_store
        from oprel.server.tools import store

        cid = _connect(client, token).json()["id"]
        ref = store.get_connection(cid)["secret_ref"]
        assert client.delete(f"/integrations/{cid}", headers={"X-Oprel-Token": token}).status_code == 200
        assert get_secret_store().get(ref) == {}

    def test_policy_update(self, client, token, ado_ok):
        cid = _connect(client, token).json()["id"]
        tools = {t["name"]: t for t in client.get("/integrations", headers={"X-Oprel-Token": token}).json()[0]["tools"]}
        assert tools["ado_create_work_item"]["approval_mode"] == "manual"
        assert tools["ado_get_work_item"]["approval_mode"] == "auto"
        r = client.put(
            f"/integrations/{cid}/tools",
            headers={"X-Oprel-Token": token},
            json={"tools": [{"name": "ado_create_work_item", "approval_mode": "auto"}]},
        )
        assert {t["name"]: t for t in r.json()["tools"]}["ado_create_work_item"]["approval_mode"] == "auto"
        bad = client.put(
            f"/integrations/{cid}/tools",
            headers={"X-Oprel-Token": token},
            json={"tools": [{"name": "nope", "approval_mode": "auto"}]},
        )
        assert bad.status_code == 400
