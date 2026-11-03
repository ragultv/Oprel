"""
The Azure DevOps connector as an MCP server (oprel.mcp_servers), driven through a real
in-process MCP client. Credentials come from the connection stored in Oprel by default:
the PAT must never reach the client, tool switches set in Oprel must apply, and every
call must be audited.
"""

import base64
import json

import anyio
import httpx
import pytest

pytest.importorskip("mcp")

from mcp import Client  # noqa: E402

from oprel.integrations.azure_devops import integration as ado  # noqa: E402
from oprel.integrations.azure_devops import tools as ado_tools  # noqa: E402
from oprel.integrations.azure_devops.client import AzureDevOpsClient  # noqa: E402
from oprel.integrations.base import ConnectionTestResult  # noqa: E402
from oprel.mcp_servers import client_config  # noqa: E402
from oprel.mcp_servers.bridge import IntegrationMCPServer  # noqa: E402
from oprel.mcp_servers.credentials import CredentialError, EnvCredentials, StoredConnection  # noqa: E402

PAT = "mcp-SECRET-pat-0123456789abcdefghijklmn"
ALL_TOOLS = {"ado_create_work_item", "ado_update_work_item", "ado_get_work_item", "ado_search_work_items", "ado_add_comment"}

WORK_ITEM = {
    "id": 42,
    "fields": {"System.Title": "Fix login", "System.WorkItemType": "Task", "System.State": "New"},
    "_links": {"html": {"href": "https://dev.azure.com/contoso/Web/_workitems/edit/42"}},
}


@pytest.fixture
def connected(client, token, monkeypatch):
    async def fake_test(config, secrets):
        return ConnectionTestResult(ok=True, message="Connected.", config_updates={"work_item_types": ["Task", "Bug"]})

    monkeypatch.setattr(ado, "test_connection", fake_test)
    r = client.post(
        "/integrations/azure_devops/connect",
        headers={"X-Oprel-Token": token},
        json={"values": {"organization": "contoso", "project": "Web", "pat": PAT}},
    )
    assert r.status_code == 200, r.text
    return r.json()["id"]


@pytest.fixture
def ado_http(monkeypatch):
    """Fake dev.azure.com. Returns the list of requests it received."""
    class Requests(list):
        state: dict

    seen = Requests()
    state = {"status": 200}

    def handler(request: httpx.Request):
        seen.append(request)
        if state["status"] != 200:
            return httpx.Response(state["status"], json={"message": "nope"})
        return httpx.Response(200, json=WORK_ITEM)

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        ado_tools, "_client",
        lambda ctx: AzureDevOpsClient(ctx.config["organization"], ctx.config["project"], ctx.secrets["pat"], transport=transport),
    )
    seen.state = state
    return seen


def _run(server: IntegrationMCPServer, fn):
    async def go():
        async with Client(server.server) as mcp_client:
            return await fn(mcp_client)

    return anyio.run(go)


def _tool_names(server):
    async def fn(c):
        return {t.name for t in (await c.list_tools()).tools}

    return _run(server, fn)


def _call(server, name, args):
    async def fn(c):
        return await c.call_tool(name, args)

    return _run(server, fn)


class TestStoredConnection:
    def test_requires_a_connection(self, tool_env):
        with pytest.raises(CredentialError, match="not connected"):
            StoredConnection(ado)

    def test_lists_tools_with_annotations(self, connected):
        server = IntegrationMCPServer(ado, StoredConnection(ado))

        async def fn(c):
            return {t.name: t for t in (await c.list_tools()).tools}

        tools = _run(server, fn)
        assert set(tools) == ALL_TOOLS
        assert tools["ado_get_work_item"].annotations.read_only_hint is True
        assert tools["ado_create_work_item"].annotations.read_only_hint is False
        assert tools["ado_update_work_item"].annotations.destructive_hint is True

    def test_call_uses_keychain_pat_and_never_returns_it(self, connected, ado_http):
        server = IntegrationMCPServer(ado, StoredConnection(ado))
        result = _call(server, "ado_create_work_item", {"work_item_type": "task", "title": "Fix login"})

        assert result.is_error is False, result.content[0].text
        assert '"id": 42' in result.content[0].text
        assert PAT not in result.content[0].text and PAT not in json.dumps(result.structured_content)
        auth = ado_http[0].headers["authorization"]
        assert auth == "Basic " + base64.b64encode(f":{PAT}".encode()).decode()

    def test_calls_are_audited(self, connected, ado_http):
        from oprel.server.tools import store

        _call(IntegrationMCPServer(ado, StoredConnection(ado)), "ado_get_work_item", {"id": 42})
        entries = store.list_audit_entries()
        assert entries[0]["tool_name"] == "ado_get_work_item"
        assert entries[0]["approval"] == "mcp_client"
        assert PAT not in json.dumps(entries)

    def test_tool_switched_off_in_oprel_is_hidden_and_refused(self, connected, client, token, ado_http):
        client.put(
            f"/integrations/{connected}/tools",
            headers={"X-Oprel-Token": token},
            json={"tools": [{"name": "ado_update_work_item", "enabled": False}]},
        )
        server = IntegrationMCPServer(ado, StoredConnection(ado))
        assert "ado_update_work_item" not in _tool_names(server)
        result = _call(server, "ado_update_work_item", {"id": 42, "title": "x"})
        assert result.is_error and "disabled" in result.content[0].text
        assert ado_http == []

    def test_paused_connection_refuses_calls(self, connected, client, token, ado_http):
        server = IntegrationMCPServer(ado, StoredConnection(ado))
        client.patch(f"/integrations/{connected}", headers={"X-Oprel-Token": token}, json={"enabled": False})
        result = _call(server, "ado_get_work_item", {"id": 42})
        assert result.is_error
        assert ado_http == []

    def test_rejected_token_marks_connection_expired(self, connected, ado_http):
        from oprel.server.tools import store

        ado_http.state["status"] = 401
        result = _call(IntegrationMCPServer(ado, StoredConnection(ado)), "ado_get_work_item", {"id": 42})
        assert result.is_error and PAT not in result.content[0].text
        assert store.get_connection(connected)["status"] == "expired"


class TestReadOnlyAndValidation:
    def test_read_only_hides_and_refuses_writes(self, connected, ado_http):
        server = IntegrationMCPServer(ado, StoredConnection(ado), read_only=True)
        assert _tool_names(server) == {"ado_get_work_item", "ado_search_work_items"}
        result = _call(server, "ado_create_work_item", {"work_item_type": "Task", "title": "x"})
        assert result.is_error
        assert ado_http == []

    def test_invalid_arguments_never_reach_azure(self, connected, ado_http):
        result = _call(IntegrationMCPServer(ado, StoredConnection(ado)), "ado_get_work_item", {"id": "abc"})
        assert result.is_error and "Invalid arguments" in result.content[0].text
        assert ado_http == []


class TestEnvCredentials:
    def test_missing_variables(self):
        with pytest.raises(CredentialError, match="AZURE_DEVOPS_PAT"):
            EnvCredentials(ado, environ={"AZURE_DEVOPS_ORGANIZATION": "contoso", "AZURE_DEVOPS_PROJECT": "Web"})

    def test_env_mode_works(self, ado_http):
        creds = EnvCredentials(ado, environ={
            "AZURE_DEVOPS_ORGANIZATION": "https://dev.azure.com/contoso",
            "AZURE_DEVOPS_PROJECT": "Web",
            "AZURE_DEVOPS_PAT": PAT,
        })
        result = _call(IntegrationMCPServer(ado, creds), "ado_get_work_item", {"id": 42})
        assert result.is_error is False
        assert str(ado_http[0].url).startswith("https://dev.azure.com/contoso/")


class TestClientConfig:
    def test_snippet_has_no_secrets(self, connected, client, token):
        r = client.get(f"/integrations/{connected}/mcp-config", headers={"X-Oprel-Token": token})
        assert r.status_code == 200
        assert PAT not in r.text
        data = r.json()
        assert data["args"] == ["-m", "oprel.mcp_servers", "serve", "azure_devops"]
        assert "azure-devops" in data["clients"]["mcpServers"]["mcpServers"]

    def test_snippet_requires_token(self, connected, client):
        assert client.get(f"/integrations/{connected}/mcp-config").status_code == 401

    def test_read_only_flag(self):
        assert client_config("azure_devops", read_only=True)["args"][-1] == "--read-only"
