import asyncio
import base64
import json

import httpx
import pytest

from oprel.integrations.azure_devops import integration as ado
from oprel.integrations.azure_devops import tools as ado_tools
from oprel.integrations.azure_devops.client import (
    AzureDevOpsClient,
    AzureDevOpsError,
    normalize_organization,
    validate_project,
)
from oprel.tools.base import ToolContext, ToolError

PAT = "testpat-0123456789abcdefghijklmnop"
CONFIG = {"organization": "contoso", "project": "My Project", "work_item_types": ["Bug", "Task", "User Story"]}


def _ctx():
    return ToolContext(user_id=1, connection_id="conn_x", config=dict(CONFIG), secrets={"pat": PAT})


def _patch_transport(monkeypatch, handler):
    seen = []

    def wrapped(request: httpx.Request):
        seen.append(request)
        return handler(request)

    transport = httpx.MockTransport(wrapped)
    monkeypatch.setattr(
        ado_tools,
        "_client",
        lambda ctx: AzureDevOpsClient(ctx.config["organization"], ctx.config["project"], ctx.secrets["pat"], transport=transport),
    )
    return seen


def _work_item(wid=101, **fields):
    return {
        "id": wid,
        "fields": {"System.WorkItemType": "Task", "System.Title": "T", "System.State": "New", **fields},
    }


class TestConfig:
    @pytest.mark.parametrize(
        "raw",
        ["contoso", "https://dev.azure.com/contoso", "https://dev.azure.com/contoso/My%20Project/_boards",
         "https://contoso.visualstudio.com/"],
    )
    def test_org_forms(self, raw):
        assert normalize_organization(raw) == "contoso"

    @pytest.mark.parametrize("raw", ["", "evil.com/x", "a/../b", "-bad", "https://attacker.example/contoso", "x" * 60])
    def test_org_rejects_anything_that_could_redirect_requests(self, raw):
        with pytest.raises(ValueError):
            normalize_organization(raw)

    @pytest.mark.parametrize("raw", ["", "a/b", "a?b", "a#b", "..", "x" * 65])
    def test_project_rejects_path_tricks(self, raw):
        with pytest.raises(ValueError):
            validate_project(raw)


class TestCreate:
    def test_request_shape(self, monkeypatch):
        seen = _patch_transport(monkeypatch, lambda r: httpx.Response(200, json=_work_item(101)))
        args = {
            "work_item_type": "task",
            "title": "Add login page",
            "description": "## Scope\n- form",
            "tags": ["auth", "ui"],
            "parent_id": 7,
        }
        result = asyncio.run(ado_tools.create_work_item(_ctx(), args))

        req = seen[0]
        assert req.method == "POST"
        assert str(req.url).startswith("https://dev.azure.com/contoso/My%20Project/_apis/wit/workitems/$Task?")
        assert "api-version=7.1" in str(req.url)
        assert req.headers["content-type"] == "application/json-patch+json"
        assert req.headers["authorization"] == "Basic " + base64.b64encode(f":{PAT}".encode()).decode()
        ops = json.loads(req.content)
        by_path = {o["path"]: o["value"] for o in ops}
        assert by_path["/fields/System.Title"] == "Add login page"
        assert by_path["/fields/System.Description"] == "<h2>Scope</h2><ul><li>form</li></ul>"
        assert by_path["/fields/System.Tags"] == "auth; ui"
        assert by_path["/relations/-"]["rel"] == "System.LinkTypes.Hierarchy-Reverse"
        assert by_path["/relations/-"]["url"].endswith("/contoso/_apis/wit/workItems/7")

        assert result["created"]["id"] == 101
        assert result["created"]["url"] == "https://dev.azure.com/contoso/My%20Project/_workitems/edit/101"

    def test_unknown_type_lists_valid_ones(self):
        with pytest.raises(ToolError, match="Bug, Task, User Story"):
            asyncio.run(ado_tools.create_work_item(_ctx(), {"work_item_type": "Epic", "title": "x"}))

    def test_alias(self, monkeypatch):
        seen = _patch_transport(monkeypatch, lambda r: httpx.Response(200, json=_work_item()))
        asyncio.run(ado_tools.create_work_item(_ctx(), {"work_item_type": "story", "title": "x"}))
        assert "/workitems/$User%20Story" in str(seen[0].url)


class TestErrors:
    @pytest.mark.parametrize("status,headers", [(401, {}), (203, {"content-type": "text/html"})])
    def test_bad_pat(self, monkeypatch, status, headers):
        _patch_transport(monkeypatch, lambda r: httpx.Response(status, text="<html>sign in</html>", headers=headers))
        with pytest.raises(AzureDevOpsError, match="invalid or expired") as exc:
            asyncio.run(ado_tools.get_work_item(_ctx(), {"id": 1}))
        assert exc.value.status_code == 401

    def test_error_messages_never_echo_the_pat(self, monkeypatch):
        _patch_transport(monkeypatch, lambda r: httpx.Response(400, json={"message": f"bad token {PAT}"}))
        from oprel.utils.redaction import register_secret

        register_secret(PAT)
        with pytest.raises(AzureDevOpsError) as exc:
            asyncio.run(ado_tools.get_work_item(_ctx(), {"id": 1}))
        assert PAT not in str(exc.value)


class TestSearch:
    def test_wiql_is_escaped_and_scoped(self, monkeypatch):
        def handler(r):
            if r.url.path.endswith("/wiql"):
                return httpx.Response(200, json={"workItems": [{"id": 5}, {"id": 6}]})
            return httpx.Response(200, json={"value": [_work_item(5), _work_item(6)]})

        seen = _patch_transport(monkeypatch, handler)
        out = asyncio.run(ado_tools.search_work_items(_ctx(), {"text": "it's ' OR 1=1", "assigned_to_me": True, "top": 5}))
        wiql = json.loads(seen[0].content)["query"]
        assert "[System.TeamProject] = @project" in wiql
        assert "CONTAINS 'it''s '' OR 1=1'" in wiql
        assert "[System.AssignedTo] = @me" in wiql
        assert out["count"] == 2


class TestConnectionTest:
    def test_success_records_types(self, monkeypatch):
        def handler(r):
            if r.url.path.endswith("/workitemtypes"):
                return httpx.Response(200, json={"value": [{"name": "Task"}, {"name": "Bug"}, {"name": "Old", "isDisabled": True}]})
            return httpx.Response(200, json={"authenticatedUser": {"providerDisplayName": "Jane"}})

        transport = httpx.MockTransport(handler)
        real = AzureDevOpsClient.__init__
        monkeypatch.setattr(
            AzureDevOpsClient, "__init__", lambda self, o, p, pat, transport_=None: real(self, o, p, pat, transport=transport)
        )
        result = asyncio.run(ado.test_connection({"organization": "contoso", "project": "P"}, {"pat": PAT}))
        assert result.ok and result.config_updates == {"work_item_types": ["Bug", "Task"]}
        assert "Jane" in result.message
