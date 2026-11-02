"""
End-to-end tool loop with a scripted fake model and a mocked Azure DevOps API:
manual approval -> approve / edit / reject, auto approval, bad arguments, fallback parsing.
"""

import json

import httpx
import pytest

from oprel.integrations.azure_devops import integration as ado
from oprel.integrations.azure_devops import tools as ado_tools
from oprel.integrations.azure_devops.client import AzureDevOpsClient
from oprel.integrations.base import ConnectionTestResult
from oprel.server.tools import adapters, loop
from oprel.server.tools.adapters import AssistantTurn, parse_tool_calls
from oprel.tools.base import ToolCall

PAT = "loop-test-pat-0123456789abcdefghij"


class ScriptedModel:
    """Returns pre-baked turns and records the messages it was shown."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.seen = []

    async def complete(self, messages, tools, params):
        self.seen.append(json.loads(json.dumps(messages)))
        return self.turns.pop(0)


def _call(name, **args):
    return ToolCall(id=f"call_{name}_{len(args)}", name=name, arguments=args, raw_arguments=json.dumps(args))


@pytest.fixture
def connected(client, token, monkeypatch):
    async def fake_test(config, secrets):
        return ConnectionTestResult(ok=True, message="ok", config_updates={"work_item_types": ["Task", "Bug"]})

    monkeypatch.setattr(ado, "test_connection", fake_test)
    r = client.post(
        "/integrations/azure_devops/connect",
        headers={"X-Oprel-Token": token},
        json={"values": {"organization": "contoso", "project": "Web", "pat": PAT}},
    )
    assert r.status_code == 200, r.text

    requests = []

    def handler(req):
        requests.append(req)
        ops = json.loads(req.content) if req.content else []
        title = next((o["value"] for o in ops if o.get("path") == "/fields/System.Title"), "T")
        return httpx.Response(200, json={"id": 4242, "fields": {"System.WorkItemType": "Task", "System.Title": title, "System.State": "New"}})

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        ado_tools, "_client",
        lambda ctx: AzureDevOpsClient(ctx.config["organization"], ctx.config["project"], ctx.secrets["pat"], transport=transport),
    )
    return {"id": r.json()["id"], "requests": requests}


def _use_model(monkeypatch, model):
    monkeypatch.setattr(loop, "resolve_adapter", lambda name: model)


def _events(response):
    events, text = [], ""
    for line in response.text.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        chunk = json.loads(line[6:])
        if chunk.get("oprel_event"):
            events.append(chunk["oprel_event"])
        for c in chunk.get("choices", []):
            text += c.get("delta", {}).get("content") or ""
    return events, text


def _chat(client, token, content="Create a task for the login page"):
    return client.post(
        "/v1/chat/completions",
        headers={"X-Oprel-Token": token},
        json={"model": "fake-local", "stream": True, "enable_tools": ["azure_devops"],
              "messages": [{"role": "user", "content": content}]},
    )


def test_manual_approval_then_approve(client, token, connected, monkeypatch):
    model = ScriptedModel([
        AssistantTurn(content="", tool_calls=[_call("ado_create_work_item", work_item_type="Task", title="Login page")]),
        AssistantTurn(content="Created Task 4242."),
    ])
    _use_model(monkeypatch, model)

    events, text = _events(_chat(client, token))
    pending = [e for e in events if e["type"] == "approval_required"]
    assert len(pending) == 1 and pending[0]["arguments"]["title"] == "Login page"
    assert events[-1] == {"type": "done", "finish_reason": "approval_required", "conversation_id": events[-1]["conversation_id"]}
    assert connected["requests"] == []  # nothing ran yet

    # system prompt got the tool guidance; PAT never shown to the model
    assert "Azure DevOps is connected" in model.seen[0][0]["content"]
    assert PAT not in json.dumps(model.seen)

    r = client.post(f"/tool-approvals/{pending[0]['approval_id']}", headers={"X-Oprel-Token": token}, json={"decision": "approve"})
    events2, text2 = _events(r)
    result = next(e for e in events2 if e["type"] == "tool_result")
    assert result["ok"] and result["data"]["created"]["id"] == 4242
    assert text2 == "Created Task 4242."
    assert len(connected["requests"]) == 1

    # the model saw the tool result on its second turn, wired to the right call id
    tool_msg = model.seen[1][-1]
    assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == pending[0]["call_id"]
    assert PAT not in json.dumps(model.seen)

    # an approval can't be replayed
    again = client.post(f"/tool-approvals/{pending[0]['approval_id']}", headers={"X-Oprel-Token": token}, json={"decision": "approve"})
    assert again.status_code == 409


def test_edit_arguments_before_approving(client, token, connected, monkeypatch):
    _use_model(monkeypatch, ScriptedModel([
        AssistantTurn(content="", tool_calls=[_call("ado_create_work_item", work_item_type="Task", title="draft")]),
        AssistantTurn(content="done"),
    ]))
    events, _ = _events(_chat(client, token))
    approval_id = next(e for e in events if e["type"] == "approval_required")["approval_id"]

    bad = client.post(f"/tool-approvals/{approval_id}", headers={"X-Oprel-Token": token},
                      json={"decision": "approve", "arguments": {"work_item_type": "Task"}})
    assert bad.status_code == 400  # missing title; approval stays pending

    ok = client.post(f"/tool-approvals/{approval_id}", headers={"X-Oprel-Token": token},
                     json={"decision": "approve", "arguments": {"work_item_type": "Task", "title": "Final title"}})
    assert ok.status_code == 200
    sent = json.loads(connected["requests"][0].content)
    assert {"op": "add", "path": "/fields/System.Title", "value": "Final title"} in sent


def test_reject(client, token, connected, monkeypatch):
    model = ScriptedModel([
        AssistantTurn(content="", tool_calls=[_call("ado_create_work_item", work_item_type="Task", title="x")]),
        AssistantTurn(content="Okay, I won't create it."),
    ])
    _use_model(monkeypatch, model)
    events, _ = _events(_chat(client, token))
    approval_id = next(e for e in events if e["type"] == "approval_required")["approval_id"]

    r = client.post(f"/tool-approvals/{approval_id}", headers={"X-Oprel-Token": token},
                    json={"decision": "reject", "reason": "wrong project"})
    events2, text = _events(r)
    assert any(e.get("rejected") for e in events2)
    assert text == "Okay, I won't create it."
    assert connected["requests"] == []
    assert "rejected" in model.seen[1][-1]["content"] and "wrong project" in model.seen[1][-1]["content"]


def test_auto_approval_runs_immediately(client, token, connected, monkeypatch):
    client.put(f"/integrations/{connected['id']}/tools", headers={"X-Oprel-Token": token},
               json={"tools": [{"name": "ado_create_work_item", "approval_mode": "auto"}]})
    _use_model(monkeypatch, ScriptedModel([
        AssistantTurn(content="", tool_calls=[_call("ado_create_work_item", work_item_type="Task", title="auto")]),
        AssistantTurn(content="Created."),
    ]))
    events, text = _events(_chat(client, token))
    assert not any(e["type"] == "approval_required" for e in events)
    assert next(e for e in events if e["type"] == "tool_call")["approval"] == "auto"
    assert text == "Created." and len(connected["requests"]) == 1

    audit = client.get("/integrations/audit/log", headers={"X-Oprel-Token": token}).json()
    assert audit[0]["tool_name"] == "ado_create_work_item" and audit[0]["approval"] == "auto"


def test_invalid_arguments_go_back_to_model(client, token, connected, monkeypatch):
    model = ScriptedModel([
        AssistantTurn(content="", tool_calls=[_call("ado_get_work_item", id="not-a-number")]),
        AssistantTurn(content="Sorry, which ID?"),
    ])
    _use_model(monkeypatch, model)
    events, text = _events(_chat(client, token, "show the item"))
    assert text == "Sorry, which ID?"
    assert "Invalid arguments" in model.seen[1][-1]["content"]
    assert connected["requests"] == []


def test_disabled_tool_is_not_offered(client, token, connected, monkeypatch):
    client.put(f"/integrations/{connected['id']}/tools", headers={"X-Oprel-Token": token},
               json={"tools": [{"name": "ado_create_work_item", "enabled": False}]})
    model = ScriptedModel([
        AssistantTurn(content="", tool_calls=[_call("ado_create_work_item", work_item_type="Task", title="x")]),
        AssistantTurn(content="I can't create items."),
    ])
    _use_model(monkeypatch, model)
    events, _ = _events(_chat(client, token))
    assert "not available" in model.seen[1][-1]["content"]
    assert connected["requests"] == []


def test_no_connection_is_a_clear_400(client, token):
    r = _chat(client, token)
    assert r.status_code == 400 and "Integrations" in r.json()["detail"]


class TestModelResolution:
    def test_local_quant_suffix_is_stripped(self, tool_env, monkeypatch):
        seen = []
        monkeypatch.setattr(adapters, "_local_adapter", lambda m: seen.append(m) or "local")
        assert adapters.resolve_adapter("Qwen/Qwen3-8B-GGUF::Q5_K_M") == "local"
        assert seen == ["Qwen/Qwen3-8B-GGUF"]

    def test_provider_model_routes_to_provider(self, tool_env, monkeypatch):
        from oprel.server import db

        db.upsert_provider({"id": "groq1", "name": "Groq", "type": "groq", "api_key": "k" * 20,
                            "base_url": "", "enabled": True, "enabled_model_ids": [], "available_model_ids": []})
        adapter = adapters.resolve_adapter("groq1::llama-3.3-70b")
        assert adapter.base_url == "https://api.groq.com/openai/v1"
        assert adapter.model == "llama-3.3-70b"


class TestParsing:
    NAMES = {"ado_get_work_item"}

    def test_native_tool_calls(self):
        msg = {"tool_calls": [{"id": "c1", "function": {"name": "ado_get_work_item", "arguments": '{"id": 5}'}}]}
        [c] = parse_tool_calls(msg, self.NAMES)
        assert (c.id, c.name, c.arguments) == ("c1", "ado_get_work_item", {"id": 5})

    def test_malformed_json_is_reported_not_raised(self):
        msg = {"tool_calls": [{"id": "c1", "function": {"name": "ado_get_work_item", "arguments": '{"id": 5'}}]}
        [c] = parse_tool_calls(msg, self.NAMES)
        assert c.parse_error

    def test_fallback_hermes_tags(self):
        msg = {"content": '<think>hmm</think><tool_call>{"name": "ado_get_work_item", "arguments": {"id": 9}}</tool_call>'}
        [c] = parse_tool_calls(msg, self.NAMES)
        assert c.arguments == {"id": 9}

    def test_fallback_ignores_unknown_tools_and_prose(self):
        assert parse_tool_calls({"content": '{"name": "rm_rf", "arguments": {}}'}, self.NAMES) == []
        assert parse_tool_calls({"content": 'Use {"name": "ado_get_work_item"} like this'}, self.NAMES) == []
