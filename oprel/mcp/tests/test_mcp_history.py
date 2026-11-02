"""A reply that looks like JSON must never reach a provider as an object, and bare-JSON tool calls still run."""

from test_mcp import _card, _collect, mcp_db  # noqa: F401  (shared helpers and fixture)

from oprel.mcp import approvals, manager, tool_loop
from oprel.server import db
from oprel.server.services import providers

BARE = '{"connector": "Memory", "tool": "read_graph", "arguments": {}}'






def test_bare_json_and_fenced_json_are_treated_as_tool_calls():
    want = [{"connector": "Memory", "tool": "read_graph", "arguments": {}}]
    assert tool_loop.extract_tool_calls(BARE) == want
    assert tool_loop.extract_tool_calls("```json\n" + BARE + "\n```") == want
    assert tool_loop.strip_tool_calls(BARE) == ""
    # ordinary JSON answers are not mistaken for calls
    assert tool_loop.extract_tool_calls('{"name": "x", "value": 1}') == []
    assert tool_loop.extract_tool_calls("The call looks like " + BARE + " ok") == []
    assert tool_loop.strip_tool_calls('{"name": "x"}') == '{"name": "x"}'


def test_model_that_answers_with_bare_json_still_gets_its_tool_run(mcp_db, monkeypatch):
    calls = []

    async def fake_call(cid, tool, arguments):
        calls.append(tool)
        return True, '{"entities": []}', 1.0

    monkeypatch.setattr(manager, "call_tool", fake_call)
    c = mcp_db.add_connector("Memory", "npx", [], catalog_id="memory")
    tool = {"name": "read_graph", "description": "r", "input_schema": {}, "annotations": {"readOnlyHint": True}}
    mcp_db.save_tools(c["id"], [tool])
    monkeypatch.setitem(manager._sessions, c["id"], {"session": object(), "tools": [tool]})

    async def must_not_ask(*a, **k):
        raise AssertionError("read-only Memory tools run without a prompt")

    monkeypatch.setattr(approvals, "request", must_not_ask)
    replies = iter([BARE, "Your salary is 31k INR."])

    async def once(messages):
        return next(replies)

    out = _collect(tool_loop.tool_aware_generate_messages(generate_once=once, messages=[{"role": "user", "content": "salary?"}]))
    assert calls == ["read_graph"] and _card(out)["status"] == "ok" and out.strip().endswith("Your salary is 31k INR.")
