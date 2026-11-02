"""The assistant must use a connector when it should, and never claim it saved something it did not."""

import asyncio

from test_mcp import _card, _collect, mcp_db  # noqa: F401  (shared helpers and fixture)

from oprel.mcp import approvals, manager, tool_loop


def _memory(mcp_db, monkeypatch):
    calls = []

    async def fake_call(cid, tool, arguments):
        calls.append((tool, arguments))
        return True, '{"entities": []}', 1.0

    monkeypatch.setattr(manager, "call_tool", fake_call)
    c = mcp_db.add_connector("Memory", "npx", [], catalog_id="memory")
    tools = [
        {"name": "read_graph", "description": "reads", "input_schema": {}, "annotations": {"readOnlyHint": True}},
        {"name": "create_entities", "description": "writes", "input_schema": {}, "annotations": {}},
    ]
    mcp_db.save_tools(c["id"], tools)
    monkeypatch.setitem(manager._sessions, c["id"], {"session": object(), "tools": tools})

    async def approve(*a, **k):
        return "once"

    monkeypatch.setattr(approvals, "request", approve)
    return c, calls


READ_CALL = '<tool_call>{"connector": "Memory", "tool": "read_graph", "arguments": {}}</tool_call>'


def test_false_save_claim_triggers_a_tool_call_instead(mcp_db, monkeypatch):
    _, calls = _memory(mcp_db, monkeypatch)
    replies = iter([
        "Your favorite color has been noted as red.",   # claims success, called nothing
        READ_CALL,                                       # after the nudge it uses the tool
        "Your favorite color is teal.",
    ])
    seen = []

    async def once(messages):
        seen.append(messages[-1]["content"])
        return next(replies)

    out = _collect(tool_loop.tool_aware_generate_messages(generate_once=once, messages=[{"role": "user", "content": "x"}]))
    assert calls == [("read_graph", {})] and _card(out)["tool"] == "read_graph"
    assert out.strip().endswith("Your favorite color is teal.")
    assert seen[1] == tool_loop.TOOL_FIRST_NUDGE


def test_not_knowing_triggers_a_lookup_first_for_local_models(mcp_db, monkeypatch):
    _, calls = _memory(mcp_db, monkeypatch)
    replies = iter(["I don’t have that information. Could you let me know?", READ_CALL, "Your salary is 31k INR."])
    prompts = []

    class Client:
        def generate(self, prompt, stream, **kw):
            prompts.append(prompt)
            return next(replies)

    out = _collect(tool_loop.tool_aware_generate(client=Client(), prompt="P", gen_kwargs={}))
    assert calls == [("read_graph", {})] and out.strip().endswith("Your salary is 31k INR.")
    assert tool_loop.TOOL_FIRST_NUDGE in prompts[1]


def test_it_nudges_only_once_and_never_for_normal_answers(mcp_db, monkeypatch):
    _, calls = _memory(mcp_db, monkeypatch)
    n = {"calls": 0}

    async def stubborn(messages):
        n["calls"] += 1
        return "I don't know."

    out = _collect(tool_loop.tool_aware_generate_messages(generate_once=stubborn, messages=[{"role": "user", "content": "x"}]))
    assert n["calls"] == 2 and out == "I don't know." and calls == []  # one nudge, then the answer is accepted

    async def normal(messages):
        n["calls"] += 1
        return "Paris is the capital of France."

    n["calls"] = 0
    out = _collect(tool_loop.tool_aware_generate_messages(generate_once=normal, messages=[{"role": "user", "content": "x"}]))
    assert n["calls"] == 1 and out == "Paris is the capital of France."


def test_no_nudge_when_a_tool_already_ran(mcp_db, monkeypatch):
    _, calls = _memory(mcp_db, monkeypatch)
    replies = iter([READ_CALL, "I don't have any notes about that yet."])
    n = {"calls": 0}

    async def once(messages):
        n["calls"] += 1
        return next(replies)

    out = _collect(tool_loop.tool_aware_generate_messages(generate_once=once, messages=[{"role": "user", "content": "x"}]))
    assert n["calls"] == 2 and out.strip().endswith("I don't have any notes about that yet.")


def test_prompt_tells_the_model_when_to_use_memory(mcp_db, monkeypatch):
    _memory(mcp_db, monkeypatch)
    block = tool_loop.prompt_block()
    assert "When to use them" in block and "search_nodes or read_graph BEFORE answering" in block
    assert "Never say something was saved" in block
