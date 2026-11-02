"""The model must read tool descriptions and results, and not say "saved" after a result that says "skipped"."""

import json
import pathlib

from test_mcp import _card, _collect, mcp_db  # noqa: F401  (shared helpers and fixture)

from oprel.mcp import approvals, manager, tool_loop

SKIPPED = '[]\nSkipped 1 entity that already exists: User. Its observations were not added; use add_observations.'
CREATE = '<tool_call>{"connector": "Memory", "tool": "create_entities", "arguments": {"entities": [{"name": "User", "entityType": "person", "observations": ["likes apples"]}]}}</tool_call>'
ADD = '<tool_call>{"connector": "Memory", "tool": "add_observations", "arguments": {"observations": [{"entityName": "User", "contents": ["likes apples"]}]}}</tool_call>'


def _snapshot_tools():
    data = json.loads((pathlib.Path(tool_loop.__file__).parent / "catalog_tools.json").read_text(encoding="utf-8"))
    return [
        {"name": t["name"], "description": t["description"], "input_schema": {}, "annotations": {"readOnlyHint": t["read_only"]}}
        for t in data["memory"]["tools"]
    ]


def _memory(mcp_db, monkeypatch, results):
    calls = []

    async def fake_call(cid, tool, arguments):
        calls.append(tool)
        return True, results.get(tool, "ok"), 1.0

    monkeypatch.setattr(manager, "call_tool", fake_call)
    c = mcp_db.add_connector("Memory", "npx", [], catalog_id="memory")
    tools = _snapshot_tools()
    mcp_db.save_tools(c["id"], tools)
    monkeypatch.setitem(manager._sessions, c["id"], {"session": object(), "tools": tools})

    async def approve(*a, **k):
        return "once"

    monkeypatch.setattr(approvals, "request", approve)
    return calls


def test_the_prompt_keeps_the_part_of_the_description_that_says_what_to_do_next(mcp_db, monkeypatch):
    _memory(mcp_db, monkeypatch, {})
    block = tool_loop.prompt_block()
    assert "Use add_observations to add observations to an existing entity" in block  # was cut off at 200 characters
    assert "search_nodes for" in block and "add_observations on it" in block  # the usage hint


def test_a_skipped_result_is_flagged_for_the_model(mcp_db, monkeypatch):
    _memory(mcp_db, monkeypatch, {"create_entities": SKIPPED})
    tools = manager.all_live_tools()
    block, card = __import__("asyncio").run(tool_loop.execute_call(json.loads(CREATE[11:-12]), tools))
    assert tool_loop.INCOMPLETE_NOTE in block and "Skipped 1 entity" in block
    assert _card(card)["status"] == "ok"  # the call itself worked; the card shows the real result text
    assert tool_loop.result_incomplete(SKIPPED) and not tool_loop.result_incomplete("Saved 1 entity.")


def test_saying_saved_right_after_a_skipped_result_makes_it_follow_up(mcp_db, monkeypatch):
    calls = _memory(mcp_db, monkeypatch, {"create_entities": SKIPPED, "add_observations": "Added 1 observation"})
    replies = iter([
        CREATE,
        "Your favorite fruit, apple, has been noted.",   # the exact mistake from the chat
        ADD,                                              # after the nudge it does what the result said
        "Done: I saved that your favorite fruit is apple.",
    ])
    seen = []

    async def once(messages):
        seen.append(messages[-1]["content"])
        return next(replies)

    out = _collect(tool_loop.tool_aware_generate_messages(generate_once=once, messages=[{"role": "user", "content": "remember"}]))
    assert calls == ["create_entities", "add_observations"]
    assert tool_loop.RESULT_NUDGE in seen
    assert out.strip().endswith("I saved that your favorite fruit is apple.")


def test_no_follow_up_nudge_when_the_result_was_clean(mcp_db, monkeypatch):
    calls = _memory(mcp_db, monkeypatch, {"create_entities": "Created 1 entity."})
    replies = iter([CREATE, "Saved that your favorite fruit is apple."])
    seen = []

    async def once(messages):
        seen.append(messages[-1]["content"])
        return next(replies)

    out = _collect(tool_loop.tool_aware_generate_messages(generate_once=once, messages=[{"role": "user", "content": "remember"}]))
    assert calls == ["create_entities"] and tool_loop.RESULT_NUDGE not in seen and len(seen) == 2
    assert out.strip().endswith("Saved that your favorite fruit is apple.")
