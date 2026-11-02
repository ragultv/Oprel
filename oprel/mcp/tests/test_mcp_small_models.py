"""Small local models write tool calls their own way; understand them, and say so when a model is too small."""

import asyncio

from test_mcp import _card, _collect, mcp_db  # noqa: F401  (shared helpers and fixture)

from oprel.mcp import approvals, manager, tool_loop

# Exactly what gemma3-1b produced in the chat
GEMMA = 'tool_call\nMemory.search_nodes({names: ["you"]})'


def test_function_call_style_replies_are_understood():
    want = {"connector": "Memory", "tool": "search_nodes", "arguments": {"names": ["you"]}}
    for text in (
        'Memory.search_nodes({"names": ["you"]})',
        "Memory.search_nodes({names: ['you']})",                    # bare keys, single quotes
        '```tool_call\nMemory.search_nodes({names: ["you"]})\n```',  # fenced, as the model printed it
        '```\nMemory.search_nodes({"names": ["you"]});\n```',
    ):
        assert tool_loop.extract_tool_calls(text) == [want], text
    assert tool_loop.extract_tool_calls('Memory.read_graph()') == [
        {"connector": "Memory", "tool": "read_graph", "arguments": {}}
    ]
    assert tool_loop.strip_tool_calls('Memory.read_graph()') == ""


def test_prose_that_mentions_a_call_is_not_executed():
    for text in (
        "You could run Memory.search_nodes({query: 'x'}) to look it up.",
        "Use Memory.read_graph() when you need everything.",
        "e.g. foo.bar(1, 2)",
        "Totally normal answer.",
    ):
        assert tool_loop.extract_tool_calls(text) == [], text


def test_small_model_note_only_for_small_models():
    assert "very small" in tool_loop.small_model_note("gemma3-1b")
    assert "very small" in tool_loop.small_model_note("llama3.2-1b")
    assert "very small" in tool_loop.small_model_note("qwen3-1.7b")
    assert tool_loop.small_model_note("qwen3-4b") == ""
    assert tool_loop.small_model_note("llama3.1-8b") == ""
    assert tool_loop.small_model_note("gemma3-270m") == ""
    assert tool_loop.small_model_note(None) == ""


def test_a_tiny_model_that_writes_function_calls_gets_its_tool_run(mcp_db, monkeypatch):
    calls = []

    async def fake_call(cid, tool, arguments):
        calls.append((tool, arguments))
        return True, '{"entities": []}', 1.0

    monkeypatch.setattr(manager, "call_tool", fake_call)
    c = mcp_db.add_connector("Memory", "npx", [], catalog_id="memory")
    tool = {"name": "search_nodes", "description": "s", "input_schema": {}, "annotations": {"readOnlyHint": True}}
    mcp_db.save_tools(c["id"], [tool])
    monkeypatch.setitem(manager._sessions, c["id"], {"session": object(), "tools": [tool]})

    async def must_not_ask(*a, **k):
        raise AssertionError("read-only tools run without a prompt")

    monkeypatch.setattr(approvals, "request", must_not_ask)
    replies = iter([GEMMA, "You have not told me yet."])

    class Client:
        def generate(self, prompt, stream, **kw):
            return next(replies)

    out = _collect(tool_loop.tool_aware_generate(client=Client(), prompt="P", gen_kwargs={"model": "gemma3-1b"}))
    assert calls == [("search_nodes", {"names": ["you"]})] and _card(out)["tool"] == "search_nodes"
    assert out.strip().endswith("You have not told me yet.")


def test_an_empty_tiny_model_explains_that_it_is_too_small():
    class Silent:
        def generate(self, prompt, stream, **kw):
            return ""

    out = _collect(tool_loop.tool_aware_generate(client=Silent(), prompt="P", gen_kwargs={"model": "gemma3-1b"}))
    assert out.startswith(tool_loop.EMPTY_REPLY) and "very small" in out and "Qwen3" in out
    big = _collect(tool_loop.tool_aware_generate(client=Silent(), prompt="P", gen_kwargs={"model": "qwen3-8b"}))
    assert big == tool_loop.EMPTY_REPLY
