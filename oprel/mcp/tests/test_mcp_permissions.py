"""Tools, prompts and permissions in the database; tool cards; replies always end with an answer."""

import asyncio
import sqlite3

import pytest
from test_mcp import TOOL, _card, _collect, _setup, mcp_db  # noqa: F401  (shared helpers and fixture)

from oprel.mcp import manager, store, tool_loop

READ = {"name": "read_graph", "description": "reads", "input_schema": {}, "annotations": {"readOnlyHint": True}}
WRITE = {"name": "create_entities", "description": "writes", "input_schema": {}, "annotations": {"readOnlyHint": False}}
DELETE = {"name": "delete_entities", "description": "deletes", "input_schema": {},
          "annotations": {"readOnlyHint": False, "destructiveHint": True}}


def test_tools_and_prompts_are_cached_and_replaced(mcp_db):
    c = mcp_db.add_connector("Memory", "npx", [], catalog_id="memory")
    mcp_db.save_tools(c["id"], [READ, WRITE, DELETE])
    tools = {t["name"]: t for t in mcp_db.get_tools(c["id"])}
    assert tools["read_graph"]["read_only"] and not tools["create_entities"]["read_only"]
    assert tools["delete_entities"]["destructive"]
    mcp_db.save_tools(c["id"], [READ])  # a reconnect replaces the list
    assert [t["name"] for t in mcp_db.get_tools(c["id"])] == ["read_graph"]
    mcp_db.save_prompts(c["id"], [{"name": "p", "description": "d"}])
    assert mcp_db.get_prompts(c["id"]) == [{"name": "p", "description": "d"}]
    mcp_db.delete_connector(c["id"])
    assert mcp_db.get_tools(c["id"]) == [] and mcp_db.get_prompts(c["id"]) == []


def test_group_permissions_apply_to_the_right_tools(mcp_db):
    c = mcp_db.add_connector("Memory", "npx", [], catalog_id="memory")
    mcp_db.save_tools(c["id"], [READ, WRITE, DELETE])
    assert mcp_db.set_group_mode(c["id"], read_only=False, mode="deny") == 2
    assert mcp_db.stored_modes(c["id"]) == {"create_entities": "deny", "delete_entities": "deny"}
    assert mcp_db.effective_mode(c, "read_graph", True) == "allow"  # untouched default
    with pytest.raises(ValueError):
        mcp_db.set_tool_mode(c["id"], "read_graph", "sometimes")


def test_old_grants_are_migrated_to_allow_permissions(tmp_path, monkeypatch):
    from oprel.server import db

    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.execute("CREATE TABLE mcpc_grants (connector_id TEXT, tool TEXT, created_at TEXT, PRIMARY KEY (connector_id, tool))")
    old.execute("INSERT INTO mcpc_grants VALUES ('c1', 'write_file', 'now')")
    old.commit()
    old.close()
    monkeypatch.setattr(db, "DB_PATH", path)
    store.init_tables()
    store.init_tables()
    assert store.stored_modes("c1") == {"write_file": "allow"}
    conn = sqlite3.connect(path)
    assert conn.execute("SELECT 1 FROM sqlite_master WHERE name='mcpc_grants'").fetchone() is None


def test_tool_cards_round_trip_and_are_stripped_from_history():
    from oprel.mcp.policy import strip_tool_cards
    from oprel.server.services import providers  # still strips cards from old chats, even with the feature parked

    card = tool_loop.tool_card("Memory", "create_entities", '{"a": 1}', "ok result", "ok")
    reply = "Saving it." + card + "Done."
    assert _card(reply)["title"] == "Create entities"
    assert "<toolcall>" not in strip_tool_cards(reply) and strip_tool_cards(reply).split() == ["Saving", "it.", "Done."]
    assert strip_tool_cards("plain") == "plain" and strip_tool_cards(None) is None
    kept = providers._drop_empty_messages(
        [{"role": "assistant", "content": card}, {"role": "assistant", "content": reply}]
    )
    assert len(kept) == 1 and "<toolcall>" not in kept[0]["content"] and "Done." in kept[0]["content"]  # card-only turn disappears; cards never reach a model


def _tool_reply(n=0):
    return '<tool_call>{"connector":"Files","tool":"write_file","arguments":{"path":"a%d"}}</tool_call>' % n


def test_reply_still_ends_with_an_answer_when_the_model_keeps_calling_tools(mcp_db, monkeypatch):
    calls = []
    connector, tools = _setup(mcp_db, monkeypatch, calls)
    monkeypatch.setitem(manager._sessions, connector["id"], {"session": object(), "tools": [dict(TOOL)]})
    mcp_db.set_tool_mode(connector["id"], "write_file", "allow")
    seen = []

    async def once(messages):
        seen.append(messages[-1]["content"])
        return "Here is the summary." if messages[-1]["content"] == tool_loop.FINAL_NUDGE else _tool_reply(len(seen))

    out = _collect(tool_loop.tool_aware_generate_messages(generate_once=once, messages=[{"role": "user", "content": "go"}]))
    assert out.endswith("Here is the summary.") and seen[-1] == tool_loop.FINAL_NUDGE
    assert len(calls) <= tool_loop.MAX_ITERATIONS * tool_loop.MAX_CALLS_PER_TURN


def test_reply_after_tools_is_never_blank(mcp_db, monkeypatch):
    calls = []
    connector, tools = _setup(mcp_db, monkeypatch, calls)
    monkeypatch.setitem(manager._sessions, connector["id"], {"session": object(), "tools": [dict(TOOL)]})
    mcp_db.set_tool_mode(connector["id"], "write_file", "allow")

    class Client:
        n = 0

        def generate(self, prompt, stream, **kw):
            Client.n += 1
            if prompt.rstrip().endswith(tool_loop.FINAL_NUDGE):
                return "Final words."
            return _tool_reply() if Client.n == 1 else ""  # one tool call, then the model goes silent

    out = _collect(tool_loop.tool_aware_generate(client=Client(), prompt="P", gen_kwargs={}))
    assert _card(out)["status"] == "ok" and out.strip().endswith("Final words.")
