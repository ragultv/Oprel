"""What is sent to a model provider: no empty turns, no objects as content, JSON-looking replies stay text."""

from oprel.server import db
from oprel.server.services import providers

BARE = '{"connector": "Memory", "tool": "read_graph", "arguments": {}}'


def test_empty_messages_are_dropped_before_provider_calls():
    from oprel.server.services import providers

    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": ""},      # blank bubble left by an earlier failed reply
        {"role": "assistant", "content": None},
        {"role": "assistant", "content": "   "},
        {"role": "user", "content": [{"type": "text", "text": "x"}]},
    ]
    kept = providers._drop_empty_messages(msgs)
    assert [m["role"] for m in kept] == ["system", "user", "user"]


def test_history_keeps_json_looking_replies_as_text_but_still_decodes_multimodal(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "h.db")
    db.init_db()
    cid = db.create_conversation(model_id="m", title="t")
    db.add_message(cid, "user", "hello")
    db.add_message(cid, "assistant", BARE)  # the exact reply that broke Groq with 400 "messages.5"
    db.add_message(cid, "assistant", '[1, 2, 3]')
    db.add_message(cid, "user", '[{"type": "text", "text": "hi"}, {"type": "image_url", "image_url": {"url": "x"}}]')
    msgs = db.get_conversation_messages(cid)
    assert msgs[1]["content"] == BARE and isinstance(msgs[1]["content"], str)
    assert msgs[2]["content"] == "[1, 2, 3]"
    assert isinstance(msgs[3]["content"], list) and msgs[3]["content"][0]["type"] == "text"


def test_provider_never_receives_an_object_as_content():
    kept = providers._drop_empty_messages([
        {"role": "assistant", "content": {"connector": "Memory", "tool": "read_graph"}},
        {"role": "assistant", "content": 42},
        {"role": "user", "content": "ok"},
    ])
    assert all(isinstance(m["content"], (str, list)) for m in kept) and len(kept) == 3
    assert '"connector"' in kept[0]["content"]
