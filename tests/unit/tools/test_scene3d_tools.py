"""
3D scene toolset end-to-end: the chat tool loop with a scripted model, the scene_* tools,
built-in binding (no connection, auto-approved, no write cap) and the /scene3d routes.
"""

import io
import json

import trimesh

from oprel.server.tools import loop
from oprel.server.tools.adapters import AssistantTurn
from oprel.tools.base import ToolCall

CONV = "scene-test-conv"


class ScriptedModel:
    def __init__(self, turns):
        self.turns = list(turns)
        self.seen = []

    async def complete(self, messages, tools, params):
        self.seen.append({"messages": json.loads(json.dumps(messages)), "tools": [t["function"]["name"] for t in tools],
                          "temperature": params.temperature})
        return self.turns.pop(0)


def _call(name, n=0, **args):
    return ToolCall(id=f"call_{name}_{n}", name=name, arguments=args, raw_arguments=json.dumps(args))


def _chat(client, token, monkeypatch, turns, enable=("scene3d",)):
    model = ScriptedModel(turns)
    monkeypatch.setattr(loop, "resolve_adapter", lambda name: model)
    r = client.post(
        "/v1/chat/completions",
        headers={"X-Oprel-Token": token},
        json={"model": "fake-local", "stream": True, "enable_tools": list(enable), "conversation_id": CONV,
              "messages": [{"role": "user", "content": "draw something"}]},
    )
    events, text = [], ""
    for line in r.text.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        chunk = json.loads(line[6:])
        if chunk.get("oprel_event"):
            events.append(chunk["oprel_event"])
        for c in chunk.get("choices", []):
            text += c.get("delta", {}).get("content") or ""
    return r, model, events, text


def _results(events):
    return [e for e in events if e["type"] == "tool_result"]


CYL_AND_RECT = [
    {"id": "cyl", "shape": "cylinder", "params": {"radius": 1, "height": 2}},
    {"id": "rect", "shape": "rectangle", "params": {"width": 1.5, "depth": 1},
     "placement": {"relative_to": "cyl", "side": "right", "gap": 0.2}},
]


def test_draw_runs_without_approval_and_is_saved(client, token, monkeypatch):
    r, model, events, text = _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_add_objects", plan="cylinder + plate", objects=CYL_AND_RECT)]),
        AssistantTurn(content="Drew a cylinder with a plate to its right."),
    ])
    assert r.status_code == 200, r.text
    assert not [e for e in events if e["type"] == "approval_required"]
    (result,) = _results(events)
    assert result["ok"], result
    data = result["data"]
    assert data["scene_version"] == 1 and data["total_objects"] == 2
    rect = next(o for o in data["objects"] if o["id"] == "rect")
    assert rect["min"] == [1.2, 0.0, -0.5] and rect["max"] == [2.7, 0.002, 0.5]
    assert text == "Drew a cylinder with a plate to its right."

    # The model got the conventions and only the scene tools.
    system = model.seen[0]["messages"][0]["content"]
    assert "draw in a 3D scene" in system and "bottom-centre" in system
    assert set(model.seen[0]["tools"]) == {"scene_add_objects", "scene_update_objects", "scene_delete_objects",
                                           "scene_get_scene"}
    # ...and saw exact bounds in the tool result
    tool_msg = json.loads(model.seen[1]["messages"][-1]["content"])
    assert tool_msg["ok"] and tool_msg["objects"][1]["max"] == [2.7, 0.002, 0.5]

    scene = client.get(f"/scene3d/{CONV}", headers={"X-Oprel-Token": token}).json()
    assert scene["version"] == 1 and [o["id"] for o in scene["objects"]] == ["cyl", "rect"]

    glb = client.get(f"/scene3d/{CONV}/scene.glb", headers={"X-Oprel-Token": token})
    assert glb.status_code == 200 and glb.headers["content-type"] == "model/gltf-binary"
    assert glb.headers["x-scene-version"] == "1"
    loaded = trimesh.load(io.BytesIO(glb.content), file_type="glb")
    assert sorted(loaded.graph.nodes_geometry) == ["cyl", "rect"]


def test_bad_call_changes_nothing_and_tells_the_model_why(client, token, monkeypatch):
    _, model, events, _ = _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_add_objects", objects=[
            {"id": "b", "shape": "box", "params": {"width": 1}},
        ])]),
        AssistantTurn(content="", tool_calls=[_call("scene_add_objects", n=1, objects=[
            {"id": "b", "shape": "box", "params": {"width": 1, "height": 1, "depth": 1}},
        ])]),
        AssistantTurn(content="Fixed."),
    ])
    first, second = _results(events)
    assert not first["ok"]
    assert "Nothing was changed" in first["data"]["error"] and "height" in first["data"]["error"]
    assert second["ok"] and second["data"]["scene_version"] == 1  # the failed call didn't bump the version


def test_update_and_delete_keep_dependents_in_place(client, token, monkeypatch):
    _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_add_objects", objects=CYL_AND_RECT)]),
        AssistantTurn(content="ok"),
    ])
    _, _, events, _ = _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_update_objects", updates=[
            {"id": "cyl", "params": {"radius": 2}, "color": "red"},
        ])]),
        AssistantTurn(content="", tool_calls=[_call("scene_delete_objects", n=1, ids=["cyl"])]),
        AssistantTurn(content="", tool_calls=[_call("scene_get_scene", n=2)]),
        AssistantTurn(content="done"),
    ])
    updated, deleted, listed = _results(events)
    assert updated["ok"]
    cyl = next(o for o in updated["data"]["objects"] if o["id"] == "cyl")
    assert cyl["size"] == [4.0, 2.0, 4.0]  # radius merged, height kept

    assert deleted["ok"] and deleted["data"]["deleted"] == ["cyl"]
    (rect,) = listed["data"]["objects"]
    # rect was placed next to cyl (radius 2): it keeps that spot as a fixed position
    assert rect["id"] == "rect" and "placement" not in rect
    assert rect["min"] == [2.2, 0.0, -0.5]


def test_many_writes_in_one_turn_never_ask_for_approval(client, token, monkeypatch):
    calls = [_call("scene_add_objects", n=i, objects=[
        {"id": f"b{i}", "shape": "box", "params": {"width": 0.1, "height": 0.1, "depth": 0.1}, "position": [i, 0, 0]},
    ]) for i in range(12)]
    _, _, events, _ = _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=calls),
        AssistantTurn(content="12 boxes"),
    ])
    assert len(_results(events)) == 12 and all(r["ok"] for r in _results(events))
    assert not [e for e in events if e["type"] == "approval_required"]


def test_forgotten_parents_are_put_in_the_new_group(client, token, monkeypatch):
    # Qwen3 made the group, then forgot "parent" on every part.
    _, model, events, _ = _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_add_objects", objects=[
            {"id": "table", "shape": "group", "position": [2, 0, 0]},
            {"id": "legs", "shape": "cylinder", "params": {"radius": 0.03, "height": 0.7},
             "pattern": {"type": "grid", "count": 2, "rows": 2, "spacing": [1.0, 0.4]}},
            {"id": "top", "shape": "box", "params": {"width": 1.2, "height": 0.03, "depth": 0.6}, "position": [0, 0.7, 0]},
        ])]),
        AssistantTurn(content="done"),
    ])
    (result,) = _results(events)
    assert result["data"]["notes"] == ["Put 'legs', 'top' in group 'table'."]
    assert result["data"]["warnings"] == []
    table = next(o for o in result["data"]["objects"] if o["id"] == "table")
    assert table["min"] == [1.4, 0.0, -0.3]  # the parts moved with the group
    saved = client.get(f"/scene3d/{CONV}", headers={"X-Oprel-Token": token}).json()["objects"]
    assert {o["id"]: o.get("parent") for o in saved} == {"table": None, "legs": "table", "top": "table"}


def test_announced_fix_is_followed_through_once(client, token, monkeypatch):
    # Qwen3-8B: "...the legs extend beyond the table top. Let me update the scene." and then stopped.
    _, model, events, text = _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_add_objects", objects=CYL_AND_RECT)]),
        AssistantTurn(content="The plate is too far away. Let me update the scene."),
        AssistantTurn(content="", tool_calls=[_call("scene_update_objects", n=1, updates=[
            {"id": "rect", "placement": {"relative_to": "cyl", "side": "right", "gap": 0}},
        ])]),
        AssistantTurn(content="Moved the plate next to the cylinder."),
    ])
    assert [r["tool"] for r in _results(events)] == ["scene_add_objects", "scene_update_objects"]
    assert all(r["ok"] for r in _results(events))
    assert "Let me update the scene." in text and text.endswith("Moved the plate next to the cylinder.")
    nudge = model.seen[2]["messages"][-1]
    assert nudge["role"] == "user" and "Make that tool call now" in nudge["content"]


def test_follow_through_nudge_happens_at_most_once(client, token, monkeypatch):
    _, model, events, text = _chat(client, token, monkeypatch, [
        AssistantTurn(content="I'll draw it now."),
        AssistantTurn(content="I will draw it now."),
    ])
    assert len(model.seen) == 2 and not _results(events)
    assert text.endswith("I will draw it now.")


def test_questions_are_not_nudged(client, token, monkeypatch):
    _, model, _, text = _chat(client, token, monkeypatch, [
        AssistantTurn(content="Should it be round or square? I'll draw it once you tell me."),
    ])
    assert len(model.seen) == 1


def test_let_me_know_is_not_an_announced_action(client, token, monkeypatch):
    # Qwen3-8B ended with this and got nudged into repeating itself.
    _, model, _, _ = _chat(client, token, monkeypatch, [
        AssistantTurn(content="The table is done. Let me know if you'd like any further adjustments!"),
    ])
    assert len(model.seen) == 1


def test_older_drawing_steps_are_compacted_for_the_model(client, token, monkeypatch):
    _, model, events, _ = _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_add_objects", plan="p", objects=CYL_AND_RECT)]),
        AssistantTurn(content="", tool_calls=[_call("scene_update_objects", n=1, updates=[{"id": "cyl", "color": "red"}])]),
        AssistantTurn(content="Done."),
    ])
    assert all(r["ok"] for r in _results(events))
    last_view = model.seen[2]["messages"]
    first_call = next(m for m in last_view if m.get("tool_calls"))
    assert first_call["tool_calls"][0]["function"]["arguments"] == "{}"
    first_result = json.loads(next(m for m in last_view if m["role"] == "tool")["content"])
    assert set(first_result) <= {"ok", "summary", "warnings", "notes", "scene_version"} and "objects" not in first_result
    latest_result = json.loads(last_view[-1]["content"])
    assert "objects" in latest_result  # the latest round stays complete
    # The stored history itself is untouched.
    assert "objects" in json.loads(model.seen[1]["messages"][-1]["content"])


def test_running_out_of_context_after_the_work_ends_with_a_summary(client, token, monkeypatch):
    from oprel.server.tools.adapters import ContextFullError

    class FillsUp(ScriptedModel):
        async def complete(self, messages, tools, params):
            if len(self.seen) == 1:
                self.seen.append({"messages": messages, "tools": []})
                raise ContextFullError("no room")
            return await super().complete(messages, tools, params)

    model = FillsUp([AssistantTurn(content="", tool_calls=[_call("scene_add_objects", objects=CYL_AND_RECT)])])
    monkeypatch.setattr(loop, "resolve_adapter", lambda name: model)
    r = client.post("/v1/chat/completions", headers={"X-Oprel-Token": token},
                    json={"model": "m", "stream": False, "enable_tools": ["scene3d"], "conversation_id": CONV,
                          "messages": [{"role": "user", "content": "draw"}]})
    body = r.json()
    text = body["choices"][0]["message"]["content"]
    assert text.startswith("Done. Added cylinder 'cyl', rectangle 'rect'.") and "ran out of context" in text
    assert not [e for e in body["oprel_events"] if e["type"] == "error"]


def test_drawing_runs_at_a_capped_temperature(client, token, monkeypatch):
    model = ScriptedModel([AssistantTurn(content="hi")])
    monkeypatch.setattr(loop, "resolve_adapter", lambda name: model)
    client.post("/v1/chat/completions", headers={"X-Oprel-Token": token},
                json={"model": "m", "stream": False, "enable_tools": ["scene3d"], "temperature": 0.7,
                      "messages": [{"role": "user", "content": "draw"}]})
    assert model.seen[0]["temperature"] == 0.3


# The user's table (chat_095ab5381eaa): top 3 cm above the legs, legs sticking out past the top.
FLOATING_TOP_TABLE = [
    {"shape": "group", "id": "table"},
    {"shape": "cylinder", "id": "leg", "params": {"radius": 0.025, "height": 0.69},
     "pattern": {"type": "radial", "count": 5, "radius": 0.35}, "parent": "table", "position": [0, 0, 0]},
    {"shape": "box", "id": "top", "params": {"width": 1.2, "height": 0.03, "depth": 0.6},
     "position": [0, 0.72, 0], "parent": "table"},
]


def test_engine_repairs_a_floating_top_and_legs_sticking_out(client, token, monkeypatch):
    _, _, events, _ = _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_add_objects", objects=FLOATING_TOP_TABLE)]),
        AssistantTurn(content="done"),
    ])
    (result,) = _results(events)
    data = result["data"]
    assert data["warnings"] == []
    assert data["notes"][0] == "Lowered 'top' 0.030 m so it rests on the part below it."
    assert data["notes"][1].startswith("Moved the 5 copies of 'leg' in so they fit under 'top' (radius 0.35 ->")
    by_id = {o["id"]: o for o in data["objects"]}
    assert by_id["top"]["min"][1] == by_id["leg"]["max"][1] == 0.69  # resting on the legs
    assert -0.3 <= by_id["leg"]["min"][2] and by_id["leg"]["max"][2] <= 0.3  # inside the top's depth
    # The fixes are saved in the recipe, so the panel shows the repaired table.
    saved = {o["id"]: o for o in client.get(f"/scene3d/{CONV}", headers={"X-Oprel-Token": token}).json()["objects"]}
    assert saved["top"]["position"] == [0, 0.69, 0] and saved["leg"]["pattern"]["radius"] < 0.3


def test_grid_legs_are_pulled_in_and_intended_gaps_are_left_alone(client, token, monkeypatch):
    _, _, events, _ = _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_add_objects", objects=[
            {"id": "table", "shape": "group"},
            {"id": "legs", "parent": "table", "shape": "cylinder", "params": {"radius": 0.03, "height": 0.7},
             "pattern": {"type": "grid", "count": 2, "rows": 2, "spacing": [1.6, 0.8]}},
            {"id": "top", "parent": "table", "shape": "box", "params": {"width": 1.2, "height": 0.03, "depth": 0.6},
             "placement": {"relative_to": "legs", "side": "top"}},
            {"id": "lamp", "shape": "sphere", "params": {"radius": 0.1}, "position": [0, 1.5, 0]},
        ])]),
        AssistantTurn(content="done"),
    ])
    data = _results(events)[0]["data"]
    by_id = {o["id"]: o for o in data["objects"]}
    assert -0.6 <= by_id["legs"]["min"][0] and by_id["legs"]["max"][0] <= 0.6
    assert by_id["top"]["min"][1] == 0.7
    assert by_id["lamp"]["min"][1] == 1.5  # a top-level object hanging above is not 'fixed'
    assert any("'lamp' floats" in w for w in data["warnings"])


# Verbatim from Qwen3-8B ("dining table with four legs and chairs around it"): no ids, and the
# chairs refer to "table". It failed 8 times in a row before ids were named after shapes.
DINING_CALL = {"objects": [
    {"shape": "table", "params": {"width": 1.8, "depth": 1.2, "height": 0.8, "top": "rectangle", "legs": 4,
                                  "top_thickness": 0.05, "leg_radius": 0.05}, "position": [0, 0, 0]},
    {"shape": "chair", "pattern": {"type": "around", "count": 4, "relative_to": "table"}, "color": "brown"},
]}


def test_objects_without_ids_are_named_after_their_shape(client, token, monkeypatch):
    _, _, events, _ = _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_add_objects", **DINING_CALL)]),
        AssistantTurn(content="done"),
    ])
    (result,) = _results(events)
    assert result["ok"], result["data"]
    by_id = {o["id"]: o for o in result["data"]["objects"]}
    assert set(by_id) == {"table", "chair"} and by_id["chair"]["copies"] == 4
    assert result["data"]["warnings"] == []


def test_a_shape_name_reference_finds_the_only_object_of_that_shape(client, token, monkeypatch):
    _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_add_objects", objects=[
            {"id": "dining_table", "shape": "table"},
        ])]),
        AssistantTurn(content="ok"),
    ])
    _, _, events, _ = _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_add_objects", objects=[
            {"shape": "chair", "pattern": {"type": "around", "relative_to": "table", "count": 2}},
        ])]),
        AssistantTurn(content="ok"),
    ])
    (result,) = _results(events)
    assert result["ok"] and result["data"]["notes"] == ["Read 'table' as 'dining_table' for 'chair'."]


def test_a_failing_call_repeated_unchanged_stops_the_turn(client, token, monkeypatch):
    bad = {"objects": [{"id": "b", "shape": "box", "params": {"width": 1}}]}
    _, model, events, text = _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_add_objects", **bad)]),
        AssistantTurn(content="", tool_calls=[_call("scene_add_objects", n=1, **bad)]),
        AssistantTurn(content="", tool_calls=[_call("scene_add_objects", n=2, **bad)]),
        AssistantTurn(content="never reached"),
    ])
    results = _results(events)
    assert len(results) == 3 and not any(r["ok"] for r in results)
    assert "already failed" in results[1]["data"]["error"]
    assert len(model.seen) == 3  # stopped instead of asking the model again
    assert text.startswith("I couldn't finish") and "height" in text
    assert events[-1]["finish_reason"] == "error"


def test_builtin_is_only_bound_when_named(client, token, monkeypatch):
    # "*" means "all connected apps": nothing is connected, so there are no tools at all.
    r, _, _, _ = _chat(client, token, monkeypatch, [AssistantTurn(content="hi")], enable=("*",))
    assert r.status_code == 400


def test_catalog_marks_it_built_in_and_connect_is_refused(client, token):
    catalog = client.get("/integrations/catalog", headers={"X-Oprel-Token": token}).json()
    entry = next(e for e in catalog if e["id"] == "scene3d")
    assert entry["requires_connection"] is False
    r = client.post("/integrations/scene3d/connect", headers={"X-Oprel-Token": token}, json={"values": {}})
    assert r.status_code == 400 and "built in" in r.json()["detail"]


def test_empty_scene_glb_is_204_and_clear_works(client, token, monkeypatch):
    assert client.get("/scene3d/nothing-here/scene.glb", headers={"X-Oprel-Token": token}).status_code == 204
    _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_add_objects", objects=CYL_AND_RECT)]),
        AssistantTurn(content="ok"),
    ])
    cleared = client.delete(f"/scene3d/{CONV}", headers={"X-Oprel-Token": token}).json()
    assert cleared["objects"] == [] and cleared["version"] == 2
    assert client.get(f"/scene3d/{CONV}/scene.glb", headers={"X-Oprel-Token": token}).status_code == 204
