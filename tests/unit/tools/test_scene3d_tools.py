"""
3D scene toolset end-to-end: the chat tool loop with a scripted model, the scene_* tools,
built-in binding (no connection, auto-approved, no write cap) and the /scene3d routes.
"""

import io
import json

import pytest
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
                          "temperature": params.temperature, "thinking": params.thinking})
        # Out of script (e.g. after a warnings reminder): answer plainly, like a model keeping its result.
        return self.turns.pop(0) if self.turns else AssistantTurn(content="Kept as it is.")


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
    assert model.seen[0]["thinking"] is False  # drawing skips the reasoning step on local models


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


def test_several_things_on_one_table_are_spread_out(client, token, monkeypatch):
    # Verbatim from Qwen3-8B: everything placed {"side": "top"} of the table, so all in its centre.
    _, _, events, _ = _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_add_objects", objects=[
            {"shape": "table", "id": "table", "params": {"width": 1.2, "depth": 0.6}},
            {"shape": "bowl", "id": "bowl", "placement": {"relative_to": "table", "side": "top"}},
            {"shape": "cup", "id": "cup1", "placement": {"relative_to": "table", "side": "top"}},
            {"shape": "cup", "id": "cup2", "placement": {"relative_to": "table", "side": "top"}},
            {"shape": "wine_glass", "id": "wine_glass", "placement": {"relative_to": "table", "side": "top"}},
            {"shape": "plant_pot", "id": "plant_pot", "placement": {"relative_to": "table", "side": "right", "gap": 0.2}},
        ])]),
        AssistantTurn(content="done"),
    ])
    data = _results(events)[0]["data"]
    assert any(n.startswith("Spread 'bowl', 'cup1', 'cup2', 'wine_glass'") for n in data["notes"])
    on_top = sorted((o for o in data["objects"] if o["id"] in ("bowl", "cup1", "cup2", "wine_glass")),
                    key=lambda o: o["min"][0])
    for a, b in zip(on_top, on_top[1:]):
        assert a["max"][0] <= b["min"][0]  # side by side, not inside each other
    assert all(-0.6 <= o["min"][0] and o["max"][0] <= 0.6 and o["min"][1] == 0.75 for o in on_top)


def test_follow_up_messages_see_what_is_already_drawn(client, token, monkeypatch):
    # The web UI sends only chat text, so without the scene in the system prompt Qwen3-8B
    # answered "reduce the bottom radius" by drawing a second vase.
    _, first, _, _ = _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_add_objects", objects=[
            {"id": "vase", "shape": "vase", "params": {"height": 0.4}},
        ])]),
        AssistantTurn(content="done"),
    ])
    assert "The 3D scene is empty." in first.seen[0]["messages"][0]["content"]
    _, second, _, _ = _chat(client, token, monkeypatch, [AssistantTurn(content="ok")])
    system = second.seen[0]["messages"][0]["content"]
    assert "Already in the 3D scene" in system and "do NOT add it again" in system
    assert '- vase: vase {"height":0.4}' in system and "max [" in system


def test_relisting_an_existing_object_updates_it_instead_of_failing(client, token, monkeypatch):
    # Verbatim from Qwen3-8B for "add a small table and put the vase on it": the vase is listed
    # again with its id. It used to fail and the model ended up adding a second vase.
    _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_add_objects", objects=[
            {"shape": "vase", "params": {"base_radius": 0.08, "radius": 0.2, "neck_radius": 0.15, "top_radius": 0.3},
             "position": [0, 0, 0]},
        ])]),
        AssistantTurn(content="ok"),
    ])
    _, _, events, _ = _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_add_objects", objects=[
            {"shape": "table", "id": "table", "params": {"width": 1.2, "depth": 1.2, "height": 0.8}},
            {"shape": "vase", "id": "vase", "params": {"base_radius": 0.05},
             "placement": {"relative_to": "table", "side": "top", "align": "center"}},
        ])]),
        AssistantTurn(content="done"),
    ])
    (result,) = _results(events)
    assert result["ok"], result["data"]
    assert result["data"]["notes"][0] == "'vase' was already there, so it was updated instead of adding a copy."
    saved = client.get(f"/scene3d/{CONV}", headers={"X-Oprel-Token": token}).json()["objects"]
    assert [o["id"] for o in saved] == ["vase", "table"]  # still one vase
    vase = saved[0]
    assert vase["params"] == {"base_radius": 0.05, "radius": 0.2, "neck_radius": 0.15, "top_radius": 0.3}
    assert "position" not in vase and vase["placement"]["relative_to"] == "table"
    by_id = {o["id"]: o for o in result["data"]["objects"]}
    assert by_id["vase"]["min"][1] == 0.8  # on the table


def test_a_new_vase_on_the_old_vases_spot_is_an_edit(client, token, monkeypatch):
    # Verbatim from Qwen3-8B for "bottom increase the radius to 2, top radius 1": no id, same spot.
    _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_add_objects", objects=[
            {"shape": "vase", "params": {"base_radius": 0.08, "radius": 0.18, "neck_radius": 0.12, "top_radius": 0.25,
                                         "height": 1.2}, "position": [0, 0, 0]},
        ])]),
        AssistantTurn(content="ok"),
    ])
    _, _, events, _ = _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_add_objects", objects=[
            {"shape": "vase", "params": {"base_radius": 2, "radius": 0.18, "neck_radius": 0.12, "top_radius": 1,
                                         "height": 1.2}, "position": [0, 0, 0]},
        ])]),
        AssistantTurn(content="done"),
    ])
    data = _results(events)[0]["data"]
    assert data["notes"][0] == "'vase' was already there, so it was updated instead of adding a copy."
    saved = client.get(f"/scene3d/{CONV}", headers={"X-Oprel-Token": token}).json()["objects"]
    assert [o["id"] for o in saved] == ["vase"] and saved[0]["params"]["base_radius"] == 2
    # ...and the absurd result is pointed out to the model.
    assert any("'vase' is 4.00 m wide but only 1.20 m tall" in w and "base_radius 2 m" in w for w in data["warnings"])


def test_absurd_proportions_come_with_numbers_that_keep_the_ratios(client, token, monkeypatch):
    _, _, events, _ = _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_add_objects", objects=[
            {"shape": "vase", "params": {"base_radius": 2, "radius": 1, "neck_radius": 0.5, "top_radius": 1,
                                         "height": 1.2}},
        ])]),
        AssistantTurn(content="done"),
    ])
    (warning,) = _results(events)[0]["data"]["warnings"]
    suggested = json.loads(warning.split("would be params ", 1)[1].rstrip("."))
    assert suggested == {"base_radius": 0.3, "radius": 0.15, "neck_radius": 0.075, "top_radius": 0.15}


def test_a_reply_that_ignores_warnings_gets_one_reminder(client, token, monkeypatch):
    _, model, events, text = _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_add_objects", objects=[
            {"id": "vase", "shape": "vase", "params": {"base_radius": 2, "top_radius": 1, "height": 1.2}},
        ])]),
        AssistantTurn(content="Done, the vase has a 2 m base."),
        AssistantTurn(content="", tool_calls=[_call("scene_update_objects", n=1, updates=[
            {"id": "vase", "params": {"base_radius": 0.3, "top_radius": 0.15}},
        ])]),
        AssistantTurn(content="Made the base twice the top at a vase-like size."),
    ])
    results = _results(events)
    assert [r["ok"] for r in results] == [True, True] and results[1]["data"]["warnings"] == []
    reminder = model.seen[2]["messages"][-1]
    assert reminder["role"] == "user" and "warnings you have not dealt with" in reminder["content"]
    assert "4.00 m wide" in reminder["content"]
    assert "2 m base" not in text  # the premature reply is not shown
    assert text == "Made the base twice the top at a vase-like size."


def test_the_same_warnings_are_not_pointed_out_twice(client, token, monkeypatch):
    _, model, events, text = _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_add_objects", objects=[
            {"id": "ball", "shape": "sphere", "params": {"radius": 0.2}, "position": [0, 2, 0]},
        ])]),
        AssistantTurn(content="Done."),
        AssistantTurn(content="The user asked for a floating ball, so I kept it."),
    ])
    assert len(model.seen) == 3
    assert text == "The user asked for a floating ball, so I kept it."


def test_an_update_that_changes_nothing_is_flagged(client, token, monkeypatch):
    # Seen with Qwen3-8B: for "bottom radius to 2" it resent the old values and said it was done.
    vase = {"shape": "vase", "params": {"base_radius": 0.02, "top_radius": 0.06, "height": 0.25}, "position": [0, 0, 0]}
    _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_add_objects", objects=[vase])]),
        AssistantTurn(content="ok"),
    ])
    _, model, events, _ = _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_add_objects", objects=[vase])]),
        AssistantTurn(content="Updated the vase."),
        AssistantTurn(content="", tool_calls=[_call("scene_update_objects", n=1, updates=[
            {"id": "vase", "params": {"base_radius": 0.04}}, {"id": "vase", "position": [0, 0, 0]},
        ])]),
        AssistantTurn(content="Doubled the base."),
    ])
    first, second = _results(events)
    assert first["data"]["warnings"][0].startswith("'vase' did not change")
    assert "did not change" in model.seen[2]["messages"][-1]["content"]  # the reminder carried it
    # The real change is not flagged; the no-op one in the same call is ([0,0,0] = where it was).
    assert [w for w in second["data"]["warnings"] if "did not change" in w] == [
        "'vase' did not change: every value you sent is what it already had. Apply what the user asked "
        "(new numbers) or tell them you didn't change it."]


def test_another_vase_somewhere_else_is_still_added(client, token, monkeypatch):
    _, _, events, _ = _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_add_objects", objects=[
            {"shape": "vase", "position": [0, 0, 0]}, {"shape": "vase", "position": [0.5, 0, 0]},
        ])]),
        AssistantTurn(content="ok"),
    ])
    assert [o["id"] for o in _results(events)[0]["data"]["objects"]] == ["vase", "vase2"]


def test_reusing_an_id_for_a_different_shape_is_still_an_error(client, token, monkeypatch):
    _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_add_objects", objects=[{"id": "a", "shape": "vase"}])]),
        AssistantTurn(content="ok"),
    ])
    _, _, events, _ = _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_add_objects", objects=[{"id": "a", "shape": "chair"}])]),
        AssistantTurn(content="ok"),
    ])
    (result,) = _results(events)
    assert not result["ok"] and "already a vase" in result["data"]["error"]


def test_a_floating_vase_is_set_down(client, token, monkeypatch):
    # Verbatim from Qwen3-8B: a 1.5 m vase "positioned" by its centre, 0.75 m in the air.
    _, _, events, _ = _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_add_objects", objects=[
            {"shape": "vase", "params": {"height": 1.5, "radius": 0.2, "neck_radius": 0.1}, "position": [0, 0.75, 0]},
        ])]),
        AssistantTurn(content="done"),
    ])
    data = _results(events)[0]["data"]
    assert data["notes"] == ["Lowered 'vase' 0.750 m onto the ground (position is the bottom-centre)."]
    assert data["objects"][0]["min"][1] == 0.0 and data["warnings"] == []


def test_a_cup_floating_above_a_table_lands_on_the_table_and_hanging_primitives_stay(client, token, monkeypatch):
    _, _, events, _ = _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_add_objects", objects=[
            {"id": "table", "shape": "table"},
            {"id": "cup", "shape": "cup", "position": [0, 1.2, 0]},
            {"id": "light", "shape": "sphere", "params": {"radius": 0.1}, "position": [2, 2, 0]},
        ])]),
        AssistantTurn(content="done"),
    ])
    by_id = {o["id"]: o for o in _results(events)[0]["data"]["objects"]}
    assert by_id["cup"]["min"][1] == 0.75  # on the table top
    assert by_id["light"]["min"][1] == 2.0  # a plain shape may hang on purpose


def test_relative_to_the_scene_means_just_put_it_in_the_scene(client, token, monkeypatch):
    _, _, events, _ = _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_add_objects", objects=[
            {"shape": "vase", "placement": {"relative_to": "scene", "side": "center"}},
        ])]),
        AssistantTurn(content="done"),
    ])
    (result,) = _results(events)
    assert result["ok"] and "'scene' is not an object" in result["data"]["notes"][0]


def test_two_separate_identical_objects_get_no_axis_warning(client, token, monkeypatch):
    _, _, events, _ = _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_add_objects", objects=[
            {"id": "shelf", "shape": "box", "params": {"width": 0.5, "height": 1, "depth": 0.3}},
            {"id": "v1", "shape": "vase", "placement": {"relative_to": "shelf", "side": "top"}},
            {"id": "v2", "shape": "vase", "position": [1, 0, 0]},
        ])]),
        AssistantTurn(content="done"),
    ])
    assert _results(events)[0]["data"]["warnings"] == []


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


def test_cuts_through_the_tools_and_in_the_follow_up_summary(client, token, monkeypatch):
    plate = {"id": "plate", "shape": "box", "params": {"width": 0.4, "height": 0.02, "depth": 0.3},
             "cuts": [{"shape": "cylinder", "params": {"radius": 0.05}, "position": [2, 0, 0]}]}
    _, _, events, _ = _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_add_objects", objects=[plate])]),
        AssistantTurn(content="", tool_calls=[_call("scene_update_objects", n=1, updates=[
            {"id": "plate", "cuts": [{"shape": "cylinder", "params": {"radius": 0.05}}]}])]),
        AssistantTurn(content="Drilled a hole through the middle of the plate."),
    ])
    missed, fixed = _results(events)
    # A cut that misses is a warning the model can act on (and it did).
    assert missed["ok"] and any("misses the object" in w for w in missed["data"]["warnings"])
    assert fixed["ok"] and not fixed["data"].get("warnings")

    # The next message sees the cuts, so "add another hole" can resend the whole list.
    _, model, _, _ = _chat(client, token, monkeypatch, [AssistantTurn(content="ok")])
    system = model.seen[0]["messages"][0]["content"]
    assert '- plate: box' in system and 'cuts [{"shape":"cylinder","params":{"radius":0.05}}]' in system

    # Removing the cuts gives the solid plate back.
    _, _, events, _ = _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_update_objects", updates=[{"id": "plate", "remove": ["cuts"]}])]),
        AssistantTurn(content="ok"),
    ])
    (result,) = _results(events)
    assert result["ok"]
    scene = client.get(f"/scene3d/{CONV}", headers={"X-Oprel-Token": token}).json()
    assert "cuts" not in scene["objects"][0]


def test_update_that_drops_earlier_cuts_is_flagged(client, token, monkeypatch):
    hole = {"shape": "cylinder", "params": {"radius": 0.03}}
    corners = {"shape": "cylinder", "params": {"radius": 0.004},
               "pattern": {"type": "grid", "count": 2, "rows": 2, "spacing": [0.24, 0.24]}}
    _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_add_objects", objects=[
            {"id": "plate", "shape": "box", "params": {"width": 0.3, "height": 0.01, "depth": 0.3}, "cuts": [hole]}])]),
        AssistantTurn(content="ok"),
    ])
    # "add 4 screw holes": only the new holes sent, then fixed after the warning.
    _, _, events, _ = _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_update_objects", updates=[{"id": "plate", "cuts": [corners]}])]),
        AssistantTurn(content="", tool_calls=[_call("scene_update_objects", n=1, updates=[
            {"id": "plate", "cuts": [hole, corners]}])]),
        AssistantTurn(content="Added 4 screw holes."),
    ])
    dropped, fixed = _results(events)
    assert any("replaced 1 earlier cut(s) (cylinder)" in w for w in dropped["data"]["warnings"])
    assert not fixed["data"]["warnings"]

    # Clearing the cuts on purpose is not flagged.
    _, _, events, _ = _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_update_objects", updates=[{"id": "plate", "remove": ["cuts"]}])]),
        AssistantTurn(content="ok"),
    ])
    assert not _results(events)[0]["data"]["warnings"]


def test_solid_object_named_like_a_hole_is_flagged(client, token, monkeypatch):
    _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_add_objects", objects=[
            {"id": "plate", "shape": "rectangle", "params": {"width": 0.3, "depth": 0.3, "thickness": 0.01}}])]),
        AssistantTurn(content="ok"),
    ])
    _, _, events, _ = _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_add_objects", objects=[
            {"shape": "box", "params": {"width": 0.1, "height": 0.1, "depth": 0.1}, "label": "screw_holes"}])]),
        AssistantTurn(content="ok"),
    ])
    (result,) = _results(events)
    # Overlapping the plate, so it becomes a hole in it (right through: it spans the plate's thickness).
    assert any("a hole in 'plate' (right through)" in n for n in result["data"]["notes"])
    saved = {o["id"]: o for o in client.get(f"/scene3d/{CONV}", headers={"X-Oprel-Token": token}).json()["objects"]}
    assert set(saved) == {"plate"} and len(saved["plate"]["cuts"]) == 1


def test_solid_window_inside_a_wall_is_flagged_but_a_door_on_its_own_is_not(client, token, monkeypatch):
    _, _, events, _ = _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_add_objects", objects=[
            {"id": "wall", "shape": "box", "params": {"width": 4, "height": 2.5, "depth": 0.2}},
            {"id": "window", "shape": "box", "params": {"width": 1, "height": 1.2, "depth": 0.2}, "position": [1, 0.9, 0]},
            {"id": "door", "shape": "box", "params": {"width": 0.9, "height": 2, "depth": 0.04}, "position": [0, 0, 2]},
        ])]),
        AssistantTurn(content="ok"),
    ])
    data = _results(events)[0]["data"]
    assert any("Made 'window' a hole in 'wall' (right through)" in n for n in data["notes"])
    saved = {o["id"]: o for o in client.get(f"/scene3d/{CONV}", headers={"X-Oprel-Token": token}).json()["objects"]}
    assert "window" not in saved and "depth" not in saved["wall"]["cuts"][0]["params"]  # through the wall
    assert "door" in saved  # a door panel on its own is a real thing


def test_plate_between_two_cylinders_follows_and_survives_deleting_one(client, token, monkeypatch):
    big = {"id": "big", "shape": "cylinder", "params": {"radius": 0.2, "height": 1.0}}
    thin = {"id": "thin", "shape": "cylinder", "params": {"radius": 0.07, "height": 1.1},
            "placement": {"relative_to": "big", "side": "left", "gap": 0.15}}
    plate = {"id": "plate", "shape": "box", "params": {"height": 0.2, "depth": 0.02},
             "placement": {"between": ["thin", "big"]}}
    _, _, events, _ = _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_add_objects", objects=[big, thin, plate])]),
        AssistantTurn(content="ok"),
    ])
    (result,) = _results(events)
    assert result["ok"] and not result["data"]["warnings"]

    # "more space": only the gap changes; the plate still reaches both centres.
    _, model, events, _ = _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_update_objects", updates=[
            {"id": "thin", "placement": {"relative_to": "big", "side": "left", "gap": 0.4}}])]),
        AssistantTurn(content="ok"),
    ])
    system = model.seen[0]["messages"][0]["content"]
    assert "- plate: box" in system and "between thin and big" in system
    assert _results(events)[0]["ok"]
    scene = client.get(f"/scene3d/{CONV}", headers={"X-Oprel-Token": token}).json()
    assert scene["objects"][2]["placement"] == {"between": ["thin", "big"]}

    # Deleting one cylinder keeps the plate exactly where and how big it was.
    _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_delete_objects", ids=["thin"])]),
        AssistantTurn(content="ok"),
    ])
    plate_after = next(o for o in client.get(f"/scene3d/{CONV}", headers={"X-Oprel-Token": token}).json()["objects"]
                       if o["id"] == "plate")
    assert "placement" not in plate_after
    assert plate_after["params"]["width"] == pytest.approx(0.67) and plate_after["position"] == [-0.335, 0.0, 0.0]


def test_summary_shows_how_objects_are_placed():
    from oprel.integrations.scene3d.tools import _placement_words
    assert _placement_words({"relative_to": "big", "side": "left", "gap": 0.15}) == " left of big gap 0.15"
    assert _placement_words({"between": ["thin", "big"], "vertical": "top"}) == " between thin and big"
    assert _placement_words(None) == ""


def test_new_warnings_after_a_fix_get_a_second_reminder(client, token, monkeypatch):
    # Live Qwen3-8B: fixing the thin cylinder raised "the plate touches only one", then it said "done".
    big = {"id": "big", "shape": "cylinder", "params": {"radius": 0.2, "height": 1}}
    thin = {"id": "thin", "shape": "cylinder", "params": {"radius": 0.07, "height": 1.1}, "position": [-0.15, 0, 0]}
    # Off the midpoint, so the midpoint repair doesn't apply and the warning path is exercised.
    plate = {"id": "plate", "shape": "box", "params": {"width": 0.14, "height": 0.2, "depth": 0.02},
             "position": [-0.03, 0, 0], "label": "Joining plate"}
    _, model, events, text = _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_add_objects", objects=[big, thin, plate])]),
        AssistantTurn(content="Done."),
        AssistantTurn(content="", tool_calls=[_call("scene_update_objects", n=1, updates=[
            {"id": "thin", "placement": {"relative_to": "big", "side": "left", "gap": 0.15}}])]),
        AssistantTurn(content="Done, they are joined."),
        AssistantTurn(content="", tool_calls=[_call("scene_update_objects", n=2, updates=[
            {"id": "plate", "placement": {"between": ["thin", "big"]}}])]),
        AssistantTurn(content="The plate now joins both cylinders."),
    ])
    first, second, third = _results(events)
    assert any("stands inside" in w for w in first["data"]["warnings"])
    assert any("touches only 'big'" in w for w in second["data"]["warnings"])
    assert third["data"]["warnings"] == []
    assert text == "The plate now joins both cylinders."
    # Moving it "between" dropped the old 14 cm width, so it reaches from centre to centre.
    plate_now = next(o for o in third["data"]["objects"] if o["id"] == "plate")
    assert plate_now["min"][0] == pytest.approx(-0.42) and plate_now["max"][0] == pytest.approx(0)


def test_plate_drawn_at_the_midpoint_becomes_between_and_follows_the_cylinders(client, token, monkeypatch):
    # Live Qwen3-8B, unlabelled: thin cylinder 15 cm from the big one's centre, plate at the midpoint.
    objects = [{"shape": "cylinder", "params": {"radius": 0.2, "height": 1}, "position": [0, 0, 0]},
               {"shape": "cylinder", "params": {"radius": 0.07, "height": 1.1}, "position": [-0.15, 0, 0]},
               {"shape": "box", "params": {"width": 0.15, "height": 0.2, "depth": 0.02}, "position": [-0.075, 0, 0]}]
    _, _, events, _ = _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_add_objects", objects=objects)]),
        AssistantTurn(content="", tool_calls=[_call("scene_update_objects", n=1, updates=[
            {"id": "cylinder2", "placement": {"relative_to": "cylinder", "side": "left", "gap": 0.15}}])]),
        AssistantTurn(content="Done."),
    ])
    first, second = _results(events)
    assert any("Placed 'box' between" in n for n in first["data"]["notes"])
    assert second["data"]["warnings"] == []
    # "more spacing" as {"gap": 0.4} alone keeps side and reference; the plate still joins both.
    _, _, events, _ = _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_update_objects", updates=[
            {"id": "cylinder2", "placement": {"gap": 0.4}}])]),
        AssistantTurn(content="Done."),
    ])
    (result,) = _results(events)
    assert result["ok"] and result["data"]["warnings"] == []
    scene = client.get(f"/scene3d/{CONV}", headers={"X-Oprel-Token": token}).json()
    by_id = {o["id"]: o for o in scene["objects"]}
    assert by_id["cylinder2"]["placement"] == {"relative_to": "cylinder", "side": "left", "gap": 0.4}
    assert by_id["box"]["placement"] == {"between": ["cylinder", "cylinder2"]}
    assert by_id["box"]["params"] == {"height": 0.2, "depth": 0.02}


def test_coffee_table_between_two_sofas_is_left_alone():
    from oprel.integrations.scene3d.engine import compile_scene
    from oprel.integrations.scene3d.repair import join_midpoint_boxes
    objects = [{"id": "s1", "shape": "sofa", "position": [-1.5, 0, 0]}, {"id": "s2", "shape": "sofa", "position": [1.5, 0, 0]},
               {"id": "coffee", "shape": "box", "params": {"width": 1, "height": 0.4, "depth": 0.6}, "position": [0, 0, 0]}]
    assert join_midpoint_boxes(objects, compile_scene(objects)) == []


THREE_PARTS = [
    {"id": "big", "shape": "cylinder", "params": {"radius": 0.2, "height": 1}},
    {"id": "thin", "shape": "cylinder", "params": {"radius": 0.07, "height": 1.1},
     "placement": {"relative_to": "big", "side": "left", "gap": 0.15}},
    {"id": "plate", "shape": "box", "params": {"height": 0.2, "depth": 0.02}, "placement": {"between": ["thin", "big"]}},
]


def test_make_it_one_solid_with_an_empty_merged_group(client, token, monkeypatch):
    _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_add_objects", objects=THREE_PARTS)]),
        AssistantTurn(content="ok"),
    ])
    # "make it one solid part": the model adds just the merged group; the engine fills it.
    _, _, events, _ = _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_add_objects", objects=[
            {"id": "part", "shape": "group", "merge": True}])]),
        AssistantTurn(content="Merged into one solid."),
    ])
    (result,) = _results(events)
    assert result["ok"] and not result["data"]["warnings"]
    assert any("in merged group 'part'" in n for n in result["data"]["notes"])
    objects = {o["id"]: o for o in client.get(f"/scene3d/{CONV}", headers={"X-Oprel-Token": token}).json()["objects"]}
    assert all(objects[i]["parent"] == "part" for i in ("big", "thin", "plate"))
    glb = trimesh.load(io.BytesIO(client.get(f"/scene3d/{CONV}/scene.glb", headers={"X-Oprel-Token": token}).content),
                       file_type="glb")
    assert sorted(glb.graph.nodes_geometry) == ["part"]


def test_merge_sent_on_one_object_merges_everything(client, token, monkeypatch):
    _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_add_objects", objects=THREE_PARTS)]),
        AssistantTurn(content="ok"),
    ])
    _, model, events, _ = _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_update_objects", updates=[{"id": "big", "merge": True}])]),
        AssistantTurn(content="ok"),
    ])
    (result,) = _results(events)
    assert result["ok"] and any("into one solid, group 'part'" in n for n in result["data"]["notes"])
    _, model, _, _ = _chat(client, token, monkeypatch, [AssistantTurn(content="ok")])
    assert "- part: group merged (one solid)" in model.seen[0]["messages"][0]["content"]


def test_canvas_merge_button_and_stl_download(client, token, monkeypatch):
    _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_add_objects", objects=THREE_PARTS)]),
        AssistantTurn(content="ok"),
    ])
    h = {"X-Oprel-Token": token}
    r = client.post(f"/scene3d/{CONV}/merge", headers=h, json={"merge": True})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["merged"] == {"part": {"bodies": 1, "parts": ["big", "thin", "plate"]}} and body["warnings"] == []

    stl = client.get(f"/scene3d/{CONV}/scene.stl", headers=h)
    assert stl.status_code == 200 and stl.headers["x-solid-bodies"] == "1" and stl.headers["x-solid-watertight"] == "true"
    assert trimesh.load(io.BytesIO(stl.content), file_type="stl").is_watertight

    # Split again: the group stays, its parts are separate meshes.
    r = client.post(f"/scene3d/{CONV}/merge", headers=h, json={"merge": False, "group": "part"})
    assert r.status_code == 200 and r.json()["merged"] == {}
    assert client.post(f"/scene3d/{CONV}/merge", headers=h, json={"merge": True, "group": "big"}).status_code == 400


def test_unnamed_pipe_run_joins_each_part_to_the_one_before(client, token, monkeypatch):
    # Seen with Qwen3-8B: no ids, every part joined to "pipe" / "elbow" (= the first of each).
    run = [{"shape": "pipe", "params": {"length": 1, "radius": 0.05}}]
    for shape, ref in [("elbow", "pipe"), ("pipe", "elbow"), ("elbow", "pipe"), ("pipe", "elbow")]:
        params = {"radius": 0.05} if shape == "elbow" else {"length": 0.5, "radius": 0.05}
        run.append({"shape": shape, "params": params, "placement": {"relative_to": ref, "side": "end"}})
    _, _, events, _ = _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_add_objects", objects=run)]),
        AssistantTurn(content="done"),
    ])
    (result,) = _results(events)
    assert result["ok"], result["data"]
    assert result["data"]["warnings"] == []
    saved = {o["id"]: o for o in client.get(f"/scene3d/{CONV}", headers={"X-Oprel-Token": token}).json()["objects"]}
    assert saved["elbow2"]["placement"]["relative_to"] == "pipe2"
    assert saved["pipe3"]["placement"]["relative_to"] == "elbow2"


def test_pipe_parts_listed_without_placement_are_joined_into_a_run(client, token, monkeypatch):
    # Seen with Qwen3-8B: pipe, elbow, pipe listed with no position or placement, all at the origin.
    run = [{"shape": "pipe", "params": {"length": 1, "radius": 0.05}}, {"shape": "elbow", "params": {"radius": 0.05}},
           {"shape": "pipe", "params": {"length": 0.5, "radius": 0.05}}]
    _, _, events, _ = _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_add_objects", objects=run)]),
        AssistantTurn(content="done"),
    ])
    (result,) = _results(events)
    assert result["ok"], result["data"]
    saved = {o["id"]: o for o in client.get(f"/scene3d/{CONV}", headers={"X-Oprel-Token": token}).json()["objects"]}
    assert saved["elbow"]["placement"] == {"relative_to": "pipe", "side": "end"}
    assert saved["pipe2"]["placement"] == {"relative_to": "elbow", "side": "end"}
    by_id = {o["id"]: o for o in result["data"]["objects"]}
    assert by_id["pipe2"]["max"][1] > 0.5  # the last pipe goes up after the elbow


def test_merge_on_a_shape_wraps_it_and_its_parts_in_a_merged_group(client, token, monkeypatch):
    # Seen with Qwen3-8B, "a bowl with two handles": merge on the bowl, handles parented to it.
    _, _, events, _ = _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_add_objects", objects=[
            {"id": "bowl", "shape": "bowl", "merge": True},
            {"id": "handle", "shape": "handle", "parent": "bowl", "params": {"height": 0.04},
             "placement": {"relative_to": "bowl", "side": "right", "vertical": "top"}},
            {"id": "handle2", "shape": "handle", "parent": "bowl", "params": {"height": 0.04},
             "placement": {"relative_to": "bowl", "side": "left", "vertical": "top"}},
        ])]),
        AssistantTurn(content="done"),
    ])
    (result,) = _results(events)
    assert result["ok"], result["data"]
    assert result["data"]["warnings"] == []
    saved = {o["id"]: o for o in client.get(f"/scene3d/{CONV}", headers={"X-Oprel-Token": token}).json()["objects"]}
    assert saved["bowl_solid"] == {"id": "bowl_solid", "shape": "group", "merge": True}
    assert saved["bowl"]["parent"] == "bowl_solid" and "merge" not in saved["bowl"]


def test_handle_without_a_height_is_sized_to_a_big_vase(client, token, monkeypatch):
    _, _, events, _ = _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_add_objects", objects=[
            {"id": "vase", "shape": "vase", "params": {"height": 1.2, "radius": 0.2}},
            {"id": "handle", "shape": "handle", "params": {"style": "box"},
             "placement": {"relative_to": "vase", "side": "right"}},
        ])]),
        AssistantTurn(content="done"),
    ])
    (result,) = _results(events)
    assert result["ok"], result["data"]
    h = {o["id"]: o for o in result["data"]["objects"]}["handle"]
    assert h["max"][1] - h["min"][1] > 0.5  # half the vase, not 7 cm


def test_handles_given_as_cuts_are_added_instead(client, token, monkeypatch):
    # Qwen3-8B, "a curvy vase with smooth handles on both sides": the handles as cuts.
    _, _, events, _ = _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_add_objects", objects=[
            {"id": "vase", "shape": "vase", "cuts": [{"shape": "handle", "position": [0.1, 0.2, 0]},
                                                     {"shape": "handle", "position": [-0.1, 0.2, 0]}]},
            {"id": "box", "shape": "box", "params": {"width": 0.3, "height": 0.2, "depth": 0.2},
             "position": [1, 0, 0], "cuts": [{"shape": "handle", "params": {"style": "box"}}]},
        ])]),
        AssistantTurn(content="done"),
    ])
    (result,) = _results(events)
    assert result["ok"], result["data"]
    saved = {o["id"]: o for o in client.get(f"/scene3d/{CONV}", headers={"X-Oprel-Token": token}).json()["objects"]}
    assert saved["vase"]["params"]["handles"] == 2 and "cuts" not in saved["vase"]
    assert saved["box_handle"]["shape"] == "handle" and saved["box_handle"]["params"]["style"] == "box"
    assert saved["box_handle"]["placement"]["relative_to"] == "box"


def test_hint_example_cuts_match_the_system_hint():
    from oprel.integrations.scene3d import SYSTEM_HINT
    from oprel.integrations.scene3d.tools import HINT_EXAMPLE_CUTS
    flat = SYSTEM_HINT.replace("\\n", "").replace("\n", " ")
    assert json.dumps(HINT_EXAMPLE_CUTS)[1:-1] in flat


def test_example_cuts_copied_into_an_edit_are_left_out(client, token, monkeypatch):
    # Verbatim from Qwen3-8B, asked "I don't need a handle for the bowl".
    _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_add_objects", objects=[
            {"id": "bowl", "shape": "bowl", "params": {"handles": 2}}])]),
        AssistantTurn(content="ok"),
    ])
    _, _, events, _ = _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_update_objects", updates=[
            {"id": "bowl", "shape": "bowl", "params": {"handles": 0},
             "cuts": [{"shape": "cylinder", "params": {"radius": 0.03, "height": 0.1}},
                      {"shape": "cylinder", "params": {"radius": 0.004},
                       "pattern": {"type": "grid", "count": 2, "rows": 2, "spacing": [0.24, 0.24]}}]}])]),
        AssistantTurn(content="done"),
    ])
    (result,) = _results(events)
    assert result["ok"], result["data"]
    saved = {o["id"]: o for o in client.get(f"/scene3d/{CONV}", headers={"X-Oprel-Token": token}).json()["objects"]}
    assert saved["bowl"]["params"]["handles"] == 0 and "cuts" not in saved["bowl"]


def test_pipe_run_listed_by_kind_is_not_chained_in_list_order(client, token, monkeypatch):
    # Verbatim shape of Qwen3-8B's retry: all pipes, then the fittings, no placements.
    run = [{"shape": "pipe", "params": {"length": 1, "radius": 0.025}},
           {"shape": "pipe", "params": {"length": 0.5, "radius": 0.025}},
           {"shape": "elbow", "params": {"radius": 0.025}}]
    _, _, events, _ = _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_add_objects", objects=run)]),
        AssistantTurn(content="done"),
    ])
    (result,) = _results(events)
    assert result["ok"], result["data"]
    assert any("are not joined" in w and '"side": "end"' in w for w in result["data"]["warnings"])
    saved = client.get(f"/scene3d/{CONV}", headers={"X-Oprel-Token": token}).json()["objects"]
    assert not any(o.get("placement") for o in saved)


def test_kitchen_counter_with_sink_vase_and_mug(client, token, monkeypatch):
    # Qwen3-8B, "a kitchen counter ... a round sink hole 40 cm across cut into the top, a vase with two
    # handles on the left end, a mug with a handle on the right end": the sink as its own cylinder
    # cut away from itself and merged, everyday shape names. Rejected three times before.
    _, _, events, _ = _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_add_objects", objects=[
            {"id": "counter", "shape": "counter", "params": {"width": 2, "height": 0.9, "depth": 0.6}},
            {"id": "sink_hole", "shape": "cylinder", "params": {"radius": 0.2, "height": 0.9}, "merge": True,
             "cuts": [{"shape": "cylinder", "params": {"radius": 0.2}}]},
            {"id": "vase", "shape": "vase", "params": {"handles": 2},
             "placement": {"relative_to": "counter", "side": "top", "align": "left"}},
            {"id": "mug", "shape": "mug", "placement": {"relative_to": "counter", "side": "top", "align": "right"}},
        ])]),
        AssistantTurn(content="done"),
    ])
    (result,) = _results(events)
    assert result["ok"], result["data"]
    saved = {o["id"]: o for o in client.get(f"/scene3d/{CONV}", headers={"X-Oprel-Token": token}).json()["objects"]}
    assert set(saved) == {"counter", "vase", "mug"}
    assert saved["counter"]["shape"] == "box" and saved["mug"]["shape"] == "cup" and saved["mug"]["params"]["handle"]
    (cut,) = saved["counter"]["cuts"]
    assert cut["params"] == {"radius": 0.2} and cut["position"][1] == pytest.approx(0.45)  # through the top
    objs = {o["id"]: o for o in result["data"]["objects"]}
    assert objs["vase"]["min"][1] == pytest.approx(0.9) and objs["mug"]["min"][1] == pytest.approx(0.9)


def test_one_broken_object_does_not_stop_the_rest(client, token, monkeypatch):
    _, _, events, _ = _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_add_objects", objects=[
            {"id": "table", "shape": "table"},
            {"id": "spring", "shape": "spring", "params": {"radius": 0.03, "height": 0.05, "coils": 10}},
            {"id": "vase", "shape": "vase", "placement": {"relative_to": "table", "side": "top"}},
            {"id": "tag", "shape": "box", "params": {"width": 0.02, "height": 0.02, "depth": 0.02},
             "placement": {"relative_to": "spring", "side": "top"}},
        ])]),
        AssistantTurn(content="done"),
    ])
    (result,) = _results(events)
    assert result["ok"], result["data"]
    warnings = result["data"]["warnings"]
    assert any(w.startswith("NOT ADDED: 'spring' (spring): the coils would touch") for w in warnings)
    assert any("NOT ADDED: 'tag' needs 'spring'" in w for w in warnings)
    saved = {o["id"] for o in client.get(f"/scene3d/{CONV}", headers={"X-Oprel-Token": token}).json()["objects"]}
    assert saved == {"table", "vase"}


def test_counter_retry_verbatim_cut_object_and_things_beside_lifted(client, token, monkeypatch):
    # Verbatim from Qwen3-8B's first call: the sink as {"shape": "cut", ...} on the counter's top, the
    # vase and mug "left" / "right" of it lifted by 0.45 (meant: on the counter at its ends).
    _, _, events, _ = _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_add_objects", objects=[
            {"shape": "box", "id": "counter", "params": {"width": 2, "height": 0.9, "depth": 0.6}, "position": [0, 0, 0]},
            {"shape": "cut", "id": "sink_cut", "params": {"shape": "cylinder", "params": {"radius": 0.2}},
             "placement": {"relative_to": "counter", "side": "top", "toward": "down", "offset": [0, 0.45, 0]}},
            {"shape": "vase", "id": "vase", "params": {"height": 0.5, "radius": 0.15, "neck_radius": 0.1,
                                                       "base_radius": 0.15, "top_radius": 0.1, "handles": 2},
             "placement": {"relative_to": "counter", "side": "left", "toward": "left", "offset": [-0.3, 0.45, 0]}},
            {"shape": "cup", "id": "mug", "params": {"height": 0.3, "radius": 0.08, "handle": True},
             "placement": {"relative_to": "counter", "side": "right", "toward": "right", "offset": [0.3, 0.45, 0]}},
        ])]),
        AssistantTurn(content="done"),
    ])
    (result,) = _results(events)
    assert result["ok"], result["data"]
    saved = {o["id"]: o for o in client.get(f"/scene3d/{CONV}", headers={"X-Oprel-Token": token}).json()["objects"]}
    assert set(saved) == {"counter", "vase", "mug"}
    (cut,) = saved["counter"]["cuts"]
    assert cut["shape"] == "cylinder" and cut["params"] == {"radius": 0.2}  # right through the top
    objs = {o["id"]: o for o in result["data"]["objects"]}
    assert objs["vase"]["min"][1] == pytest.approx(0.9) and objs["vase"]["min"][0] == pytest.approx(-1.0)
    assert objs["mug"]["min"][1] == pytest.approx(0.9) and objs["mug"]["max"][0] == pytest.approx(1.0)
    assert not any("floats" in w for w in result["data"]["warnings"])


def test_hole_placed_beside_the_object_it_names_becomes_a_hole_in_it(client, token, monkeypatch):
    # Qwen3-8B's retry: a "sink_hole" cylinder placed "left" of the counter with a gap.
    _, _, events, _ = _chat(client, token, monkeypatch, [
        AssistantTurn(content="", tool_calls=[_call("scene_add_objects", objects=[
            {"shape": "box", "id": "counter", "params": {"width": 2, "height": 0.9, "depth": 0.6}},
            {"shape": "cylinder", "id": "sink_hole", "params": {"radius": 0.2, "height": 0.9},
             "placement": {"relative_to": "counter", "side": "left", "gap": 0.01}},
        ])]),
        AssistantTurn(content="done"),
    ])
    (result,) = _results(events)
    assert result["ok"], result["data"]
    saved = {o["id"]: o for o in client.get(f"/scene3d/{CONV}", headers={"X-Oprel-Token": token}).json()["objects"]}
    assert set(saved) == {"counter"} and saved["counter"]["cuts"][0]["position"][0] == pytest.approx(0)
