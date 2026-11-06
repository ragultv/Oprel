"""Geometry engine for the 3D scene toolset: shapes, anchoring, placement, patterns, groups, checks."""

import io

import numpy as np
import pytest
import trimesh

from oprel.integrations.scene3d.engine import SceneError, compile_scene, export_glb
from oprel.integrations.scene3d.engine.registry import all_shapes, get_shape


def _bbox(scene, oid):
    lo, hi = scene.objects[oid].bbox
    return lo.round(4).tolist(), hi.round(4).tolist()


SAMPLE_PARAMS = {
    "box": {"width": 1, "height": 2, "depth": 3},
    "cylinder": {"radius": 0.5, "height": 2},
    "cone": {"radius": 0.5, "height": 1},
    "sphere": {"radius": 0.5},
    "capsule": {"radius": 0.2, "height": 1},
    "torus": {"major_radius": 1, "minor_radius": 0.2},
    "rectangle": {"width": 1, "depth": 2},
    "circle": {"radius": 0.5},
}


@pytest.mark.parametrize("shape", sorted(SAMPLE_PARAMS))
def test_every_shape_is_watertight_and_rests_on_the_ground(shape):
    scene = compile_scene([{"id": "s", "shape": shape, "params": SAMPLE_PARAMS[shape]}])
    co = scene.objects["s"]
    assert co.built.mesh.is_watertight
    assert co.built.mesh.volume > 0  # outward normals
    lo, hi = co.bbox
    assert lo[1] == pytest.approx(0, abs=1e-9)  # bottom on the ground
    assert (lo[0] + hi[0]) / 2 == pytest.approx(0, abs=1e-6)  # centred in X/Z
    assert (lo[2] + hi[2]) / 2 == pytest.approx(0, abs=1e-6)


READY_MADE = ["table", "chair", "stool", "bench", "bookshelf", "bed", "sofa", "lamp"]


def test_every_registered_mesh_shape_is_tested():
    meshy = {s.name for s in all_shapes() if s.builder is not None}
    assert meshy == set(SAMPLE_PARAMS) | set(READY_MADE)


@pytest.mark.parametrize("shape", READY_MADE)
def test_ready_made_objects_build_with_defaults_and_stand_on_the_ground(shape):
    scene = compile_scene([{"id": "o", "shape": shape}])
    lo, hi = scene.objects["o"].bbox
    assert lo[1] == pytest.approx(0, abs=1e-9) and hi[1] > 0.3
    assert scene.warnings == []


@pytest.mark.parametrize("legs, top", [(1, "rectangle"), (3, "round"), (4, "rectangle"), (5, "round"), (6, "rectangle")])
def test_table_variants_have_their_legs_under_the_top(legs, top):
    scene = compile_scene([{"id": "t", "shape": "table", "params": {"legs": legs, "top": top, "width": 1.2}}])
    mesh = scene.objects["t"].built.mesh
    top_lo, top_hi = mesh.bounds
    under = mesh.vertices[mesh.vertices[:, 1] < 0.7]  # everything below the top
    assert np.all(np.abs(under[:, 0]) <= top_hi[0] + 1e-9) and np.all(np.abs(under[:, 2]) <= top_hi[2] + 1e-9)


def test_chair_backrest_is_at_the_back():
    mesh = compile_scene([{"id": "c", "shape": "chair"}]).objects["c"].built.mesh
    high = mesh.vertices[mesh.vertices[:, 1] > 0.5]  # above the seat: only the backrest
    assert high[:, 2].max() < -0.15  # it sits at -Z, so the chair faces +Z


def _facing(m):
    return np.round(m[:3, :3] @ np.array([0, 0, 1]), 3).tolist()


def test_chairs_around_a_rectangular_table_sit_on_the_long_sides_facing_it():
    scene = compile_scene([
        {"id": "table", "shape": "table", "params": {"width": 1.6, "depth": 0.9}},
        {"id": "chairs", "shape": "chair", "pattern": {"type": "around", "relative_to": "table", "count": 4}},
    ])
    seats = sorted((np.round(m[:3, 3], 3).tolist(), _facing(m)) for m in scene.objects["chairs"].instances)
    assert sorted(abs(s[0][2]) for s in seats) == [0.675] * 4  # two per long side
    for (x, _, z), facing in seats:
        assert facing == [0.0, 0.0, -1.0 if z > 0 else 1.0]  # toward the table
    assert scene.warnings == []


def test_six_chairs_use_the_ends_and_round_tables_get_a_circle():
    six = compile_scene([
        {"id": "table", "shape": "table", "params": {"width": 1.8, "depth": 0.9}},
        {"id": "chairs", "shape": "chair", "pattern": {"type": "around", "relative_to": "table", "count": 6}},
    ]).objects["chairs"].instances
    ends = [m for m in six if abs(m[2, 3]) < 1e-9]
    assert len(ends) == 2 and sorted(_facing(m)[0] for m in ends) == [-1.0, 1.0]

    round_ = compile_scene([
        {"id": "table", "shape": "table", "params": {"width": 1.2, "top": "round", "legs": 3}},
        {"id": "chairs", "shape": "chair", "pattern": {"type": "around", "relative_to": "table", "count": 5}},
    ]).objects["chairs"].instances
    dists = {round(float(np.hypot(m[0, 3], m[2, 3])), 3) for m in round_}
    assert len(dists) == 1  # all on one circle
    for m in round_:  # each faces the centre
        to_centre = -m[:3, 3] / np.linalg.norm(m[:3, 3])
        assert np.allclose(m[:3, :3] @ np.array([0, 0, 1]), to_centre, atol=1e-6)


def test_around_needs_a_real_reference():
    with pytest.raises(SceneError) as info:
        compile_scene([{"id": "c", "shape": "chair", "pattern": {"type": "around", "count": 4}}])
    assert "needs 'relative_to'" in str(info.value)


def test_hint_examples_compile_cleanly():
    import json
    import re

    from oprel.integrations.scene3d import SYSTEM_HINT

    examples = re.findall(r'(\{"plan".*?\]\})', SYSTEM_HINT, re.S)
    assert len(examples) == 2
    for text in examples:
        assert compile_scene(json.loads(text)["objects"]).warnings == []


def test_sizes_match_params():
    scene = compile_scene([
        {"id": "b", "shape": "box", "params": {"width": 1, "height": 2, "depth": 3}, "position": [5, 0, 0]},
        {"id": "taper", "shape": "cylinder", "params": {"radius_bottom": 0.5, "radius_top": 0.2, "height": 1}},
    ])
    assert _bbox(scene, "b") == ([4.5, 0, -1.5], [5.5, 2, 1.5])
    lo, hi = scene.objects["taper"].bbox
    assert hi[1] - lo[1] == pytest.approx(1)
    assert hi[0] - lo[0] == pytest.approx(1, abs=0.01)  # widest at the bottom


def test_placement_sides_and_alignment():
    scene = compile_scene([
        {"id": "cyl", "shape": "cylinder", "params": {"radius": 1, "height": 2}},
        {"id": "right", "shape": "box", "params": {"width": 1, "height": 1, "depth": 1},
         "placement": {"relative_to": "cyl", "side": "right", "gap": 0.5}},
        {"id": "on_top", "shape": "box", "params": {"width": 1, "height": 1, "depth": 1},
         "placement": {"relative_to": "cyl", "side": "top", "align": "front-right"}},
        {"id": "behind", "shape": "box", "params": {"width": 1, "height": 1, "depth": 1},
         "placement": {"relative_to": "cyl", "side": "back", "vertical": "top"}},
    ])
    assert _bbox(scene, "right") == ([1.5, 0, -0.5], [2.5, 1, 0.5])
    assert _bbox(scene, "on_top") == ([0, 2, 0], [1, 3, 1])
    assert _bbox(scene, "behind") == ([-0.5, 1, -2], [0.5, 2, -1])


def test_grid_pattern_is_centred_and_children_follow_the_group():
    scene = compile_scene([
        {"id": "table", "shape": "group", "position": [10, 0, 0]},
        {"id": "legs", "parent": "table", "shape": "cylinder", "params": {"radius": 0.05, "height": 0.7},
         "pattern": {"type": "grid", "count": 2, "rows": 2, "spacing": [1.0, 0.5]}},
        {"id": "top", "parent": "table", "shape": "box", "params": {"width": 1.2, "height": 0.05, "depth": 0.7},
         "position": [0, 0.7, 0]},
    ])
    assert len(scene.objects["legs"].instances) == 4
    assert _bbox(scene, "legs") == ([9.45, 0, -0.3], [10.55, 0.7, 0.3])
    assert _bbox(scene, "top") == ([9.4, 0.7, -0.35], [10.6, 0.75, 0.35])
    assert _bbox(scene, "table") == ([9.4, 0, -0.35], [10.6, 0.75, 0.35])
    assert scene.warnings == []


def test_radial_and_linear_patterns():
    scene = compile_scene([
        {"id": "pillars", "shape": "cylinder", "params": {"radius": 0.1, "height": 1},
         "pattern": {"type": "radial", "count": 8, "radius": 2}},
        {"id": "row", "shape": "box", "params": {"width": 0.2, "height": 0.2, "depth": 0.2}, "position": [0, 0, 5],
         "pattern": {"type": "linear", "count": 3, "step": [1, 0, 0]}},
    ])
    starts = sorted(np.round(m[:3, 3], 4).tolist() for m in scene.objects["pillars"].instances)
    assert [0.0, 0.0, 2.0] in starts and [0.0, 0.0, -2.0] in starts and [2.0, 0.0, 0.0] in starts
    assert _bbox(scene, "row") == ([-0.1, 0, 4.9], [2.1, 0.2, 5.1])


def test_patterned_group_repeats_its_parts():
    scene = compile_scene([
        {"id": "chair", "shape": "group", "pattern": {"type": "radial", "count": 4, "radius": 1, "face_center": True}},
        {"id": "seat", "parent": "chair", "shape": "box", "params": {"width": 0.4, "height": 0.05, "depth": 0.4}},
    ])
    assert len(scene.objects["seat"].instances) == 4


def test_rotation_keeps_the_object_resting_in_place():
    scene = compile_scene([
        {"id": "lying", "shape": "capsule", "params": {"radius": 0.2, "height": 1}, "rotation": [0, 0, 90],
         "position": [0, 0, 3]},
    ])
    assert _bbox(scene, "lying") == ([-0.5, 0, 2.8], [0.5, 0.4, 3.2])
    assert scene.warnings == []


def test_part_placed_relative_to_its_parent_uses_the_parent_shape():
    scene = compile_scene([
        {"id": "cap", "shape": "cylinder", "params": {"radius": 0.1, "height": 0.5}},
        {"id": "clip", "parent": "cap", "shape": "box", "params": {"width": 0.02, "height": 0.3, "depth": 0.03},
         "placement": {"relative_to": "cap", "side": "right", "vertical": "top"}},
    ])
    assert _bbox(scene, "clip") == ([0.1, 0.2, -0.015], [0.12, 0.5, 0.015])


def test_warnings_for_floating_and_below_ground():
    scene = compile_scene([
        {"id": "ball", "shape": "sphere", "params": {"radius": 0.5}, "position": [0, 2, 0]},
        {"id": "sunk", "shape": "box", "params": {"width": 1, "height": 1, "depth": 1}, "position": [3, -0.5, 0]},
        {"id": "base", "shape": "box", "params": {"width": 1, "height": 1, "depth": 1}, "position": [6, 0, 0]},
        {"id": "stacked", "shape": "box", "params": {"width": 0.5, "height": 0.5, "depth": 0.5}, "position": [6, 1, 0]},
    ])
    text = " ".join(scene.warnings)
    assert "'ball' floats" in text
    assert "'sunk' goes 0.500 m below" in text
    assert "stacked" not in text  # resting on base


@pytest.mark.parametrize("objects, fragment", [
    ([{"id": "a", "shape": "blob"}], "must be one of"),
    ([{"id": "a", "shape": "box", "params": {"width": 1}}], "Missing required argument 'height'"),
    ([{"id": "a", "shape": "box", "params": {"width": -1, "height": 1, "depth": 1}}], "must be >="),
    ([{"id": "a", "shape": "cylinder", "params": {"height": 1}}], "needs 'radius'"),
    ([{"id": "a", "shape": "group", "parent": "nope"}], "parent 'nope' does not exist"),
    ([{"id": "a", "shape": "group"}, {"id": "a", "shape": "group"}], "Duplicate id 'a'"),
    ([{"id": "a", "shape": "group", "parent": "b"}, {"id": "b", "shape": "group", "parent": "a"}], "Parent loop"),
    ([{"id": "a", "shape": "sphere", "params": {"radius": 1}, "placement": {"relative_to": "b", "side": "left"}},
      {"id": "b", "shape": "sphere", "params": {"radius": 1}, "placement": {"relative_to": "a", "side": "left"}}],
     "Circular placement"),
    ([{"id": "a", "shape": "box", "params": {"width": 1, "height": 1, "depth": 1},
       "pattern": {"type": "linear", "count": 3}}], "needs 'step'"),
    ([{"id": "a", "shape": "box", "params": {"width": 1, "height": 1, "depth": 1},
       "placement": {"relative_to": "a"}}], "relative to itself"),
])
def test_bad_recipes_explain_what_to_fix(objects, fragment):
    with pytest.raises(SceneError) as info:
        compile_scene(objects)
    assert fragment in str(info.value)


def test_small_model_sloppiness_is_coerced():
    scene = compile_scene([
        {"id": "b", "shape": "Box", "params": {"width": "1", "height": 2, "depth": 1}, "position": "1, 0, 0"},
    ])
    assert _bbox(scene, "b") == ([0.5, 0, -0.5], [1.5, 2, 0.5])


def test_glb_has_one_named_node_per_copy():
    scene = compile_scene([
        {"id": "base", "shape": "box", "params": {"width": 1, "height": 1, "depth": 1}, "color": "wood"},
        {"id": "legs", "shape": "cylinder", "params": {"radius": 0.1, "height": 1}, "position": [3, 0, 0],
         "pattern": {"type": "linear", "count": 3, "step": [0.5, 0, 0]}, "color": "#123456", "opacity": 0.5},
        {"id": "empty", "shape": "group"},
    ])
    loaded = trimesh.load(io.BytesIO(export_glb(scene)), file_type="glb")
    assert sorted(loaded.graph.nodes_geometry) == ["base", "legs#0", "legs#1", "legs#2"]


def test_empty_scene_exports_nothing():
    assert export_glb(compile_scene([{"id": "g", "shape": "group"}])) is None


def test_legs_that_reach_outside_the_top_are_flagged():
    # Exactly what Qwen3-8B drew for "a table with five legs": 5 legs in a 4 m row under a 1.2 m top.
    scene = compile_scene([
        {"id": "table", "shape": "group"},
        {"id": "leg", "parent": "table", "shape": "cylinder", "params": {"radius": 0.03, "height": 0.72},
         "pattern": {"type": "grid", "count": 5, "rows": 1, "spacing": [1.0, 0.45]}},
        {"id": "top", "parent": "table", "shape": "box", "params": {"width": 1.2, "height": 0.03, "depth": 0.6},
         "position": [0, 0.72, 0]},
    ])
    assert any("5 copies of 'leg' stick out 1.43 m past 'top'" in w and "at least 1.43 m" in w
               for w in scene.warnings)


def test_legs_that_fit_and_a_vase_on_a_top_are_not_flagged():
    scene = compile_scene([
        {"id": "legs", "shape": "cylinder", "params": {"radius": 0.03, "height": 0.72},
         "pattern": {"type": "radial", "count": 5, "radius": 0.4}},
        {"id": "top", "shape": "cylinder", "params": {"radius": 0.5, "height": 0.03}, "position": [0, 0.72, 0]},
        {"id": "vase", "shape": "cylinder", "params": {"radius": 0.05, "height": 0.2},
         "placement": {"relative_to": "top", "side": "top"}},
    ])
    assert scene.warnings == []


def test_part_left_hanging_in_a_group_is_flagged():
    # Qwen3-8B "fixed" an overhang by shortening the legs to 0.7 m under a top at 0.75 m.
    scene = compile_scene([
        {"id": "table", "shape": "group"},
        {"id": "leg", "parent": "table", "shape": "cylinder", "params": {"radius": 0.03, "height": 0.7},
         "pattern": {"type": "radial", "count": 5, "radius": 0.3}},
        {"id": "top", "parent": "table", "shape": "box", "params": {"width": 1.2, "height": 0.03, "depth": 0.6},
         "position": [0, 0.75, 0]},
    ])
    assert any("'top' touches nothing: it is 0.050 m from 'leg'" in w for w in scene.warnings)


def test_y_used_instead_of_z_is_flagged():
    # Qwen3-8B listed legs one by one and wrote [0, 0.5, 0] meaning z = 0.5.
    leg = {"radius": 0.03, "height": 0.75}
    scene = compile_scene([
        {"id": "table", "shape": "group"},
        {"id": "leg1", "parent": "table", "shape": "cylinder", "params": leg, "position": [0.5, 0, 0]},
        {"id": "leg2", "parent": "table", "shape": "cylinder", "params": leg, "position": [-0.5, 0, 0]},
        {"id": "leg3", "parent": "table", "shape": "cylinder", "params": leg, "position": [0, 0.5, 0]},
        {"id": "leg4", "parent": "table", "shape": "cylinder", "params": leg, "position": [0, 0.75, 0]},
        {"id": "top", "parent": "table", "shape": "box", "params": {"width": 1, "height": 0.03, "depth": 1},
         "position": [0, 0.75, 0]},
    ])
    text = " ".join(scene.warnings)
    assert "'leg3' is identical to 'leg1'" in text and "y=0.50" in text and "change z, not y" in text
    assert "'leg4' is identical to 'leg1'" in text  # the one it 'fixed' by standing it on the top


def test_side_attachments_count_as_connected():
    scene = compile_scene([
        {"id": "pen", "shape": "group"},
        {"id": "cap", "parent": "pen", "shape": "cylinder", "params": {"radius": 0.006, "height": 0.04}},
        {"id": "clip", "parent": "pen", "shape": "box", "params": {"width": 0.002, "height": 0.03, "depth": 0.003},
         "placement": {"relative_to": "cap", "side": "right", "vertical": "top"}},
    ])
    assert scene.warnings == []


def test_group_without_parts_is_flagged():
    # Seen with Qwen3-4B: it made the group but forgot "parent" on the parts.
    scene = compile_scene([
        {"id": "table", "shape": "group"},
        {"id": "top", "shape": "box", "params": {"width": 1, "height": 0.05, "depth": 1}},
    ])
    assert any("'table' is an empty group" in w for w in scene.warnings)


def test_shape_signatures_are_prompt_friendly():
    assert get_shape("box").signature() == "box(width, height, depth)"
    assert get_shape("cylinder").signature() == "cylinder(radius?, radius_top?, radius_bottom?, height, segments?)"
