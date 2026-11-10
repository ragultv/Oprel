"""Row 5: shapes along a path (pipe, elbow, tee, reducer, beam, spring, handle, sweep) and joining ends."""

import numpy as np
import pytest

from oprel.integrations.scene3d.engine import SceneError, compile_scene
from oprel.integrations.scene3d.engine.frame import transform_points

PATH_PARAMS = {
    "pipe": {"length": 1, "radius": 0.05},
    "elbow": {"radius": 0.05},
    "tee": {"radius": 0.05},
    "reducer": {"radius": 0.05, "end_radius": 0.03},
    "beam": {"length": 2},
    "spring": {"radius": 0.03, "height": 0.12},
    "handle": {},
    "sweep": {"path": [[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0], [0.5, 0.5, 0.5]], "radius": 0.02},
}


def _end(scene, oid, name):
    co = scene.objects[oid]
    m = co.instances[0]
    p, d = co.built.port(name)
    return transform_points(m, p[None])[0], m[:3, :3] @ d


@pytest.mark.parametrize("shape", sorted(PATH_PARAMS))
def test_path_shapes_are_closed_solids_on_the_ground(shape):
    scene = compile_scene([{"id": "s", "shape": shape, "params": PATH_PARAMS[shape]}])
    mesh = scene.objects["s"].built.mesh
    assert mesh.is_watertight and mesh.is_winding_consistent
    assert mesh.volume > 0
    assert scene.objects["s"].bbox[0][1] == pytest.approx(0, abs=1e-9)


def test_pipe_is_hollow_with_the_given_wall():
    scene = compile_scene([{"id": "p", "shape": "pipe", "params": {"length": 1, "radius": 0.05, "wall": 0.005}}])
    mesh = scene.objects["p"].built.mesh
    assert mesh.volume == pytest.approx(np.pi * (0.05 ** 2 - 0.045 ** 2), rel=0.02)
    lo, hi = scene.objects["p"].bbox
    assert (hi - lo).round(4).tolist() == [1, 0.1, 0.1]


def test_pipe_axis_y_stands_upright():
    scene = compile_scene([{"id": "p", "shape": "pipe", "params": {"length": 1, "radius": 0.05, "axis": "y"}}])
    lo, hi = scene.objects["p"].bbox
    assert (hi - lo).round(4).tolist() == [0.1, 1, 0.1]
    start, facing = _end(scene, "p", "start")
    assert start.round(4).tolist() == [0, 0, 0] and facing.round(4).tolist() == [0, -1, 0]


def test_beam_section_height_is_vertical():
    scene = compile_scene([{"id": "b", "shape": "beam", "params": {"length": 2, "width": 0.1, "height": 0.2}}])
    lo, hi = scene.objects["b"].bbox
    assert (hi - lo).round(4).tolist() == [2, 0.2, 0.1]


def test_pipe_run_joins_end_to_end_and_turns_where_told():
    scene = compile_scene([
        {"id": "p1", "shape": "pipe", "params": {"length": 1, "radius": 0.05}},
        {"id": "e1", "shape": "elbow", "params": {"radius": 0.05}, "placement": {"relative_to": "p1", "side": "end"}},
        {"id": "p2", "shape": "pipe", "params": {"length": 0.8, "radius": 0.05},
         "placement": {"relative_to": "e1", "side": "end"}},
        {"id": "e2", "shape": "elbow", "params": {"radius": 0.05},
         "placement": {"relative_to": "p2", "side": "end", "toward": "front"}},
        {"id": "p3", "shape": "pipe", "params": {"length": 0.5, "radius": 0.05},
         "placement": {"relative_to": "e2", "side": "end"}},
        {"id": "t", "shape": "tee", "params": {"radius": 0.05},
         "placement": {"relative_to": "p3", "side": "end", "toward": "left"}},
    ])
    for a, b in [("p1", "e1"), ("e1", "p2"), ("p2", "e2"), ("e2", "p3"), ("p3", "t")]:
        end, out = _end(scene, a, "end")
        start, back = _end(scene, b, "start")
        assert np.allclose(end, start, atol=1e-6)
        assert np.allclose(out, -back, atol=1e-6)
    assert np.allclose(_end(scene, "p2", "end")[1], [0, 1, 0], atol=1e-6)    # elbow 1 turned up
    assert np.allclose(_end(scene, "p3", "end")[1], [0, 0, 1], atol=1e-6)    # elbow 2 toward the front
    assert np.allclose(_end(scene, "t", "branch")[1], [-1, 0, 0], atol=1e-6)  # branch to the left
    assert scene.warnings == []


def test_joined_run_merges_into_one_body():
    scene = compile_scene([
        {"id": "run", "shape": "group", "merge": True},
        {"id": "p1", "parent": "run", "shape": "pipe", "params": {"length": 1, "radius": 0.05}},
        {"id": "e1", "parent": "run", "shape": "elbow", "params": {"radius": 0.05},
         "placement": {"relative_to": "p1", "side": "end"}},
        {"id": "p2", "parent": "run", "shape": "pipe", "params": {"length": 0.5, "radius": 0.05},
         "placement": {"relative_to": "e1", "side": "end"}},
    ])
    assert scene.merged["run"].bodies == 1


def test_gap_between_ends_names_the_fix():
    scene = compile_scene([
        {"id": "p1", "shape": "pipe", "params": {"length": 1, "radius": 0.05}},
        {"id": "p2", "shape": "pipe", "params": {"length": 1, "radius": 0.05}, "position": [1.03, 0, 0]},
    ])
    assert any('"relative_to": "p1", "side": "end"' in w and "'p2'" in w for w in scene.warnings)


def test_joining_to_a_shape_without_ends_is_an_error():
    with pytest.raises(SceneError, match="box has no open ends"):
        compile_scene([
            {"id": "b", "shape": "box", "params": {"width": 1, "height": 1, "depth": 1}},
            {"id": "p", "shape": "pipe", "params": {"length": 1, "radius": 0.05},
             "placement": {"relative_to": "b", "side": "end"}},
        ])


def test_only_parts_with_ends_can_be_joined_to_one():
    with pytest.raises(SceneError, match="have ends to join"):
        compile_scene([
            {"id": "p", "shape": "pipe", "params": {"length": 1, "radius": 0.05}},
            {"id": "b", "shape": "box", "params": {"width": 1, "height": 1, "depth": 1},
             "placement": {"relative_to": "p", "side": "end"}},
        ])


def test_handle_is_pushed_onto_a_tapered_cup():
    scene = compile_scene([
        {"id": "mug", "shape": "group", "merge": True},
        {"id": "cup", "parent": "mug", "shape": "cup", "params": {"height": 0.1, "radius": 0.04}},
        {"id": "h", "parent": "mug", "shape": "handle", "params": {"height": 0.06},
         "placement": {"relative_to": "cup", "side": "right", "vertical": "center"}},
    ])
    cup_right = scene.objects["cup"].bbox[1][0]
    assert scene.objects["h"].bbox[0][0] < cup_right - 0.003  # in past the rim's bounding box
    assert scene.merged["mug"].bodies == 1


@pytest.mark.parametrize("shape, params, fix", [
    ("spring", {"radius": 0.03, "height": 0.05, "coils": 10}, "use coils"),
    ("sweep", {"path": [[0, 0, 0], [0.05, 0, 0], [0.05, 0.5, 0]], "radius": 0.02, "bend_radius": 0.2},
     "use bend_radius"),
    ("sweep", {"path": [[0, 0, 0], [1, 0, 0], [1, 1, 0]], "radius": 0.05, "bend_radius": 0.03}, "use 0.075"),
    ("elbow", {"radius": 0.05, "bend_radius": 0.04}, "use 0.15"),
    ("pipe", {"length": 1, "radius": 0.05, "wall": 0.06}, "wall"),
])
def test_impossible_sizes_name_a_value_that_works(shape, params, fix):
    with pytest.raises(SceneError, match=fix):
        compile_scene([{"id": "s", "shape": shape, "params": params}])


def test_two_parts_on_one_end_are_flagged():
    scene = compile_scene([
        {"id": "p1", "shape": "pipe", "params": {"length": 1, "radius": 0.05}},
        {"id": "e1", "shape": "elbow", "params": {"radius": 0.05}, "placement": {"relative_to": "p1", "side": "end"}},
        {"id": "e2", "shape": "elbow", "params": {"radius": 0.05}, "placement": {"relative_to": "p1", "side": "end"}},
    ])
    assert any("'e1' and 'e2' are all joined to the end of 'p1'" in w for w in scene.warnings)


def test_mug_is_a_cup_with_a_handle_fused_to_it():
    scene = compile_scene([
        {"id": "m", "shape": "group", "merge": True},
        {"id": "plain", "parent": "m", "shape": "cup"},
    ])
    plain = scene.objects["plain"].bbox
    scene = compile_scene([
        {"id": "m", "shape": "group", "merge": True},
        {"id": "mug", "parent": "m", "shape": "cup", "params": {"handle": True}},
    ])
    lo, hi = scene.objects["mug"].bbox
    assert hi[0] - lo[0] > plain[1][0] - plain[0][0] + 0.01  # the handle sticks out
    assert scene.merged["m"].bodies == 1


def test_sweep_reads_what_small_models_send_for_a_rod():
    # Qwen3-8B, "a hook bent from a 1 cm thick rod": wall true, two points as the profile, thickness.
    scene = compile_scene([{"id": "hook", "shape": "sweep", "params": {
        "path": [[0, 0, 0], [0, 0.3, 0], [0, 0.3, -0.3], [0, 0, -0.3]],
        "profile": [[0, 0], [0, 0.01]], "thickness": 0.01, "wall": True}}])
    lo, hi = scene.objects["hook"].bbox
    assert hi[0] - lo[0] == pytest.approx(0.01, abs=1e-4)  # a 1 cm rod
    assert scene.objects["hook"].built.mesh.is_watertight


def test_pattern_spread_too_wide_names_the_spacing_that_fits():
    scene = compile_scene([{"id": "b", "shape": "beam", "params": {"length": 2, "width": 0.1, "height": 0.2},
                            "cuts": [{"shape": "cylinder", "params": {"radius": 0.01},
                                      "pattern": {"type": "grid", "count": 4, "rows": 2, "spacing": [0.1, 0.1]}}]}])
    assert any("spacing along z of at most 0.076" in w for w in scene.warnings)


@pytest.mark.parametrize("side", ["right", "left", "front", "back"])
def test_handle_turns_and_leans_onto_a_bowl_on_any_side(side):
    scene = compile_scene([
        {"id": "g", "shape": "group", "merge": True},
        {"id": "b", "parent": "g", "shape": "bowl"},
        {"id": "h", "parent": "g", "shape": "handle", "params": {"height": 0.04},
         "placement": {"relative_to": "b", "side": side, "vertical": "top"}},
    ])
    assert scene.merged["g"].bodies == 1  # both ends on the flared wall
    lo, hi = scene.objects["h"].bbox
    blo, bhi = scene.objects["b"].bbox
    axis, outward = {"right": (0, 1), "left": (0, -1), "front": (2, 1), "back": (2, -1)}[side]
    assert (hi[axis] > bhi[axis]) if outward > 0 else (lo[axis] < blo[axis])  # sticks out on that side


def test_bowl_with_two_handles_is_one_solid():
    scene = compile_scene([{"id": "g", "shape": "group", "merge": True},
                           {"id": "b", "parent": "g", "shape": "bowl", "params": {"handles": 2}}])
    lo, hi = scene.objects["b"].bbox
    assert hi[0] - lo[0] > 0.2 and scene.merged["g"].bodies == 1


def test_box_handle_is_a_square_bar_grip():
    scene = compile_scene([{"id": "h", "shape": "handle", "params": {"style": "box"}}])
    mesh = scene.objects["h"].built.mesh
    assert mesh.is_watertight
    assert (scene.objects["h"].bbox[1] - scene.objects["h"].bbox[0]).round(3).tolist() == [0.03, 0.07, 0.012]


def test_handle_joined_to_an_end_goes_on_the_side_and_toward_is_ignored():
    # Qwen3-8B, "a curvy vase with a box handle": side "end", then "toward" on side "right".
    for placement in ({"relative_to": "v", "side": "end"}, {"relative_to": "v", "side": "right", "toward": "up"}):
        scene = compile_scene([{"id": "v", "shape": "vase"},
                               {"id": "h", "shape": "handle", "params": {"height": 0.1}, "placement": placement}])
        assert scene.objects["h"].bbox[0][0] > 0  # on the right


def test_box_named_handle_touching_nothing_names_the_handle_shape():
    scene = compile_scene([
        {"id": "vase", "shape": "vase", "params": {"height": 1.2, "radius": 0.2}},
        {"id": "handle", "shape": "box", "params": {"width": 0.05, "height": 0.08, "depth": 0.05},
         "position": [0.3, 0, 0.2]},
    ])
    assert any('"shape": "handle", "params": {"style": "box"}' in w for w in scene.warnings)


@pytest.mark.parametrize("shape, params", [
    ("vase", {"handles": 2}), ("vase", {"handles": 2, "height": 1.2, "radius": 0.2, "handle_style": "box"}),
    ("cup", {"handles": 2}), ("bowl", {"handles": 1, "handle_style": "box"}),
])
def test_vessel_handles_fuse_with_the_body(shape, params):
    scene = compile_scene([{"id": "g", "shape": "group", "merge": True},
                           {"id": "o", "parent": "g", "shape": shape, "params": params}])
    assert scene.merged["g"].bodies == 1


def test_smooth_and_other_style_words_are_read():
    from oprel.integrations.scene3d.engine.sweep import handle_style
    assert [handle_style(w) for w in ("smooth", "round", "curvy", None, "square", "Box")] == \
        ["round", "round", "round", "round", "box", "box"]
    scene = compile_scene([{"id": "v", "shape": "vase", "params": {"handles": 2, "handle_style": "smooth"}}])
    assert scene.objects["v"].built.mesh.is_watertight


def test_vase_ear_handles_stay_outside_the_body():
    # The ear is drawn between its two points on the wall; tilting a C swung it into the shoulder.
    from trimesh.intersections import mesh_plane
    plain = compile_scene([{"id": "v", "shape": "vase"}]).objects["v"].built.mesh
    eared = compile_scene([{"id": "v", "shape": "vase", "params": {"handles": 2}}]).objects["v"].built.mesh
    widest = lambda m, y: max(abs(p[0]) for seg in mesh_plane(m, [0, 1, 0], [0, y, 0]) for p in seg)  # noqa: E731
    shoulder = 0.3 * 0.6  # 60% up a 0.3 m vase, between the handle's ends
    assert widest(eared, shoulder) > widest(plain, shoulder) + 0.02  # the ear is out beside the wall there


def test_screw_thread_from_a_helix_sweep_fuses_with_its_rod():
    scene = compile_scene([
        {"id": "bolt", "shape": "group", "merge": True},
        {"id": "rod", "parent": "bolt", "shape": "cylinder", "params": {"radius": 0.0085, "height": 0.07}},
        {"id": "thread", "parent": "bolt", "shape": "sweep", "position": [0, 0.005, 0], "params": {
            "profile": "outline", "outline": [[-0.0005, -0.0011], [0.0015, 0], [-0.0005, 0.0011]],
            "path": {"helix": {"radius": 0.0085, "pitch": 0.0025, "turns": 24}}}},
    ])
    lo, hi = scene.objects["thread"].bbox
    assert hi[0] == pytest.approx(0.01, abs=2e-4)  # the outline's tip points away from the axis
    assert scene.objects["thread"].built.mesh.is_watertight and scene.merged["bolt"].bodies == 1


def test_thread_can_be_cut_and_overlapping_turns_name_the_fix():
    scene = compile_scene([{"id": "nut", "shape": "cylinder", "params": {"radius": 0.02, "height": 0.03},
                            "cuts": [{"shape": "sweep", "params": {
                                "profile": "outline", "outline": [[-0.002, -0.001], [0.0015, 0], [-0.002, 0.001]],
                                "path": {"helix": {"radius": 0.02, "pitch": 0.003, "turns": 8}}}}]}])
    assert scene.objects["nut"].built.mesh.is_watertight
    with pytest.raises(SceneError, match="make it shorter than 0.002"):
        compile_scene([{"id": "t", "shape": "sweep", "params": {
            "profile": "outline", "outline": [[0, -0.002], [0.001, 0], [0, 0.002]],
            "path": {"helix": {"radius": 0.01, "pitch": 0.002, "turns": 5}}}}])


def test_params_beside_shape_and_diameters_are_read():
    # Qwen3-8B wrote a pipe run with "length" next to "shape" (it was dropped: "missing 'length'").
    scene = compile_scene([{"id": "p", "shape": "pipe", "length": 1, "diameter": 0.05},
                           {"id": "e", "shape": "elbow", "params": {"diameter": 0.05},
                            "placement": {"relative_to": "p", "side": "end"}}])
    lo, hi = scene.objects["p"].bbox
    assert (hi - lo).round(4).tolist() == [1, 0.05, 0.05]


def test_a_part_that_failed_is_not_also_called_missing():
    with pytest.raises(SceneError) as err:
        compile_scene([{"id": "p", "shape": "pipe", "params": {"radius": 0.05}},
                       {"id": "e", "shape": "elbow", "params": {"radius": 0.05},
                        "placement": {"relative_to": "p", "side": "end"}}])
    assert err.value.errors == ["'p' (pipe): Missing required argument 'length'. Expected pipe(length, radius?, wall?, axis?, size?)"]


def test_direction_words_a_shape_does_not_take_are_dropped_quietly():
    scene = compile_scene([{"id": "p", "shape": "pipe", "params": {"length": 1, "radius": 0.05}},
                           {"id": "e", "shape": "elbow", "params": {"radius": 0.05, "axis": "x"},
                            "placement": {"relative_to": "p", "side": "end"}}])
    assert scene.warnings == []
