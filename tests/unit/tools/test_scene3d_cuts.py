"""Holes and cuts: shapes removed from an object with exact mesh booleans."""

import math

import numpy as np
import pytest

from oprel.integrations.scene3d.engine import SceneError, compile_scene

PLATE = {"id": "plate", "shape": "box", "params": {"width": 0.4, "height": 0.02, "depth": 0.3}}
PLATE_VOLUME = 0.4 * 0.02 * 0.3


def _cut(cuts, base=PLATE):
    scene = compile_scene([{**base, "cuts": cuts}])
    co = scene.objects[base["id"]]
    assert co.built.mesh.is_watertight
    return scene, co


def _hole(radius, height):
    return math.pi * radius ** 2 * height


def test_hole_without_a_height_goes_right_through():
    scene, co = _cut([{"shape": "cylinder", "params": {"radius": 0.05}}])
    assert co.built.mesh.volume == pytest.approx(PLATE_VOLUME - _hole(0.05, 0.02), rel=0.01)
    assert scene.warnings == []


def test_hole_exactly_as_thick_as_the_plate_cuts_cleanly():
    # Flush faces would otherwise leave a skin or a partial hole.
    _, co = _cut([{"shape": "cylinder", "params": {"radius": 0.05, "height": 0.02}, "position": [0, 0.01, 0]}])
    assert co.built.mesh.volume == pytest.approx(PLATE_VOLUME - _hole(0.05, 0.02), rel=0.01)


def test_pocket_from_the_top_face():
    _, co = _cut([{"shape": "box", "params": {"width": 0.1, "height": 0.01, "depth": 0.1}, "side": "top"}])
    assert co.built.mesh.volume == pytest.approx(PLATE_VOLUME - 0.1 * 0.01 * 0.1, rel=1e-4)
    # The bottom is untouched: the object keeps its size and frame.
    assert co.built.mesh.bounds.ravel().tolist() == pytest.approx([-0.2, 0, -0.15, 0.2, 0.02, 0.15])


def test_notch_in_the_front_edge():
    _, co = _cut([{"shape": "box", "params": {"width": 0.05, "depth": 0.04}, "side": "front"}])
    assert co.built.mesh.volume == pytest.approx(PLATE_VOLUME - 0.05 * 0.02 * 0.04, rel=1e-4)


def test_bolt_circle_with_a_radial_pattern():
    flange = {"id": "flange", "shape": "cylinder", "params": {"radius": 0.15, "height": 0.02}}
    scene, co = _cut([
        {"shape": "cylinder", "params": {"radius": 0.05}},
        {"shape": "cylinder", "params": {"radius": 0.01}, "pattern": {"type": "radial", "count": 6, "radius": 0.11}},
    ], flange)
    solid = _hole(0.15, 0.02)
    assert co.built.mesh.volume == pytest.approx(solid - _hole(0.05, 0.02) - 6 * _hole(0.01, 0.02), rel=0.02)
    assert scene.warnings == []


def test_rotated_hole_runs_front_to_back():
    block = {"id": "block", "shape": "box", "params": {"width": 0.4, "height": 0.2, "depth": 0.3}}
    _, co = _cut([{"shape": "cylinder", "params": {"radius": 0.03}, "rotation": [90, 0, 0],
                   "position": [0, 0.1, 0]}], block)
    assert co.built.mesh.volume == pytest.approx(0.4 * 0.2 * 0.3 - _hole(0.03, 0.3), rel=0.01)
    # It goes along Z: the hole's rim (radius 0.03 around x=0, y=0.1) is on the front face.
    v = co.built.mesh.vertices
    front = v[abs(v[:, 2] - 0.15) < 1e-6]
    rim = abs(np.hypot(front[:, 0], front[:, 1] - 0.1) - 0.03) < 1e-4
    assert rim.sum() >= 16


def test_window_in_a_wall():
    wall = {"id": "wall", "shape": "box", "params": {"width": 3, "height": 2.5, "depth": 0.2}}
    _, co = _cut([{"shape": "box", "params": {"width": 1, "height": 1.2}, "position": [0, 1.4, 0]}], wall)
    assert co.built.mesh.volume == pytest.approx(3 * 2.5 * 0.2 - 1 * 1.2 * 0.2, rel=1e-4)


def test_ready_made_objects_can_be_cut():
    # A table is several overlapping parts; they are merged before cutting.
    scene, co = _cut([{"shape": "cylinder", "params": {"radius": 0.03}, "side": "top"}],
                     {"id": "table", "shape": "table"})
    assert scene.warnings == []
    assert co.built.mesh.volume > 0


def test_cut_that_misses_is_reported_with_the_objects_extent():
    scene, co = _cut([{"shape": "cylinder", "params": {"radius": 0.05}, "position": [2, 0, 0]}])
    assert co.built.mesh.volume == pytest.approx(PLATE_VOLUME)
    assert len(scene.warnings) == 1
    assert "cuts[0]" in scene.warnings[0] and "misses" in scene.warnings[0] and "from x = -0.2 to 0.2" in scene.warnings[0]


def test_pattern_copies_that_miss_are_counted():
    scene, _ = _cut([{"shape": "cylinder", "params": {"radius": 0.01},
                      "pattern": {"type": "linear", "count": 4, "step": [0.15, 0, 0]}}])
    # Copies at x = 0, 0.15, 0.30, 0.45: the last two are past the plate's edge at 0.2.
    assert any("2 of the 4 copies" in w for w in scene.warnings)


def test_cut_that_splits_the_object_is_reported():
    scene, _ = _cut([{"shape": "box", "params": {"width": 0.02}}])
    assert any("2 separate pieces" in w for w in scene.warnings)


def test_cut_that_removes_everything_is_an_error():
    with pytest.raises(SceneError, match="remove the whole object"):
        compile_scene([{**PLATE, "cuts": [{"shape": "box"}]}])


@pytest.mark.parametrize("cut, message", [
    ({"shape": "chair"}, "can't be cut out"),
    ({"shape": "cylinder"}, "radius"),
    ({"shape": "cylinder", "params": {"radius": 0.01}, "side": "up"}, "side must use"),
    ({"shape": "cylinder", "params": {"radius": 0.01}, "pattern": {"type": "around", "count": 3}}, "not 'around'"),
])
def test_bad_cuts_are_errors_the_model_can_fix(cut, message):
    with pytest.raises(SceneError, match=message):
        compile_scene([{**PLATE, "cuts": [cut]}])


def test_a_group_cannot_be_cut():
    with pytest.raises(SceneError, match="put the cuts on one of its parts"):
        compile_scene([{"id": "g", "shape": "group", "cuts": [{"shape": "box"}]}])


def test_cut_object_still_places_and_patterns_like_any_other():
    scene = compile_scene([
        {**PLATE, "cuts": [{"shape": "cylinder", "params": {"radius": 0.05}}],
         "pattern": {"type": "linear", "count": 2, "step": [1, 0, 0]}},
        {"id": "cup", "shape": "cup", "placement": {"relative_to": "plate", "side": "top", "align": "left"}},
    ])
    assert len(scene.objects["plate"].instances) == 2
    assert scene.objects["cup"].bbox[0][1] == pytest.approx(0.02)


def test_hole_measured_from_a_corner_is_reported():
    # Qwen3-8B put a 30 cm plate's "hole in the middle" at [0.15, 0, 0.15]: a quarter hole on the corner.
    plate = {"id": "plate", "shape": "box", "params": {"width": 0.3, "height": 0.01, "depth": 0.3}}
    scene, _ = _cut([{"shape": "cylinder", "params": {"radius": 0.05}, "position": [0.15, 0, 0.15]}], plate)
    (warning,) = scene.warnings
    assert "breaks open through the object's right face" in warning and "not a corner" in warning


@pytest.mark.parametrize("cut", [
    {"shape": "box", "params": {"width": 0.05, "depth": 0.04}, "side": "front"},             # notch
    {"shape": "cylinder", "params": {"radius": 0.005}, "rotation": [90, 0, 0], "position": [0, 0.01, 0]},
    {"shape": "box", "params": {"width": 0.1, "height": 0.01, "depth": 0.1}, "side": "top"},  # pocket
])
def test_cuts_meant_to_open_onto_a_face_are_not_reported(cut):
    scene, _ = _cut([cut])
    assert scene.warnings == []


def test_door_opens_onto_the_bottom_of_a_wall():
    wall = {"id": "wall", "shape": "box", "params": {"width": 4, "height": 2.5, "depth": 0.2}}
    scene, co = _cut([{"shape": "box", "params": {"width": 0.9, "height": 2.1}, "side": "bottom",
                       "position": [-1, 0, 0]},
                      {"shape": "box", "params": {"width": 1.2, "height": 1}, "position": [1, 1.5, 0]}], wall)
    assert scene.warnings == []
    assert co.built.mesh.volume == pytest.approx((4 * 2.5 - 0.9 * 2.1 - 1.2 * 1) * 0.2, rel=1e-4)


def test_bolt_holes_in_a_row_on_a_round_flange_suggest_a_ring():
    # Qwen3-8B wrote 6 bolt holes as a 1-row grid 0.12 apart on a 20 cm flange.
    flange = {"id": "flange", "shape": "circle", "params": {"radius": 0.1, "thickness": 0.02}}
    scene, _ = _cut([{"shape": "cylinder", "params": {"radius": 0.006},
                      "pattern": {"type": "grid", "count": 6, "rows": 1, "spacing": [0.12, 0.12]}}], flange)
    (warning,) = scene.warnings
    assert "4 of the 6 copies" in warning and "[-0.3, 0.01, 0]" in warning
    assert '"type": "radial", "count": 6, "radius": 0.07' in warning


def test_hole_at_the_bottom_says_which_height_centres_it():
    # Qwen3-8B drilled "front to back" at position [0, 0, 0]: a groove along the bottom.
    block = {"id": "block", "shape": "box", "params": {"width": 0.2, "height": 0.1, "depth": 0.1}}
    scene, _ = _cut([{"shape": "cylinder", "params": {"radius": 0.02}, "rotation": [90, 0, 0],
                      "position": [0, 0, 0]}], block)
    (warning,) = scene.warnings
    assert "bottom face" in warning and "use y = 0.05" in warning


def test_two_identical_cuts_are_an_error():
    window = {"shape": "box", "params": {"width": 1, "height": 1.5}, "position": [1.5, 1.2, 0]}
    with pytest.raises(SceneError, match=r"cuts\[1\] is the same as cuts\[0\]"):
        compile_scene([{"id": "wall", "shape": "box", "params": {"width": 4, "height": 2.5, "depth": 0.2},
                        "cuts": [window, dict(window)]}])


def test_window_measured_from_the_wall_end_gets_the_corrected_number():
    wall = {"id": "wall", "shape": "box", "params": {"width": 4, "height": 2.5, "depth": 0.2}}
    scene, _ = _cut([{"shape": "box", "params": {"width": 1, "height": 1.5}, "position": [2.5, 1.2, 0]}], wall)
    (warning,) = scene.warnings
    assert "x = 2.5 is outside it" in warning and "subtract 2 from every cut's x: cuts[0] x = 0.5." in warning


def test_linear_row_that_runs_off_the_end_gets_a_centred_start():
    # Qwen3-8B: 3 peg holes 0.333 apart starting from the middle of a 1 m board.
    board = {"id": "board", "shape": "box", "params": {"width": 1, "height": 0.02, "depth": 0.2}}
    scene, _ = _cut([{"shape": "cylinder", "params": {"radius": 0.01},
                      "pattern": {"type": "linear", "count": 3, "step": [0.333, 0, 0]}}], board)
    (warning,) = scene.warnings
    assert "run off the object" in warning and '"position": [-0.333, 0.01, 0]' in warning


def test_wall_fix_names_the_new_x_of_every_cut():
    # Qwen3-8B applied "this one: x = 0.5" to the wrong cut; each cut now gets its own number.
    wall = {"id": "wall", "shape": "box", "params": {"width": 4, "height": 2.5, "depth": 0.2}}
    scene, _ = _cut([{"shape": "box", "params": {"width": 1, "height": 1.5}, "position": [1.5, 0.75, 0]},
                     {"shape": "box", "params": {"width": 1, "height": 1.5}, "position": [2.5, 0.75, 0]}], wall)
    assert any("cuts[0] x = -0.5, cuts[1] x = 0.5" in w for w in scene.warnings)


def test_start_position_put_inside_the_pattern_is_used():
    board = {"id": "board", "shape": "box", "params": {"width": 1, "height": 0.02, "depth": 0.2}}
    scene, _ = _cut([{"shape": "cylinder", "params": {"radius": 0.01},
                      "pattern": {"type": "linear", "count": 3, "step": [0.333, 0, 0], "position": [-0.333, 0.01, 0]}}],
                    board)
    assert scene.warnings == []
