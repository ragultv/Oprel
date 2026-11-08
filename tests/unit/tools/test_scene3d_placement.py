"""Placement the engine works out so the model doesn't have to: between two objects, gaps, warnings."""

import numpy as np
import pytest

from oprel.integrations.scene3d.engine import SceneError, compile_scene
from oprel.integrations.scene3d.engine.frame import transform_points

BIG = {"id": "big", "shape": "cylinder", "params": {"radius": 0.2, "height": 1.0}}
THIN = {"id": "thin", "shape": "cylinder", "params": {"radius": 0.07, "height": 1.1}}
PLATE = {"id": "plate", "shape": "box", "params": {"height": 0.2, "depth": 0.02},
         "placement": {"between": ["thin", "big"]}}


def _bounds(scene, oid):
    lo, hi = scene.objects[oid].bbox
    return [round(float(v), 3) for v in (*lo, *hi)]


def test_box_between_two_objects_joins_them_centre_to_centre():
    # "two cylinders with a 15 cm gap, joined at the bottom by a plate": no arithmetic for the model.
    scene = compile_scene([BIG, {**THIN, "placement": {"relative_to": "big", "side": "left", "gap": 0.15}}, PLATE])
    assert scene.warnings == []
    assert _bounds(scene, "thin") == [-0.49, 0, -0.07, -0.35, 1.1, 0.07]
    assert _bounds(scene, "plate") == [-0.42, 0, -0.01, 0, 0.2, 0.01]
    assert scene.objects["plate"].spec["params"]["width"] == pytest.approx(0.42)


def test_box_between_diagonal_objects_is_turned_to_face_along_them():
    scene = compile_scene([BIG, {**THIN, "position": [-0.3, 0, -0.3]}, PLATE])
    co = scene.objects["plate"]
    half = co.spec["params"]["width"] / 2
    assert half * 2 == pytest.approx(0.3 * 2 ** 0.5)
    # Its long side runs along the diagonal: the two ends are at the two centres.
    ends = transform_points(co.instances[0], np.array([[-half, 0.1, 0], [half, 0.1, 0]]))
    assert sorted(map(tuple, ends[:, [0, 2]].round(3).tolist())) == [(-0.3, -0.3), (0.0, 0.0)]


def test_spanning_box_follows_when_the_objects_move():
    scene = compile_scene([BIG, {**THIN, "position": [-1.0, 0, 0]}, PLATE])
    assert _bounds(scene, "plate") == [-1.0, 0, -0.01, 0, 0.2, 0.01]


def test_other_shapes_between_keep_their_size_and_go_in_the_middle():
    scene = compile_scene([BIG, {**THIN, "position": [-1.0, 0, 0]},
                           {"id": "vase", "shape": "vase", "placement": {"between": ["thin", "big"]}}])
    lo, hi = scene.objects["vase"].bbox
    assert (lo[0] + hi[0]) / 2 == pytest.approx(-0.5) and lo[1] == pytest.approx(0)


def test_beam_between_the_tops_of_two_posts():
    post = {"shape": "box", "params": {"width": 0.1, "height": 2, "depth": 0.1}}
    scene = compile_scene([{**post, "id": "p1", "position": [-1, 0, 0]}, {**post, "id": "p2", "position": [1, 0, 0]},
                           {"id": "beam", "shape": "box", "params": {"height": 0.1, "depth": 0.1},
                            "placement": {"between": ["p1", "p2"], "vertical": "top"}}])
    assert _bounds(scene, "beam") == [-1, 1.9, -0.05, 1, 2, 0.05]


@pytest.mark.parametrize("placement, message", [
    ({"between": ["thin"]}, "needs two different ids"),
    ({"between": ["thin", "thin"]}, "needs two different ids"),
    ({"between": ["thin", "nope"]}, "'nope' does not exist"),
    ({"side": "left"}, "needs 'relative_to'"),
])
def test_bad_between_is_an_error_the_model_can_fix(placement, message):
    with pytest.raises(SceneError, match=message):
        compile_scene([BIG, {**THIN, "position": [-1, 0, 0]},
                       {**PLATE, "params": {"width": 0.3, "height": 0.2, "depth": 0.02}, "placement": placement}])


def test_cylinder_standing_inside_another_gets_the_placement_that_fixes_it():
    # Qwen3-8B read "a 15 cm gap" as 15 cm between the centres.
    scene = compile_scene([BIG, {**THIN, "position": [-0.15, 0, 0]}])
    (warning,) = scene.warnings
    assert "'thin' stands inside 'big'" in warning
    assert '"placement": {"relative_to": "big", "side": "left", "gap": 0.15}' in warning


def test_things_meant_to_overlap_are_not_reported_as_inside():
    scene = compile_scene([
        {"id": "pot", "shape": "plant_pot"},
        {"id": "stem", "shape": "cylinder", "params": {"radius": 0.01, "height": 0.4}},   # in a hollow pot
        {"id": "big", "shape": "cylinder", "params": {"radius": 0.2, "height": 1.0}, "position": [2, 0, 0]},
        {"id": "plate", "shape": "box", "params": {"width": 0.15, "height": 0.2, "depth": 0.02}, "position": [2, 0, 0]},
    ])
    assert not any("stands inside" in w for w in scene.warnings)


def test_side_bottom_into_the_ground_explains_what_bottom_means():
    scene = compile_scene([BIG, {"id": "plate", "shape": "box", "params": {"width": 0.4, "height": 0.2, "depth": 0.02},
                                 "placement": {"relative_to": "big", "side": "bottom"}}])
    (warning,) = scene.warnings
    assert 'side "bottom" puts it underneath' in warning and '"between"' in warning


def test_joining_plate_drawn_from_coordinates_that_misses_one_gets_the_between_fix():
    # Live Qwen3-8B: plate at the old midpoint, left behind when the thin cylinder was moved.
    scene = compile_scene([BIG, {**THIN, "placement": {"relative_to": "big", "side": "left", "gap": 0.15}},
                           {"id": "box", "shape": "box", "params": {"width": 0.15, "height": 0.2, "depth": 0.02},
                            "position": [-0.075, 0, 0], "label": "Joining plate"}])
    (warning,) = scene.warnings
    assert "'box' touches only 'big'" in warning and '"between": ["big", "thin"]' in warning


def test_flat_base_plate_with_one_thing_on_it_is_not_a_loose_connector():
    scene = compile_scene([{"id": "base_plate", "shape": "box", "params": {"width": 1, "height": 0.02, "depth": 1}},
                           {"id": "vase", "shape": "vase", "placement": {"relative_to": "base_plate", "side": "top"}},
                           {"id": "lamp", "shape": "lamp", "position": [3, 0, 0]}])
    assert scene.warnings == []
