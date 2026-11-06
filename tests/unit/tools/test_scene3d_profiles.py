"""Shapes from 2D outlines: lathe, extrude, and the hollow ready-made vessels built on them."""

import numpy as np
import pytest

from oprel.integrations.scene3d.engine import SceneError, compile_scene

VESSELS = ["vase", "bottle", "cup", "bowl", "plant_pot", "wine_glass"]


def _one(shape, params=None, **extra):
    scene = compile_scene([{"id": "o", "shape": shape, "params": params or {}, **extra}])
    return scene, scene.objects["o"]


@pytest.mark.parametrize("shape", VESSELS)
def test_vessels_are_watertight_hollow_and_open_at_the_top(shape):
    scene, co = _one(shape)
    mesh = co.built.mesh
    assert mesh.is_watertight and mesh.volume > 0
    lo, hi = co.bbox
    # Hollow: far less material than a solid of the same outer size would have.
    solid = np.pi * ((hi[0] - lo[0]) / 2) ** 2 * (hi[1] - lo[1])
    assert mesh.volume < 0.6 * solid
    # Open: nothing at the very top reaches the axis (a lid would), only the ring of the rim.
    v = mesh.vertices - [(lo[0] + hi[0]) / 2, 0, (lo[2] + hi[2]) / 2]
    top = v[v[:, 1] > hi[1] - 1e-6]
    assert np.hypot(top[:, 0], top[:, 2]).min() > 0.005
    assert scene.warnings == []


@pytest.mark.parametrize("shape, params, height, width", [
    ("vase", {"height": 0.4, "radius": 0.1, "neck_radius": 0.04}, 0.4, 0.2),
    ("bottle", {"height": 0.3, "radius": 0.04}, 0.3, 0.08),
    ("cup", {"height": 0.12, "radius": 0.05}, 0.12, 0.1),
    ("bowl", {"radius": 0.1, "height": 0.07}, 0.07, 0.2),
    ("wine_glass", {"height": 0.22, "radius": 0.045}, 0.22, 0.09),
])
def test_vessel_size_matches_its_settings(shape, params, height, width):
    _, co = _one(shape, params)
    size = co.bbox[1] - co.bbox[0]
    assert size[1] == pytest.approx(height, abs=1e-6)
    assert size[0] == pytest.approx(width, rel=0.05)


def test_vase_with_a_small_bottom_and_a_wide_top():
    # "a curvy vase with a big radius at the top and a small one at the bottom"
    _, co = _one("vase", {"height": 0.4, "base_radius": 0.04, "radius": 0.07, "neck_radius": 0.09,
                          "top_radius": 0.13})
    mesh = co.built.mesh
    assert mesh.is_watertight and mesh.volume > 0
    v = mesh.vertices
    def width_at(y, band=1e-4):
        ring = v[np.abs(v[:, 1] - y) < band]
        return 2 * np.hypot(ring[:, 0], ring[:, 2]).max()
    assert width_at(0.0) == pytest.approx(0.08, rel=0.02)
    assert width_at(0.4) == pytest.approx(0.26, rel=0.02)
    assert width_at(0.0) < width_at(0.2, band=0.01) < width_at(0.4)


def test_vase_placed_on_a_table_rests_on_its_top():
    scene = compile_scene([
        {"id": "table", "shape": "table"},
        {"id": "vase", "shape": "vase", "placement": {"relative_to": "table", "side": "top"}},
    ])
    assert scene.objects["vase"].bbox[0][1] == pytest.approx(0.75)
    assert scene.warnings == []


def test_lathe_profile_spins_into_a_closed_round_shape():
    _, co = _one("lathe", {"profile": [[0.05, 0], [0.08, 0.1], [0.04, 0.25], [0.06, 0.3]]})
    mesh = co.built.mesh
    assert mesh.is_watertight
    size = co.bbox[1] - co.bbox[0]
    assert size.round(4).tolist() == [0.16, 0.3, 0.16]


def test_smooth_lathe_passes_through_its_points_and_keeps_its_height():
    _, sharp = _one("lathe", {"profile": [[0.05, 0], [0.1, 0.1], [0.03, 0.3]]})
    _, smooth = _one("lathe", {"profile": [[0.05, 0], [0.1, 0.1], [0.03, 0.3]], "smooth": True})
    assert len(smooth.built.mesh.vertices) > len(sharp.built.mesh.vertices)
    assert (smooth.bbox[1] - smooth.bbox[0])[1] == pytest.approx(0.3)
    assert (smooth.bbox[1] - smooth.bbox[0])[0] == pytest.approx(0.2, rel=0.03)


@pytest.mark.parametrize("profile, fragment", [
    ([[0.1, 0]], "at least 2"),
    ([[-0.1, 0], [0.1, 0.2]], "radii must be >= 0"),
    ([[0.1, 0], [0.1, 0.2], [0.05, 0.1], [0.15, 0.1]], "crosses"),
])
def test_bad_lathe_profiles_explain_the_problem(profile, fragment):
    with pytest.raises(SceneError) as info:
        compile_scene([{"id": "l", "shape": "lathe", "params": {"profile": profile}}])
    message = str(info.value)
    assert fragment in message or ("Self-intersection" in message and fragment == "crosses")


def test_extruded_l_bracket_with_a_hole_has_the_right_volume():
    _, co = _one("extrude", {"points": [[0, 0], [0.3, 0], [0.3, 0.05], [0.05, 0.05], [0.05, 0.3], [0, 0.3]],
                             "holes": [[[0.01, 0.01], [0.03, 0.01], [0.03, 0.03]]], "thickness": 0.02})
    mesh = co.built.mesh
    area = 0.3 * 0.05 + 0.05 * 0.25 - 0.5 * 0.02 * 0.02
    assert mesh.is_watertight and mesh.volume == pytest.approx(area * 0.02)
    size = co.bbox[1] - co.bbox[0]
    assert size.round(4).tolist() == [0.3, 0.02, 0.3]  # lying on the ground, thickness up


def test_front_plane_extrude_stands_up():
    _, co = _one("extrude", {"points": [[0, 0], [1, 0], [0.5, 0.8]], "thickness": 0.05, "plane": "front"})
    size = co.bbox[1] - co.bbox[0]
    assert size.round(4).tolist() == [1.0, 0.8, 0.05]


def test_backwards_outline_order_still_works():
    clockwise = [[0, 0], [0, 0.2], [0.2, 0.2], [0.2, 0]]
    _, co = _one("extrude", {"points": clockwise, "thickness": 0.01})
    assert co.built.mesh.volume == pytest.approx(0.2 * 0.2 * 0.01)


@pytest.mark.parametrize("params, fragment", [
    ({"points": [[0, 0], [1, 1], [1, 0], [0, 1]], "thickness": 0.1}, "Self-intersection"),
    ({"points": [[0, 0], [1, 0], [1, 1], [0, 1]], "holes": [[[2, 2], [3, 2], [3, 3]]], "thickness": 0.1}, "Hole"),
    ({"points": [[0, 0], [1, 0], [2, 0]], "thickness": 0.1}, "not a valid outline"),  # all on one line
])
def test_bad_extrude_outlines_explain_the_problem(params, fragment):
    with pytest.raises(SceneError) as info:
        compile_scene([{"id": "e", "shape": "extrude", "params": params}])
    assert fragment in str(info.value)


def test_chairs_go_around_a_vase_on_a_circle():
    # Round shapes (lathe-based) get a circular 'around' layout, like round tables.
    scene = compile_scene([
        {"id": "pot", "shape": "plant_pot", "params": {"radius": 0.3, "height": 0.4}},
        {"id": "stools", "shape": "stool", "pattern": {"type": "around", "relative_to": "pot", "count": 5}},
    ])
    dists = {round(float(np.hypot(m[0, 3], m[2, 3])), 3) for m in scene.objects["stools"].instances}
    assert len(dists) == 1
