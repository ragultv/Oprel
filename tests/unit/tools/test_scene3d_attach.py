"""
Handles built onto the real surface of any solid (engine/attach.py): every shape, side,
orientation and style; rotated, grouped and thin-walled targets; and the errors.

Each case checks what a CAD user would: the handle and the object fuse into one body, the handle
doesn't run through the object (their overlap is only its two tips), and on a thin wall the tips
don't come out on the inside.
"""

import numpy as np
import pytest

from oprel.integrations.scene3d.engine import SceneError, compile_scene
from oprel.integrations.scene3d.engine.cuts import as_solid
from oprel.integrations.scene3d.engine.frame import transform_points

TARGETS = {
    "box": {"width": 0.3, "height": 0.25, "depth": 0.2},
    "cylinder": {"radius": 0.1, "height": 0.3},
    "cone": {"radius": 0.15, "height": 0.35},
    "sphere": {"radius": 0.12},
    "capsule": {"radius": 0.08, "height": 0.35},
    "vase": {"height": 0.5, "radius": 0.12, "neck_radius": 0.05},
    "cup": {},
    "bottle": {},
    "lathe": {"profile": [[0.08, 0], [0.12, 0.1], [0.07, 0.25], [0.09, 0.3]], "smooth": True},
    "extrude": {"points": [[-0.15, -0.1], [0.15, -0.1], [0.1, 0.1], [-0.1, 0.1]], "thickness": 0.2},
}


def _scene(target_shape, target_params, handle_params=None, placement=None, rotation=None):
    return compile_scene([
        {"id": "g", "shape": "group", "merge": True},
        {"id": "t", "parent": "g", "shape": target_shape, "params": target_params,
         **({"rotation": rotation} if rotation else {})},
        {"id": "h", "parent": "g", "shape": "handle", "params": handle_params or {},
         "placement": {"relative_to": "t", "side": "right", **(placement or {})}},
    ])


def _world(scene, oid):
    co = scene.objects[oid]
    m = co.built.mesh.copy()
    m.apply_transform(co.instances[0])
    return m


def _overlap_share(scene):
    """How much of the handle's volume lies inside the object (only its tips should)."""
    body, grip = as_solid(_world(scene, "t"), "t"), as_solid(_world(scene, "h"), "h")
    return (body.volume() + grip.volume() - (body + grip).volume()) / grip.volume()


@pytest.mark.parametrize("shape", sorted(TARGETS))
@pytest.mark.parametrize("style", ["round", "box"])
def test_handle_fuses_onto_every_shape_without_running_through_it(shape, style):
    scene = _scene(shape, TARGETS[shape], {"style": style})
    assert scene.objects["h"].built.mesh.is_watertight
    assert scene.merged["g"].bodies == 1
    assert _overlap_share(scene) < 0.12
    assert scene.objects["h"].bbox[1][0] > scene.objects["t"].bbox[1][0]  # sticks out on the right


@pytest.mark.parametrize("side, axis, outward", [("right", 0, 1), ("left", 0, -1), ("front", 2, 1), ("back", 2, -1)])
@pytest.mark.parametrize("orientation", ["vertical", "horizontal"])
def test_every_side_and_orientation(side, axis, outward, orientation):
    scene = _scene("vase", TARGETS["vase"], {"orientation": orientation}, {"side": side})
    assert scene.merged["g"].bodies == 1
    lo, hi = scene.objects["h"].bbox
    tlo, thi = scene.objects["t"].bbox
    assert (hi[axis] > thi[axis]) if outward > 0 else (lo[axis] < tlo[axis])
    size = hi - lo
    if orientation == "vertical":
        assert size[1] > size[2 - axis]  # ends one above the other
    else:
        assert size[2 - axis] > size[1]  # ends side by side


def test_handle_over_the_top_of_a_box_is_a_suitcase_handle():
    scene = _scene("box", {"width": 0.5, "height": 0.3, "depth": 0.2}, placement={"side": "top"})
    lo, hi = scene.objects["h"].bbox
    assert lo[1] < 0.3 < hi[1]  # its ends in the top, its loop above it
    assert hi[0] - lo[0] > hi[2] - lo[2]  # along the box's longer side
    assert scene.merged["g"].bodies == 1


@pytest.mark.parametrize("vertical", ["top", "center", "bottom"])
def test_height_on_the_object(vertical):
    scene = _scene("cylinder", TARGETS["cylinder"], placement={"vertical": vertical})
    lo, hi = scene.objects["h"].bbox
    mid = (lo[1] + hi[1]) / 2
    assert {"top": mid > 0.18, "center": 0.11 < mid < 0.19, "bottom": mid < 0.12}[vertical]


def test_offset_moves_it_along_and_across():
    base = _scene("box", TARGETS["box"], {"height": 0.1}).objects["h"].bbox
    moved = _scene("box", TARGETS["box"], {"height": 0.1}, {"offset": [0, 0.05, 0.04]}).objects["h"].bbox
    assert moved[0][1] == pytest.approx(base[0][1] + 0.05, abs=1e-6)
    assert moved[0][2] == pytest.approx(base[0][2] + 0.04, abs=1e-6)


def test_sized_to_the_object_unless_given():
    small = _scene("cup", {}).objects["h"].bbox
    big = _scene("vase", {"height": 1.2, "radius": 0.25, "neck_radius": 0.1}).objects["h"].bbox
    assert (big[1] - big[0])[1] > 5 * (small[1] - small[0])[1]
    given = _scene("vase", {"height": 1.2, "radius": 0.25, "neck_radius": 0.1}, {"height": 0.2}).objects["h"]
    lo, hi = given.bbox
    assert 0.2 < hi[1] - lo[1] < 0.3  # its ends 0.2 apart (plus the bar and the ear's rise)


def test_tips_stay_inside_a_thin_wall():
    # An upright pipe: hollow with a 3 mm wall. The bar is 9 mm thick; its tips must not reach the bore.
    scene = _scene("pipe", {"length": 0.4, "radius": 0.06, "wall": 0.003, "axis": "y"}, {"thickness": 0.009})
    co = scene.objects["h"]
    pts = transform_points(co.instances[0], np.asarray(co.built.mesh.vertices))
    radial = np.hypot(pts[:, 0], pts[:, 2])
    assert radial.min() > 0.06 - 0.003 + 1e-5
    assert scene.merged["g"].bodies == 1


def test_rotated_object():
    scene = _scene("box", TARGETS["box"], rotation=[0, 30, 0])
    assert scene.merged["g"].bodies == 1
    assert _overlap_share(scene) < 0.12


def test_handle_inside_the_object_it_is_on():
    scene = compile_scene([
        {"id": "mug", "shape": "cup", "merge": False},
        {"id": "h", "parent": "mug", "shape": "handle", "placement": {"relative_to": "mug", "side": "right"}},
    ])
    lo, hi = scene.objects["h"].bbox
    assert hi[0] > 0.04 + 0.01  # out past the cup's 4 cm rim (the mug's own box includes its handle)


def test_handle_follows_its_object_when_it_moves():
    a = _scene("cup", {}).objects["h"].bbox
    scene = compile_scene([
        {"id": "t", "shape": "cup", "position": [1, 0, 2]},
        {"id": "h", "shape": "handle", "placement": {"relative_to": "t", "side": "right"}},
    ])
    b = scene.objects["h"].bbox
    assert b[0] - a[0] == pytest.approx([1, 0, 2], abs=1e-6)


def test_end_off_the_object_names_the_range():
    with pytest.raises(SceneError, match=r"misses 't': it has a surface there from"):
        _scene("cylinder", TARGETS["cylinder"], {"height": 0.5})


def test_ends_too_close_for_the_bar_name_the_minimum():
    with pytest.raises(SceneError, match=r"use a height of at least"):
        _scene("cylinder", TARGETS["cylinder"], {"height": 0.02, "thickness": 0.02})


def test_style_words_and_orientation_words_are_read():
    for style in ("smooth", "curved", "square"):
        assert _scene("cup", {}, {"style": style}).merged["g"].bodies == 1
    assert _scene("cup", {}, {"orientation": "Horizontal"}).merged["g"].bodies == 1


@pytest.mark.parametrize("shape, params", [
    ("vase", {"handles": 2}), ("vase", {"handles": 2, "height": 1.2, "radius": 0.2, "handle_style": "box"}),
    ("cup", {"handle": True}), ("cup", {"handles": 2, "handle_style": "square"}),
    ("bowl", {"handles": 2}), ("bowl", {"handles": 2, "radius": 0.2, "height": 0.1}),
])
def test_vessels_own_handles_use_the_same_engine(shape, params):
    scene = compile_scene([{"id": "g", "shape": "group", "merge": True},
                           {"id": "o", "parent": "g", "shape": shape, "params": params}])
    assert scene.merged["g"].bodies == 1
    assert scene.objects["o"].built.mesh.is_watertight or scene.merged["g"].mesh.is_watertight


def test_one_handle_on_a_vase_is_a_jug_handle_from_the_rim():
    scene = compile_scene([{"id": "g", "shape": "group", "merge": True},
                           {"id": "jug", "parent": "g", "shape": "vase", "params": {"height": 0.6, "handles": 1}}])
    lo, hi = scene.objects["jug"].bbox
    assert scene.merged["g"].bodies == 1
    plo, phi = compile_scene([{"id": "v", "shape": "vase", "params": {"height": 0.6}}]).objects["v"].bbox
    assert hi[0] - lo[0] > phi[0] - plo[0] + 0.03  # the handle sticks out (the box is re-centred, so: width)
    verts = scene.objects["jug"].built.mesh.vertices
    outer = verts[verts[:, 0] > verts[:, 0].max() - 0.03]  # the handle's far side
    assert verts[verts[:, 0] > verts[:, 0].max() - 0.06][:, 1].max() > 0.5 > outer[:, 1].min()  # rim down to body


def test_cooking_pot_lugs():
    scene = compile_scene([{"id": "g", "shape": "group", "merge": True},
                           {"id": "pot", "parent": "g", "shape": "plant_pot", "params": {"handles": 2}}])
    lo, hi = scene.objects["pot"].bbox
    assert scene.merged["g"].bodies == 1 and hi[0] - lo[0] > hi[2] - lo[2] + 0.01  # lugs on both sides in X


def test_a_param_the_shape_does_not_have_is_reported_not_dropped():
    scene = compile_scene([{"id": "b", "shape": "bench", "params": {"handles": 2}}])
    assert any("a bench has no 'handles', so it was ignored" in w for w in scene.warnings)


def test_amphora_with_a_funnel_rim_gets_a_neck_and_a_warning():
    # Verbatim from Qwen3-8B, "a tall amphora ... narrow neck and flared rim, two smooth handles".
    params = {"height": 0.6, "radius": 0.15, "neck_radius": 0.05, "top_radius": 0.2, "handles": 2,
              "handle_style": "smooth"}
    scene = compile_scene([{"id": "g", "shape": "group", "merge": True},
                           {"id": "amphora", "parent": "g", "shape": "vase", "params": params}])
    assert scene.merged["g"].bodies == 1
    assert any("top_radius 0.075" in w for w in scene.warnings)
    # A real neck under the flare: as narrow at 75% of the height as at 66%.
    from trimesh.intersections import mesh_plane
    mesh = compile_scene([{"id": "v", "shape": "vase", "params": {**params, "handles": 0}}]).objects["v"].built.mesh
    width = lambda y: max(abs(p[0]) for seg in mesh_plane(mesh, [0, 1, 0], [0, y, 0]) for p in seg)  # noqa: E731
    assert width(0.75 * 0.6) < 0.07 and abs(width(0.75 * 0.6) - width(0.66 * 0.6)) < 0.01
