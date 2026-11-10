"""
Row 6: standard parts (bolt, nut, washer, gear, flange) computed from their standards, and how
they fit together: bolts seat on parts, nuts thread on under what they hold, gears mesh.
"""

import json
import math

import numpy as np
import pytest
from trimesh import transformations as tf

from oprel.integrations.scene3d.engine import cuts, intent, standards
from oprel.integrations.scene3d.engine.compiler import SceneError, _build, compile_scene
from oprel.integrations.scene3d.engine.cuts import as_solid
from oprel.integrations.scene3d.engine.registry import ToolArgumentError, get_shape
from oprel.integrations.scene3d.engine.shapes.parts import gear_outline, thread_mesh


def build(shape, **params):
    return _build(shape, json.dumps(get_shape(shape).validate(params), sort_keys=True))


def world(co, i=0):
    m = co.built.mesh.copy()
    m.apply_transform(co.instances[i])
    return m


def overlap(a, b):
    return (as_solid(a, "a") ^ as_solid(b, "b")).volume()


# ---------------------------------------------------------------------------
# standards
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("size", ["M10", "m10", " M10 ", "M10x1.5", 10, 0.01, "10mm", "10"])
def test_metric_sizes_are_read_loosely(size):
    row, pitch = standards.metric(size)
    assert row.name == "M10" and pitch == 1.5 and row.s == 16


def test_fine_pitch_and_nonstandard_sizes():
    assert standards.metric("M10x1.25")[1] == 1.25
    with pytest.raises(ToolArgumentError, match="M11 is not a standard ISO size; use M10 or M12"):
        standards.metric("M11")
    with pytest.raises(ToolArgumentError, match="pitch 2 mm doesn't fit M10"):
        standards.metric("M10x2")


@pytest.mark.parametrize("size,dn", [("DN50", 50), (50, 50), ("2 inch", 50), ('2"', 50), ("NPS 4", 100),
                                     (0.0603, 50), ("dn 100", 100)])
def test_nominal_pipe_sizes(size, dn):
    assert standards.nominal(size).dn == dn


def test_unknown_nominal_size_names_the_nearest():
    with pytest.raises(ToolArgumentError, match="DN60 is not in the table; use DN50 or DN65"):
        standards.nominal("DN60")


def test_bolts_are_threaded_iso_4014_style():
    assert standards.thread_length(10, 100) == 26   # 2d + 6, rest plain shank
    assert standards.thread_length(10, 30) == 30    # short: threaded all the way (ISO 4017)


# ---------------------------------------------------------------------------
# shapes
# ---------------------------------------------------------------------------

def test_thread_follows_the_iso_profile():
    pitch, depth = 0.0015, 0.6134 * 0.0015
    m = thread_mesh(0.005, pitch, 12 * pitch, depth, tip_chamfer=False)  # 8 rows a pitch
    r = np.hypot(m.vertices[:, 0], m.vertices[:, 1])[:-2]  # without the two cap centres
    assert m.is_watertight
    assert r.max() == pytest.approx(0.005) and r.min() == pytest.approx(0.005 - depth)
    # one turn rises one pitch: the same angle a pitch higher has the same radius
    i = 3
    n = 32
    assert r[i] == pytest.approx(r[i + 8 * n])


@pytest.mark.parametrize("head,height", [("hex", 0.0064), ("socket", 0.010), ("countersunk", 0.0)])
def test_bolt_heads_have_standard_sizes(head, height):
    b = build("bolt", size="M10", length=0.05, head=head)
    assert b.mesh.is_watertight
    lo, hi = b.mesh.bounds
    assert hi[1] - lo[1] == pytest.approx(0.05 + height, abs=1e-6)
    seat, _ = b.port("seat")
    assert seat[1] == pytest.approx(0.05 + lo[1])
    if head == "hex":
        assert hi[2] - lo[2] == pytest.approx(0.016, abs=1e-5)  # 16 across flats


def test_bolt_length_in_millimetres_is_read_and_said():
    b = build("bolt", size="M6", length=30)
    assert np.ptp(b.mesh.bounds[:, 1]) == pytest.approx(0.034)
    assert b.notes == ("read length 30 as 30 mm",)


def test_bolt_too_short_for_its_size():
    with pytest.raises(ToolArgumentError, match="at least 20 mm long"):
        build("bolt", size="M10", length=0.01)


def test_nut_and_washer_sizes():
    n, w = build("nut", size="M10"), build("washer", size="M10")
    assert n.mesh.is_watertight and w.mesh.is_watertight
    assert np.ptp(n.mesh.bounds[:, 1]) == pytest.approx(0.0084)
    assert np.ptp(w.mesh.bounds[:, 1]) == pytest.approx(0.002, abs=1e-6)
    assert np.ptp(w.mesh.bounds[:, 0]) == pytest.approx(0.020, abs=1e-5)


def test_gear_outline_and_sizes():
    g = build("gear", teeth=20, module=2)
    assert g.mesh.is_watertight
    lo, hi = g.mesh.bounds
    assert hi[0] - lo[0] == pytest.approx(0.044, abs=2e-5)  # tip diameter m (z + 2)
    assert hi[2] - lo[2] == pytest.approx(0.020)            # face width 10 m
    pts = gear_outline(20, 0.002)
    r = np.hypot(pts[:, 0], pts[:, 1])
    assert r.min() == pytest.approx(0.02 - 0.0025) and r.max() == pytest.approx(0.022)


def test_gear_from_its_pitch_radius_and_module_in_metres():
    a = build("gear", teeth=20, radius=0.02)
    b = build("gear", teeth=20, module=0.002)
    assert np.allclose(a.mesh.bounds, b.mesh.bounds)


def test_flange_dimensions_and_ends():
    f = build("flange", size="DN50")
    assert f.mesh.is_watertight
    lo, hi = f.mesh.bounds
    assert hi[1] - lo[1] == pytest.approx(0.165, abs=1e-4) and hi[0] - lo[0] == pytest.approx(0.020)
    assert {n for n, _, _ in f.ports} == {"start", "end"}
    blind = build("flange", size="DN50", blind=True)
    assert blind.mesh.volume > f.mesh.volume


def test_pipe_by_nominal_size():
    p = build("pipe", length=1, size="DN50")
    assert np.ptp(p.mesh.bounds[:, 1]) == pytest.approx(0.0603, abs=1e-4)
    with pytest.raises(ToolArgumentError, match="pipe needs 'radius'"):
        build("pipe", length=1)


@pytest.mark.parametrize("word,shape,implied", [("screw", "bolt", {}), ("cap_screw", "bolt", {"head": "socket"}),
                                                ("cog", "gear", {}), ("blind_flange", "flange", {"blind": True}),
                                                ("hex_nut", "nut", {})])
def test_everyday_words_for_parts(word, shape, implied):
    obj = {"shape": word, "params": {"size": "M8"}}
    intent.normalize_shape(obj)
    assert obj["shape"] == shape and obj["params"] == {**implied, "size": "M8"}


# ---------------------------------------------------------------------------
# holes for bolts
# ---------------------------------------------------------------------------

def test_hole_named_by_its_bolt():
    assert cuts.fastener_hole({"bolt": "M10", "position": [0.1, 0, 0]}) == {
        "position": [0.1, 0, 0], "shape": "cylinder", "params": {"radius": 0.0055}}
    assert cuts.fastener_hole({"bolt": "M10", "tapped": True})["params"] == {"radius": 0.00425}
    assert cuts.fastener_hole({"shape": "cylinder", "params": {"size": "M6"}})["params"] == {"radius": 0.0033}
    sunk = cuts.fastener_hole({"bolt": "M8", "countersunk": True})
    assert sunk["shape"] == "lathe" and sunk["side"] == "top"
    plain = {"shape": "cylinder", "params": {"radius": 0.01}}
    assert cuts.fastener_hole(plain) is plain


PLATE = {"id": "plate", "shape": "box", "params": {"width": 0.1, "height": 0.01, "depth": 0.06}}


def _bolt(**kw):
    return {"id": "b", "shape": "bolt", "params": {"size": "M10", "length": 0.03, **kw},
            "placement": {"relative_to": "plate", "side": "top"}}


def test_bolt_on_a_part_rests_on_its_head_and_asks_for_the_hole():
    s = compile_scene([PLATE, _bolt()])
    b = s.objects["b"]
    assert b.bbox[0][1] == pytest.approx(0.01 - 0.03)  # shank down through the plate
    (w,) = s.warnings
    assert w == ("'b' has no hole: it goes into solid 'plate'. Add the hole to 'plate' (keep its other cuts): "
                 "\"cuts\": [{\"bolt\": \"M10\", \"position\": [0.0, 0, 0.0]}].")
    fixed = compile_scene([{**PLATE, "cuts": [{"bolt": "M10", "position": [0.0, 0, 0.0]}]}, _bolt()])
    assert fixed.warnings == []


def test_hole_for_a_pattern_of_bolts_keeps_the_pattern():
    bolts = {**_bolt(), "params": {"size": "M6", "length": 0.025},
             "pattern": {"type": "grid", "count": 2, "rows": 2, "spacing": [0.08, 0.04]}}
    s = compile_scene([PLATE, bolts])
    (w,) = s.warnings
    cut = json.loads(w[w.index("[{"):w.rindex("}]") + 2])[0]
    assert cut["pattern"] == bolts["pattern"] and cut["bolt"] == "M6"
    assert compile_scene([{**PLATE, "cuts": [cut]}, bolts]).warnings == []


def test_countersunk_bolt_sits_flush_in_its_countersink():
    s = compile_scene([{**PLATE, "cuts": [{"bolt": "M8", "countersunk": True}]},
                       {**_bolt(head="countersunk"), "params": {"size": "M8", "length": 0.03, "head": "countersunk"}}])
    assert s.objects["b"].bbox[1][1] == pytest.approx(0.01)
    assert s.warnings == []
    head = world(s.objects["b"])
    plate = world(s.objects["plate"])
    assert overlap(head, plate) < 1e-9


def test_nut_threads_on_under_what_the_bolt_holds():
    holed = {**PLATE, "position": [0, 0.05, 0], "cuts": [{"bolt": "M10"}]}
    s = compile_scene([holed,
                       {"id": "w", "shape": "washer", "params": {"size": "M10"},
                        "placement": {"relative_to": "plate", "side": "top"}},
                       {"id": "b", "shape": "bolt", "params": {"size": "M10", "length": 0.03},
                        "placement": {"relative_to": "w", "side": "top"}},
                       {"id": "n", "shape": "nut", "params": {"size": "M10"},
                        "placement": {"relative_to": "b", "side": "bottom"}}])
    assert s.objects["n"].bbox[1][1] == pytest.approx(0.05)  # tight under the plate
    assert s.objects["b"].bbox[0][1] == pytest.approx(0.062 - 0.03)
    assert s.warnings == []


def test_too_short_bolt_names_the_length_that_fits():
    holed = {**PLATE, "position": [0, 0.05, 0], "cuts": [{"bolt": "M10"}]}
    objs = [holed, {"id": "b", "shape": "bolt", "params": {"size": "M10", "length": 0.02},
                    "placement": {"relative_to": "plate", "side": "top"}},
            {"id": "n", "shape": "nut", "params": {"size": "M10"}, "placement": {"relative_to": "b", "side": "bottom"}}]
    (w,) = compile_scene(objs).warnings
    assert w == ("'n' leaves only 1.6 mm of 'b' sticking out (at least 2 threads, 3 mm, should): the bolt is too short "
                 "for what it holds. Use \"length\": 0.025 on 'b'.")
    objs[1]["params"]["length"] = 0.025
    assert compile_scene(objs).warnings == []


def test_bolts_through_a_plate_on_the_floor_are_not_below_ground_warnings():
    holed = {**PLATE, "cuts": [{"bolt": "M10"}]}
    s = compile_scene([holed, _bolt(), {"id": "n", "shape": "nut", "params": {"size": "M10"},
                                        "placement": {"relative_to": "b", "side": "bottom"}}])
    assert s.warnings == []


# ---------------------------------------------------------------------------
# gears
# ---------------------------------------------------------------------------

def _meshes(s, a, b):
    """Zero overlap where they are, and a clash once `b` turns half a tooth: they really interlock."""
    A, B = world(s.objects[a]), world(s.objects[b])
    co = s.objects[b]
    centre = (co.instances[0] @ np.append(co.built.port("axis")[0], 1))[:3]
    axis = co.instances[0][:3, :3] @ co.built.port("axis")[1]
    turned = B.copy()
    turned.apply_transform(tf.rotation_matrix(math.pi / co.spec["params"]["teeth"], axis, centre))
    return overlap(A, B) < 1e-12 and overlap(A, turned) > 1e-8


def test_gear_placed_beside_a_gear_meshes_with_it():
    s = compile_scene([{"id": "g1", "shape": "gear", "params": {"teeth": 20, "module": 2}},
                       {"id": "g2", "shape": "gear", "params": {"teeth": 13, "module": 2},
                        "placement": {"relative_to": "g1", "side": "right"}}])
    c1, c2 = (s.objects[g].instances[0] @ np.append(s.objects[g].built.port("axis")[0], 1) for g in ("g1", "g2"))
    assert np.linalg.norm(c2 - c1) == pytest.approx(0.033)  # (20 + 13) x 2 mm / 2
    assert _meshes(s, "g1", "g2")


def test_gear_train_with_a_turned_first_gear():
    s = compile_scene([{"id": "g1", "shape": "gear", "params": {"teeth": 17, "module": 2}, "rotation": [0, 0, 7],
                        "position": [0, 0.1, 0]},
                       {"id": "g2", "shape": "gear", "params": {"teeth": 30, "module": 2},
                        "placement": {"relative_to": "g1", "side": "top"}},
                       {"id": "g3", "shape": "gear", "params": {"teeth": 18, "module": 2},
                        "placement": {"relative_to": "g2", "side": "left"}}])
    assert _meshes(s, "g1", "g2") and _meshes(s, "g2", "g3")


def test_flat_and_helical_gears_mesh():
    flat = compile_scene([{"id": "g1", "shape": "gear", "params": {"teeth": 24, "module": 1.5, "axis": "y"}},
                          {"id": "g2", "shape": "gear", "params": {"teeth": 19, "module": 1.5, "axis": "y"},
                           "placement": {"relative_to": "g1", "side": "front"}}])
    assert _meshes(flat, "g1", "g2") and flat.warnings == []
    helical = compile_scene([{"id": "g1", "shape": "gear", "params": {"teeth": 20, "module": 2, "helix_angle": 15},
                              "position": [0, 0.01, 0]},
                             {"id": "g2", "shape": "gear", "params": {"teeth": 20, "module": 2, "helix_angle": -15},
                              "placement": {"relative_to": "g1", "side": "right"}}])
    assert _meshes(helical, "g1", "g2")


def test_gears_that_cannot_mesh_say_how_to_fix_it():
    s = compile_scene([{"id": "g1", "shape": "gear", "params": {"teeth": 12, "module": 2}},
                       {"id": "g2", "shape": "gear", "params": {"teeth": 40, "module": 2.5},
                        "placement": {"relative_to": "g1", "side": "right"}}])
    assert ("'g2' goes 0.038 m below the ground: it meshes with the gears before it at their axis height. Lift the "
            "first gear of the train: \"position\": [0, 0.0384, 0] on 'g1'.") in s.warnings
    assert ("'g2' (module 2.5 mm) can't mesh with 'g1' (module 2 mm): meshing gears need the same module. "
            "Use \"module\": 2 on 'g2'.") in s.warnings
    lifted = compile_scene([{"id": "g1", "shape": "gear", "params": {"teeth": 12, "module": 2}, "position": [0, 0.028, 0]},
                            {"id": "g2", "shape": "gear", "params": {"teeth": 40, "module": 2},
                             "placement": {"relative_to": "g1", "side": "right"}}])
    assert lifted.warnings == []
    same_hand = compile_scene([{"id": "g1", "shape": "gear", "params": {"teeth": 20, "module": 2, "helix_angle": 10},
                                "position": [0, 0.01, 0]},
                               {"id": "g2", "shape": "gear", "params": {"teeth": 20, "module": 2, "helix_angle": 10},
                                "placement": {"relative_to": "g1", "side": "right"}}])
    assert same_hand.warnings == ["'g2' and 'g1' are helical gears on parallel axes: their helix angles must be "
                                  "opposite. Use \"helix_angle\": -10 on 'g2'."]


def test_gear_with_a_gap_is_placed_like_anything_else():
    s = compile_scene([{"id": "g1", "shape": "gear", "params": {"teeth": 20, "module": 2}},
                       {"id": "g2", "shape": "gear", "params": {"teeth": 20, "module": 2},
                        "placement": {"relative_to": "g1", "side": "right", "gap": 0.01}}])
    assert s.objects["g2"].bbox[0][0] == pytest.approx(s.objects["g1"].bbox[1][0] + 0.01)


# ---------------------------------------------------------------------------
# flanges on pipes
# ---------------------------------------------------------------------------

def test_flange_joined_to_a_pipe_end():
    s = compile_scene([{"id": "p", "shape": "pipe", "params": {"length": 0.4, "size": "DN50"}, "position": [0, 0.1, 0]},
                       {"id": "f", "shape": "flange", "params": {"size": "DN50"},
                        "placement": {"relative_to": "p", "side": "end"}}])
    f = s.objects["f"]
    assert f.bbox[0][0] == pytest.approx(0.2) and f.bbox[1][0] == pytest.approx(0.22)
    assert (f.bbox[0][1] + f.bbox[1][1]) / 2 == pytest.approx(0.1 + 0.0603 / 2, abs=1e-4)  # on the pipe's axis
    assert s.warnings == []


def test_unknown_size_is_an_error_the_model_can_fix():
    with pytest.raises(SceneError, match="M7 is not a standard ISO size; use M6 or M8"):
        compile_scene([{"id": "b", "shape": "bolt", "params": {"size": "M7", "length": 0.03}}])


def test_cone_with_a_top_radius_is_cut_off_like_a_pouring_cup():
    # Qwen3-8B drew "a pouring cup cone from radius 1.5 cm to 5 cm" as a spike: cones had no top radius.
    cup = compile_scene([{"id": "cup", "shape": "cone",
                          "params": {"top_radius": 0.05, "bottom_radius": 0.015, "height": 0.1}}]).objects["cup"]
    lo, hi = cup.bbox
    assert hi[0] - lo[0] == pytest.approx(0.1, abs=1e-3)  # wide at the top
    top = cup.built.mesh.vertices[np.isclose(cup.built.mesh.vertices[:, 1], hi[1])]
    assert np.hypot(top[:, 0], top[:, 2]).max() == pytest.approx(0.05, abs=1e-6)
    spike = build("cone", radius=0.015, height=0.1)
    assert np.ptp(spike.mesh.bounds[:, 0]) == pytest.approx(0.03, abs=1e-3)