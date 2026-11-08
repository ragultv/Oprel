"""Merged groups: parts fused into one watertight solid, for flow simulation and 3D printing."""

import io

import pytest
import trimesh

from oprel.integrations.scene3d.engine import SceneError, compile_scene, export_glb, export_stl

PART = [
    {"id": "part", "shape": "group", "merge": True},
    {"id": "big", "parent": "part", "shape": "cylinder", "params": {"radius": 0.2, "height": 1}},
    {"id": "thin", "parent": "part", "shape": "cylinder", "params": {"radius": 0.07, "height": 1.1},
     "placement": {"relative_to": "big", "side": "left", "gap": 0.15}},
    {"id": "plate", "parent": "part", "shape": "box", "params": {"height": 0.2, "depth": 0.02},
     "placement": {"between": ["thin", "big"]}},
]


def _without_merge():
    return [{k: v for k, v in o.items() if k != "merge"} for o in PART]


def test_merged_group_is_one_watertight_solid_with_nothing_buried_inside():
    scene = compile_scene(PART)
    solid = scene.merged["part"]
    assert scene.warnings == [] and solid.bodies == 1 and solid.mesh.is_watertight
    assert set(solid.parts) == {"big", "thin", "plate"}
    # The union counts the overlap once: less than the three volumes added up.
    separate = sum(scene.objects[p].built.mesh.volume for p in solid.parts)
    assert solid.mesh.volume < separate - 1e-4


def test_merged_group_is_one_node_in_the_glb():
    glb = trimesh.load(io.BytesIO(export_glb(compile_scene(PART))), file_type="glb")
    assert sorted(glb.graph.nodes_geometry) == ["part"]
    glb = trimesh.load(io.BytesIO(export_glb(compile_scene(_without_merge()))), file_type="glb")
    assert sorted(glb.graph.nodes_geometry) == ["big", "plate", "thin"]


def test_stl_fuses_the_whole_scene_even_without_a_merged_group():
    data, bodies, watertight = export_stl(compile_scene(_without_merge()))
    mesh = trimesh.load(io.BytesIO(data), file_type="stl")
    assert bodies == 1 and watertight and mesh.is_watertight
    assert mesh.bounds.round(3).tolist() == [[-0.49, 0, -0.2], [0.2, 1.1, 0.2]]


def test_parts_that_do_not_touch_are_named():
    objects = [dict(o) for o in PART]
    objects[3] = {"id": "plate", "parent": "part", "shape": "box", "params": {"width": 0.05, "height": 0.2, "depth": 0.02},
                  "position": [-0.27, 0, 0]}
    scene = compile_scene(objects)
    assert scene.merged["part"].bodies == 3
    assert any("comes out as 3 separate bodies" in w and "'plate'" in w for w in scene.warnings)


def test_merge_on_something_that_is_not_a_group_is_an_error():
    with pytest.raises(SceneError, match="merge is for groups"):
        compile_scene([{"id": "big", "shape": "cylinder", "params": {"radius": 0.2, "height": 1}, "merge": True}])


def test_nested_merged_group_is_part_of_the_outer_solid():
    objects = [{"id": "outer", "shape": "group", "merge": True},
               {"id": "inner", "shape": "group", "parent": "outer", "merge": True},
               {"id": "a", "parent": "inner", "shape": "box", "params": {"width": 1, "height": 1, "depth": 1}},
               {"id": "b", "parent": "outer", "shape": "box", "params": {"width": 1, "height": 1, "depth": 1},
                "position": [0.5, 0, 0]}]
    scene = compile_scene(objects)
    assert list(scene.merged) == ["outer"] and scene.merged["outer"].bodies == 1
    assert scene.merged["outer"].mesh.volume == pytest.approx(1.5)
