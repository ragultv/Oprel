"""
Geometry engine for the 3D scene toolset.

The model writes a recipe (shapes + parameters + placement); this package turns it into
exact meshes: validate -> build -> anchor -> pattern -> place -> report -> export.
"""

from oprel.integrations.scene3d.engine.compiler import CompiledScene, SceneError, compile_scene
from oprel.integrations.scene3d.engine.export import export_glb, export_stl

__all__ = ["CompiledScene", "SceneError", "compile_scene", "export_glb", "export_stl"]
