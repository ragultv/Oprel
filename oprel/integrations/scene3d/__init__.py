"""
3D Scene: a built-in toolset that lets the model draw in a three.js canvas.

The model writes a recipe (shapes, sizes, placement); the engine in `engine/` computes the
exact geometry. No credentials, nothing to connect: enabled per request with
`enable_tools: ["scene3d"]`.
"""

from __future__ import annotations

from typing import Any

from oprel.integrations.base import ConnectionTestResult, Integration
from oprel.integrations.scene3d.tools import build_tools
from oprel.tools.base import Tool

# Short on purpose: it is resent every turn and local models often have a 4k context.
SYSTEM_HINT = """\
You can draw in a 3D scene next to the chat with the scene_* tools.
Ready-made objects (table, chair, stool, bench, bookshelf, bed, sofa, lamp) are complete and well proportioned: \
ALWAYS use them for those things, one object each, default sizes unless asked (a table's legs and round/rectangle \
top are params). Their front faces +Z. Build from primitives only what is not ready-made.
Coordinates: metres; Y is up and the ground is y=0; right = +X, front (toward the viewer) = +Z, back = -Z.
position = the object's bottom-centre ([x, 0, z] stands on the ground).
Seat things around another object: "pattern": {"type": "around", "relative_to": "<id>", "count": 4} \
(spaced along its sides, each turned to face it).
Prefer placement to coordinates: {"relative_to": "<id>", "side": "top|bottom|left|right|front|back|center", \
"align": "front-right", "gap": 0.1}; the engine computes the exact spot.
rotation = degrees [x, y, z] about the object's centre; it stays resting where it was.
Custom multi-part objects: a "group", then parts with "parent": "<group id>" (positions relative to the group). \
Never list identical parts one by one; use one object with a pattern: \
{"type": "grid", "count": 2, "rows": 2, "spacing": [sx, sz]} | {"type": "radial", "count": 8, "radius": r} | \
{"type": "linear", "count": n, "step": [x, y, z]}.
Draw everything in ONE scene_add_objects call with a short plan. The result gives exact bounds and warnings. \
Warnings are for you: fix them with scene_update_objects before replying; never ask the user to run a tool. \
Use short ids. Reply in one or two sentences.
Example, "a dining table with 6 chairs and a lamp beside it":
{"plan": "table + 6 chairs around it + floor lamp", "objects": [\
{"id": "table", "shape": "table", "params": {"width": 1.8, "depth": 0.9}}, \
{"id": "chairs", "shape": "chair", "pattern": {"type": "around", "relative_to": "table", "count": 6}}, \
{"id": "lamp", "shape": "lamp", "placement": {"relative_to": "table", "side": "right", "gap": 0.8}}]}
Example, a custom object, "a rocket":
{"plan": "body, nose cone on top, 3 fins", "objects": [{"id": "rocket", "shape": "group"}, \
{"id": "body", "parent": "rocket", "shape": "cylinder", "params": {"radius": 0.3, "height": 2}}, \
{"id": "nose", "parent": "rocket", "shape": "cone", "params": {"radius": 0.3, "height": 0.6}, \
"placement": {"relative_to": "body", "side": "top"}}, \
{"id": "fins", "parent": "rocket", "shape": "box", "params": {"width": 0.05, "height": 0.5, "depth": 0.4}, \
"pattern": {"type": "radial", "count": 3, "radius": 0.3, "face_center": true}}]}"""


class Scene3DIntegration(Integration):
    id = "scene3d"
    name = "3D Scene"
    description = "Let the model draw 3D shapes and objects in a canvas next to the chat."
    icon = "box"
    system_hint = SYSTEM_HINT
    requires_connection = False
    max_tool_iterations = 16
    # At 0.7, Qwen3-8B drifted from the worked example (legs listed one by one, y/z mixed up).
    max_temperature = 0.3

    def __init__(self) -> None:
        self._tools = build_tools()

    async def test_connection(self, config: dict[str, Any], secrets: dict[str, str]) -> ConnectionTestResult:
        return ConnectionTestResult(ok=True, message="Built in: always available.")

    def tools(self) -> list[Tool]:
        return self._tools


integration = Scene3DIntegration()
