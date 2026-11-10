"""
3D Scene: a built-in toolset that lets the model draw in a three.js canvas.

The model writes a recipe (shapes, sizes, placement); the engine in `engine/` computes the
exact geometry. No credentials, nothing to connect: enabled per request with
`enable_tools: ["scene3d"]`.
"""

from __future__ import annotations

from typing import Any

from oprel.integrations.base import ConnectionTestResult, Integration
from oprel.integrations.scene3d.tools import build_tools, scene_summary
from oprel.tools.base import Tool

# Short on purpose: it is resent every turn and local models often have a 4k context.
SYSTEM_HINT = """\
You can draw in a 3D scene next to the chat with the scene_* tools.
Ready-made shapes (table ... lamp, vessels, pipe ... handle in the shape list) are complete and well \
proportioned: ALWAYS use them for those things, one object each, default sizes unless asked. Furniture faces +Z. \
Build from primitives only what is not ready-made. Other round things: "lathe" with a [radius, height] profile \
from the bottom up. Flat outlines (bracket, letter): "extrude". Along a path (rail, bent rod, hook): "sweep". No ready-made shape for it? Build it from lathe / extrude / sweep, "cuts" and a merged group: a screw thread is a sweep of a triangle outline along {"helix": {...}} merged with a cylinder (or cut into one).
radius = half the diameter. A pipe run: list the parts in order; each is joined to the end of the one before. \
To turn an elbow or tee: "placement": {"relative_to": "<part before>", "side": "end", "toward": "up|front|..."} \
on it (toward = where its outlet or branch heads).
Holes, slots, windows, doors: "cuts" on that object, never extra objects. A plate with a hole in the middle and \
4 corner holes: "cuts": [{"shape": "cylinder", "params": {"radius": 0.03}}, {"shape": "cylinder", "params": \
{"radius": 0.004}, "pattern": {"type": "grid", "count": 2, "rows": 2, "spacing": [0.24, 0.24]}}] (a ring of holes: \
"radial"). A cut's position is its centre from the object's bottom-centre (x = z = 0 is the middle, not a corner; \
leave it out for the middle). No height (box: no size) = right through; rotation [90, 0, 0] = front to back. \
"side" only to open a cut onto a face: "bottom" for a door, "top" for a pocket. To add cuts later, update the \
object with its whole cuts list.
Coordinates: metres; Y is up and the ground is y=0; right = +X, front (toward the viewer) = +Z, back = -Z.
position = the object's bottom-centre ([x, 0, z] stands on the ground).
Seat things around another object: "pattern": {"type": "around", "relative_to": "<id>", "count": 4} \
(spaced along its sides, each turned to face it).
Prefer placement to coordinates: {"relative_to": "<id>", "side": "top|bottom|left|right|front|back|center", \
"align": "front-right", "gap": 0.1}; the engine computes the exact spot (side = where it goes, bottom = underneath; \
gap = between the surfaces; "on the left end of the table" = side "top", align "left"). A part joining two objects (plate, beam, bridge): "placement": {"between": ["a", "b"]} \
and leave out its width: the engine sizes and turns it.
rotation = degrees [x, y, z] about the object's centre; it stays resting where it was.
One solid part (for simulation, CAD export, 3D printing): the group gets "merge": true, and its parts are fused. To merge \
what is already drawn, add {"id": "part", "shape": "group", "merge": true} with no parts.
Custom multi-part objects: a "group", then parts with "parent": "<group id>" (positions relative to the group). \
Never list identical parts one by one; use one object with a pattern: \
{"type": "grid", "count": 2, "rows": 2, "spacing": [sx, sz]} | {"type": "radial", "count": 8, "radius": r} | \
{"type": "linear", "count": n, "step": [x, y, z]}.
Draw everything in ONE scene_add_objects call with a short plan. The result gives exact bounds and warnings. \
Warnings are for you: fix them with scene_update_objects before replying; never ask the user to run a tool. \
Use short ids. Reply in one or two sentences. \
User numbers without units: read them sensibly for the object's size (cm, or relative to its current size, \
e.g. "bottom 2, top 1" = bottom twice the top), never so the object becomes absurd; say how you read them.
Example, "a dining table with 6 chairs and a vase on it":
{"plan": "table + 6 chairs around it + vase on top", "objects": [\
{"id": "table", "shape": "table", "params": {"width": 1.8, "depth": 0.9}}, \
{"id": "chairs", "shape": "chair", "pattern": {"type": "around", "relative_to": "table", "count": 6}}, \
{"id": "vase", "shape": "vase", "placement": {"relative_to": "table", "side": "top"}}]}
Example, a custom object, "two posts 1.2 m apart, a beam joining their tops":
{"plan": "group; post; second post 1.2 m to its left; beam between them", "objects": [\
{"id": "gate", "shape": "group"}, \
{"id": "post1", "parent": "gate", "shape": "cylinder", "params": {"radius": 0.08, "height": 2}}, \
{"id": "post2", "parent": "gate", "shape": "cylinder", "params": {"radius": 0.08, "height": 2}, \
"placement": {"relative_to": "post1", "side": "left", "gap": 1.2}}, \
{"id": "beam", "parent": "gate", "shape": "box", "params": {"height": 0.1, "depth": 0.1}, \
"placement": {"between": ["post2", "post1"], "vertical": "top"}}]}
More space between them later = a bigger "gap" on post2; the beam follows."""


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
    # The engine checks every drawing and its warnings name the fix, so thinking adds little but
    # minutes: Qwen3-8B on a 4 GB GPU thought for 5+ minutes (and timed out) on a 3-object part.
    allow_thinking = False

    def __init__(self) -> None:
        self._tools = build_tools()

    async def test_connection(self, config: dict[str, Any], secrets: dict[str, str]) -> ConnectionTestResult:
        return ConnectionTestResult(ok=True, message="Built in: always available.")

    def tools(self) -> list[Tool]:
        return self._tools

    def context_hint(self, conversation_id: str | None) -> str:
        return scene_summary(conversation_id)


integration = Scene3DIntegration()
