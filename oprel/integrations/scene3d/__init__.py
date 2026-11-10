"""
3D Scene: a built-in toolset that lets the model draw in a three.js canvas.

The model writes a recipe (shapes, sizes, placement); the engine in `engine/` computes the
exact geometry. No credentials, nothing to connect: enabled per request with
`enable_tools: ["scene3d"]`.
"""

from __future__ import annotations

import re
from typing import Any

from oprel.integrations.base import ConnectionTestResult, Integration
from oprel.integrations.scene3d.engine.registry import shape_names
from oprel.integrations.scene3d.plan import clean_plan
from oprel.integrations.scene3d.tools import build_tools, scene_summary
from oprel.tools.base import Tool

# Short on purpose: it is resent every turn and local models often have a 4k context.
SYSTEM_HINT = """\
You can draw in a 3D scene next to the chat with the scene_* tools.
Ready-made shapes (table ... lamp, vessels, pipe ... flange in the shape list) are complete and well \
proportioned: ALWAYS use them for those things, one object each, default sizes unless asked. Furniture faces +Z. \
Standard parts take the standard size ("size": "M10" / "DN50", gear teeth + module in mm); the engine knows the \
rest. \
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
gap = between the surfaces; "on the left end of the table" = side "top", align "left"; "touching" = no gap; a \
bar running front to back with "its middle touching X" = side "right"/"left" of X, the engine centres it). A part joining two objects (plate, beam, bridge): "placement": {"between": ["a", "b"]} \
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


_EXAMPLES_START = SYSTEM_HINT.index("Example, ")

# The tool step's instructions when a plan is attached: the worked examples make room for it.
PLANNED_HINT = SYSTEM_HINT[:_EXAMPLES_START] + """\
The user message ends with a Plan made for you: draw it exactly, one object per numbered line with its id, shape \
and params, all in ONE scene_add_objects call. "where" -> placement: "on top of X" = side "top"; "at its left end" \
= align "left"; "left of / right of / in front of / behind X" = side left/right/front/back; "gap g" = gap; \
"between a and b" = between; "at the end of X" = side "end"; "on the ground at x, z" = position [x, 0, z]; \
"hole in X" = a cut in X's cuts, not an object. Then each "after:" line with its own call ("merge everything" = \
add {"id": "part", "shape": "group", "merge": true})."""

PLANNER_PROMPT = """\
You plan drawings for a 3D CAD engine; another step draws them. Do not draw or explain: write only the plan.
Shapes: {shapes}.
Params in metres (radius = half the diameter): box width (x) height (y) depth (z); cylinder radius height \
(tapered: radius_bottom radius_top); cone radius_bottom radius_top height (no radius_top = pointed; funnel, \
pouring cup "from 1.5 cm to 5 cm": radius_bottom 0.015 radius_top 0.05); sphere radius; \
other shapes keep their own params (bolt size "M10" length, gear teeth module, pipe length size "DN50").
Y is up, the ground is y=0; right = +X, front = +Z. Long = x, deep = z, tall = y; something running front to back \
has its long size as depth.
One numbered line per object, in build order, each placed against something listed before it or already drawn:
N. id | shape | params | where
id = the user's own name for it (casting, riser, sprue), never the shape name (not "box"). Each id once: one \
line holds all of an object's params; never a second line to correct it.
where is one of:
  on the ground [at x, z]
  on top of <id> [at its left|right|front|back end]
  left of | right of | in front of | behind <id> [, gap g]
  at the end of <id> (pipe parts) | between <a> and <b>
A hole or slot: "N. hole in <id> | shape | params | where in it".
Prefer these relations to coordinates. On the ground at the left / right / front / back end of X (flush, touching) \
= left of / right of / in front of / behind X. "on top of X at its left end" is only for things resting on X, \
and only where the user says so: a thing stacked "on it" / "on top" (sprue on its well, cup on the sprue) is \
plain "on top of X", centred. \
Touching = no gap. "Its middle touching Y" = beside Y, centred on it. "between a and b" only for a part that \
joins two separate objects (a beam between two posts).
Things resting on something come after it, from the bottom up.
Steps on what is drawn (merge, delete, move) go last, one per line: "after: merge everything into one solid".
Use the user's numbers converted to metres. Keep the ids of things already drawn.
Example, "a desk 1.2 m wide with a lamp on its left end and a chair in front":
1. desk | table | width 1.2 | on the ground
2. lamp | lamp | | on top of desk at its left end
3. chair | chair | | in front of desk"""

# "a sprue well", "an ingate", "another box", "two posts": each a new thing in the request
_A_THING = re.compile(r"\b(a|an|another|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?!few\b|bit\b|lot\b)",
                      re.IGNORECASE)
_AFTER_WORDS = re.compile(r"\b(one solid|single solid|merge|fuse|combine|weld)\b", re.IGNORECASE)
_DRAW_WORDS = re.compile(r"\b(draw|create|add|make|build|design|model|place|put|attach|join|connect)\b", re.IGNORECASE)


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
    planned_hint = PLANNED_HINT

    def __init__(self) -> None:
        self._tools = build_tools()
        self.planner_prompt = PLANNER_PROMPT.format(shapes=", ".join(shape_names()))

    def wants_plan(self, text: str) -> bool:
        """
        Three or more new things, or new things plus a step on what is drawn ("... and make everything
        one solid"). Live with Qwen3-8B, two-part requests came out better in one step: the planner
        renamed the casting and put the runner "between" the casting and the ingate.
        """
        if not _DRAW_WORDS.search(text):
            return False
        things = len(_A_THING.findall(text))
        return things >= 3 or (things >= 2 and bool(_AFTER_WORDS.search(text)))

    def clean_plan(self, plan: str) -> str:
        return clean_plan(plan)

    def plan_request(self, text: str, plan: str) -> str:
        return f"{text}\n\nPlan (draw exactly this):\n{plan}"

    async def test_connection(self, config: dict[str, Any], secrets: dict[str, str]) -> ConnectionTestResult:
        return ConnectionTestResult(ok=True, message="Built in: always available.")

    def tools(self) -> list[Tool]:
        return self._tools

    def context_hint(self, conversation_id: str | None) -> str:
        return scene_summary(conversation_id)


integration = Scene3DIntegration()
