"""
JSON schema of one object in a scene recipe. Shape-specific `params` are checked per shape.

Kept terse on purpose: these schemas are sent to the model on every turn, and local models
often run with a 4k context. The system hint explains placement, patterns and conventions once.
"""

from __future__ import annotations

from typing import Any

from oprel.integrations.scene3d.engine import patterns
from oprel.integrations.scene3d.engine.registry import shape_names

SIDES = ["top", "bottom", "left", "right", "front", "back", "center"]
VERTICAL = ["bottom", "center", "top"]
ALIGN_WORDS = {"left", "right", "front", "back", "center"}

_VEC3 = {"type": "array", "items": {"type": "number"}, "maxItems": 3}

PLACEMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "relative_to": {"type": "string"},
        "side": {"type": "string", "enum": SIDES},
        "align": {"type": "string"},
        "vertical": {"type": "string", "enum": VERTICAL},
        "gap": {"type": "number", "minimum": 0, "maximum": 1000},
        "offset": _VEC3,
    },
    "required": ["relative_to"],
}


def object_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "id": {"type": "string", "maxLength": 64},
            "shape": {"type": "string", "enum": shape_names()},
            "params": {"type": "object", "additionalProperties": True},
            "position": _VEC3,
            "rotation": _VEC3,
            "placement": PLACEMENT_SCHEMA,
            "parent": {"type": "string"},
            "pattern": patterns.SCHEMA,
            "color": {"type": "string"},
            "opacity": {"type": "number", "minimum": 0.05, "maximum": 1},
            "label": {"type": "string", "maxLength": 80},
        },
        "required": ["shape"],
    }


def update_schema(removable: list[str]) -> dict[str, Any]:
    """
    One change to an existing object. Loose on purpose (nested objects pass through): the
    engine re-checks the whole merged object, so this only needs to steer the model.
    """
    loose = {"type": "object", "additionalProperties": True}
    return {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "shape": {"type": "string", "enum": shape_names()},
            "params": loose,
            "position": _VEC3,
            "rotation": _VEC3,
            "placement": loose,
            "pattern": loose,
            "parent": {"type": "string"},
            "color": {"type": "string"},
            "opacity": {"type": "number", "minimum": 0.05, "maximum": 1},
            "label": {"type": "string", "maxLength": 80},
            "remove": {"type": "array", "items": {"type": "string", "enum": removable}},
        },
        "required": ["id"],
    }
