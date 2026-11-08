"""
JSON schema of one object in a scene recipe. Shape-specific `params` are checked per shape.

Kept terse on purpose: these schemas are sent to the model on every turn, and local models
often run with a 4k context. The system hint explains placement, patterns and conventions once.
"""

from __future__ import annotations

from typing import Any

from oprel.integrations.scene3d.engine import cuts, patterns
from oprel.integrations.scene3d.engine.registry import shape_names

SIDES = ["top", "bottom", "left", "right", "front", "back", "center"]
VERTICAL = ["bottom", "center", "top"]
ALIGN_WORDS = {"left", "right", "front", "back", "center"}

_VEC3 = {"type": "array", "items": {"type": "number"}, "maxItems": 3}

PLACEMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "relative_to": {"type": "string"},
        "between": {"type": "array", "items": {"type": "string"}, "maxItems": 2},
        "side": {"type": "string", "enum": SIDES},
        "align": {"type": "string"},
        "vertical": {"type": "string", "enum": VERTICAL},
        "gap": {"type": "number", "minimum": 0, "maximum": 1000},
        "offset": _VEC3,
    },
}


_LIMITS = {"minimum", "maximum", "maxLength", "maxItems"}


def lean(schema: Any) -> Any:
    """
    The schema without its numeric limits, for sending to the model: the engine checks the
    full schema anyway (compiler._check), and the limits cost ~150 tokens on every turn.
    """
    if isinstance(schema, dict):
        return {k: lean(v) for k, v in schema.items() if k not in _LIMITS}
    if isinstance(schema, list):
        return [lean(v) for v in schema]
    return schema


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
            "cuts": cuts.SCHEMA,
            "merge": {"type": "boolean"},
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
            "shape": {"type": "string"},  # the add tool lists the names; the engine checks them
            "params": loose,
            "position": _VEC3,
            "rotation": _VEC3,
            "placement": loose,
            "pattern": loose,
            "cuts": {"type": "array", "items": loose},
            "merge": {"type": "boolean"},
            "parent": {"type": "string"},
            "color": {"type": "string"},
            "opacity": {"type": "number", "minimum": 0.05, "maximum": 1},
            "label": {"type": "string", "maxLength": 80},
            "remove": {"type": "array", "items": {"type": "string", "enum": removable}},
        },
        "required": ["id"],
    }
