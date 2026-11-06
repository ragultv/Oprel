"""
Shape catalogue. Each shape is a builder function plus the schema of its `params`.

Adding a shape = one decorated function in engine/shapes/. The tool descriptions and
the system hint are generated from this registry, so the model always sees the real list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import trimesh

from oprel.tools.validation import ToolArgumentError, validate_arguments

# Sane bounds for any length, in metres: 0.1 mm .. 1 km
MIN_LEN = 0.0001
MAX_LEN = 1000.0


def length(description: str, default: float | None = None) -> dict[str, Any]:
    spec: dict[str, Any] = {"type": "number", "minimum": MIN_LEN, "maximum": MAX_LEN, "description": description}
    if default is not None:
        spec["default"] = default
    return spec


def integer(description: str, minimum: int, maximum: int, default: int | None = None) -> dict[str, Any]:
    spec: dict[str, Any] = {"type": "integer", "minimum": minimum, "maximum": maximum, "description": description}
    if default is not None:
        spec["default"] = default
    return spec


Builder = Callable[[dict[str, Any]], trimesh.Trimesh]


@dataclass(frozen=True)
class ShapeDef:
    name: str
    description: str
    params: dict[str, dict[str, Any]]
    required: tuple[str, ...]
    builder: Builder | None  # None = no mesh of its own (group)

    @property
    def schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": self.params, "required": list(self.required)}

    def signature(self) -> str:
        """Compact form for prompts: box(width, height, depth, [corner_segments])."""
        parts = [p if p in self.required else f"{p}?" for p in self.params]
        return f"{self.name}({', '.join(parts)})"

    def validate(self, params: dict[str, Any] | None) -> dict[str, Any]:
        """Check and fill defaults. Raises ToolArgumentError with a message the model can act on."""
        out = validate_arguments(self.schema, params or {})
        for key, spec in self.params.items():
            if key not in out and "default" in spec:
                out[key] = spec["default"]
        return out


_SHAPES: dict[str, ShapeDef] = {}


def shape(
    name: str,
    description: str,
    params: dict[str, dict[str, Any]] | None = None,
    required: tuple[str, ...] = (),
) -> Callable[[Builder], Builder]:
    def register(fn: Builder) -> Builder:
        if name in _SHAPES:
            raise ValueError(f"Shape '{name}' is already registered")
        _SHAPES[name] = ShapeDef(name, description, params or {}, required, fn)
        return fn

    return register


def register_meshless(name: str, description: str) -> None:
    _SHAPES[name] = ShapeDef(name, description, {}, (), None)


def _ensure_loaded() -> None:
    from oprel.integrations.scene3d.engine import shapes  # noqa: F401  (registers on import)


def get_shape(name: str) -> ShapeDef | None:
    _ensure_loaded()
    return _SHAPES.get(name)


def all_shapes() -> list[ShapeDef]:
    _ensure_loaded()
    return list(_SHAPES.values())


def shape_names() -> list[str]:
    return [s.name for s in all_shapes()]


__all__ = [
    "ShapeDef", "ToolArgumentError", "all_shapes", "get_shape", "integer", "length",
    "register_meshless", "shape", "shape_names",
]
