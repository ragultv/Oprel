"""
Minimal JSON-Schema validation for tool arguments.

Supports the subset tools actually use: object/properties/required/additionalProperties,
string/integer/number/boolean/array, enum, minLength/maxLength, minimum/maximum, maxItems.

Small local models are sloppy with types ("42" for 42, "true" for true, a bare string
where a list is expected), so scalar values are coerced when the intent is unambiguous.
"""

from __future__ import annotations

from typing import Any


class ToolArgumentError(ValueError):
    pass


def _coerce_scalar(value: Any, typ: str, path: str) -> Any:
    if typ == "string":
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
        raise ToolArgumentError(f"'{path}' must be a string")

    if typ == "integer":
        if isinstance(value, bool):
            raise ToolArgumentError(f"'{path}' must be an integer")
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            text = value.strip().lstrip("#")
            if text.lstrip("-").isdigit():
                return int(text)
        raise ToolArgumentError(f"'{path}' must be an integer")

    if typ == "number":
        if isinstance(value, bool):
            raise ToolArgumentError(f"'{path}' must be a number")
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                pass
        raise ToolArgumentError(f"'{path}' must be a number")

    if typ == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in ("true", "false"):
            return value.strip().lower() == "true"
        raise ToolArgumentError(f"'{path}' must be a boolean")

    return value


def _validate(value: Any, schema: dict[str, Any], path: str) -> Any:
    typ = schema.get("type")

    if typ == "object":
        if not isinstance(value, dict):
            raise ToolArgumentError(f"'{path or 'arguments'}' must be an object")
        props: dict[str, Any] = schema.get("properties", {})
        out: dict[str, Any] = {}
        for key in schema.get("required", []):
            if key not in value or value[key] is None or value[key] == "":
                raise ToolArgumentError(f"Missing required argument '{_join(path, key)}'")
        for key, item in value.items():
            if item is None:
                continue  # models often send explicit nulls for optional args
            if key in props:
                out[key] = _validate(item, props[key], _join(path, key))
            elif schema.get("additionalProperties", False) is False:
                continue  # drop unknown keys rather than fail; the model may hallucinate extras
            else:
                out[key] = item
        return out

    if typ == "array":
        if isinstance(value, str):
            value = [v.strip() for v in value.split(",") if v.strip()]
        if not isinstance(value, list):
            raise ToolArgumentError(f"'{path}' must be an array")
        max_items = schema.get("maxItems")
        if max_items is not None and len(value) > max_items:
            raise ToolArgumentError(f"'{path}' may contain at most {max_items} items")
        item_schema = schema.get("items", {})
        return [_validate(v, item_schema, f"{path}[{i}]") for i, v in enumerate(value)]

    value = _coerce_scalar(value, typ, path) if typ else value

    if "enum" in schema:
        allowed = schema["enum"]
        if value not in allowed:
            # Case-insensitive match for strings ("task" -> "Task")
            if isinstance(value, str):
                for option in allowed:
                    if isinstance(option, str) and option.lower() == value.lower():
                        return option
            raise ToolArgumentError(f"'{path}' must be one of: {', '.join(map(str, allowed))}")

    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            raise ToolArgumentError(f"'{path}' must be at least {schema['minLength']} characters")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise ToolArgumentError(f"'{path}' must be at most {schema['maxLength']} characters")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise ToolArgumentError(f"'{path}' must be >= {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            raise ToolArgumentError(f"'{path}' must be <= {schema['maximum']}")

    return value


def _join(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def validate_arguments(schema: dict[str, Any], arguments: Any) -> dict[str, Any]:
    """Validate and coerce `arguments` against an object schema. Raises ToolArgumentError."""
    if arguments is None:
        arguments = {}
    return _validate(arguments, schema, "")
