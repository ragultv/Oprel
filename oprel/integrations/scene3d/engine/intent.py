"""
Reading what the model meant, by the kind of thing it said rather than by shape.

Small models describe objects in everyday words ("counter", "mug", "jug") where the engine has
shape names (box, cup with a handle, vase with one handle). The table below maps those words;
adding one is a line of data. Anything not in it gets the closest shape names in its error.

Kept separate from the compiler so the tools (which save the recipe) and the compiler (which
builds it) read the words the same way.
"""

from __future__ import annotations

import difflib
import re
from typing import Any

from oprel.integrations.scene3d.engine.registry import get_shape, shape_names

# everyday word -> (shape, params it implies; the model's own params win)
SHAPE_WORDS: dict[str, tuple[str, dict[str, Any]]] = {
    # slabs and blocks
    "counter": ("box", {}), "countertop": ("box", {}), "worktop": ("box", {}), "slab": ("box", {}),
    "block": ("box", {}), "cube": ("box", {}), "brick": ("box", {}), "cuboid": ("box", {}), "panel": ("box", {}),
    "board": ("box", {}), "wall": ("box", {}), "plank": ("box", {}), "desk": ("table", {}),
    "cabinet": ("box", {}), "drawer": ("box", {}), "crate": ("box", {}), "suitcase": ("box", {}),
    "plate": ("rectangle", {}), "sheet": ("rectangle", {}), "disc": ("circle", {}), "disk": ("circle", {}),
    # round solids
    "rod": ("cylinder", {}), "bar": ("cylinder", {}), "shaft": ("cylinder", {}), "pillar": ("cylinder", {}),
    "column": ("cylinder", {}), "post": ("cylinder", {}), "pole": ("cylinder", {}), "peg": ("cylinder", {}),
    "pin": ("cylinder", {}), "can": ("cylinder", {}), "drum": ("cylinder", {}), "tank": ("cylinder", {}),
    "ball": ("sphere", {}), "globe": ("sphere", {}), "orb": ("sphere", {}), "dome": ("sphere", {}),
    "ring": ("torus", {}), "donut": ("torus", {}), "doughnut": ("torus", {}), "pill": ("capsule", {}),
    "funnel": ("cone", {}), "spike": ("cone", {}),
    # vessels
    "mug": ("cup", {"handle": True}), "teacup": ("cup", {"handle": True}), "tumbler": ("cup", {}),
    "glass": ("cup", {}), "beaker": ("cup", {}),
    "jug": ("vase", {"handles": 1}), "pitcher": ("vase", {"handles": 1}), "ewer": ("vase", {"handles": 1}),
    "amphora": ("vase", {"handles": 2}), "urn": ("vase", {"handles": 2}), "jar": ("vase", {}),
    "pot": ("plant_pot", {"handles": 2}), "cooking_pot": ("plant_pot", {"handles": 2}),
    "saucepan": ("plant_pot", {"handles": 1}), "planter": ("plant_pot", {}), "flowerpot": ("plant_pot", {}),
    "basin": ("bowl", {}), "dish": ("bowl", {}),
    # piping and structure
    "tube": ("pipe", {}), "hose": ("pipe", {}), "duct": ("pipe", {}), "bend": ("elbow", {}),
    "t_piece": ("tee", {}), "tee_piece": ("tee", {}), "girder": ("beam", {}), "joist": ("beam", {}),
    "i_beam": ("beam", {"section": "i"}), "channel": ("beam", {"section": "channel"}),
    "angle_iron": ("beam", {"section": "angle"}), "coil": ("spring", {}), "knob": ("sphere", {}),
    "grip": ("handle", {}), "pull": ("handle", {"orientation": "horizontal"}),
    # standard parts
    "screw": ("bolt", {}), "hex_bolt": ("bolt", {}), "machine_screw": ("bolt", {}), "set_screw": ("bolt", {}),
    "cap_screw": ("bolt", {"head": "socket"}), "socket_screw": ("bolt", {"head": "socket"}),
    "allen_bolt": ("bolt", {"head": "socket"}), "countersunk_screw": ("bolt", {"head": "countersunk"}),
    "fastener": ("bolt", {}), "hex_nut": ("nut", {}), "locknut": ("nut", {}), "lock_nut": ("nut", {}),
    "cog": ("gear", {}), "cogwheel": ("gear", {}), "gearwheel": ("gear", {}), "spur_gear": ("gear", {}),
    "pinion": ("gear", {}), "helical_gear": ("gear", {"helix_angle": 20}),
    "pipe_flange": ("flange", {}), "blind_flange": ("flange", {"blind": True}),
    # furniture
    "seat": ("chair", {}), "armchair": ("chair", {"arms": True}), "couch": ("sofa", {}), "shelf": ("bookshelf", {}),
    "floor_lamp": ("lamp", {}), "flower_vase": ("vase", {}),
}


def _key(word: str) -> str:
    return re.sub(r"[\s\-]+", "_", word.strip().lower())


def normalize_shape(obj: dict[str, Any]) -> str | None:
    """
    Turn an everyday shape word into a shape (in place). Returns a note for the model, or None when
    the shape was already one (or is unknown: the compiler's error then suggests the closest ones).
    """
    raw = obj.get("shape")
    if not isinstance(raw, str) or get_shape(raw) is not None:
        return None
    key = _key(raw)
    if get_shape(key) is not None:
        obj["shape"] = key
        return None
    word = SHAPE_WORDS.get(key) or SHAPE_WORDS.get(key.rstrip("s"))
    if word is None:
        # "sink_hole", "kitchen_counter", "coffee_mug": the last word that is known names it.
        for part in reversed(key.split("_")):
            word = SHAPE_WORDS.get(part) or (((part, {}) if get_shape(part) else None))
            if word:
                break
    if word is None:
        return None
    shape, implied = word
    obj["shape"] = shape
    params = obj.get("params") if isinstance(obj.get("params"), dict) else {}
    obj["params"] = {**implied, **params}
    extra = f" {implied}" if implied else ""
    return f"Read shape '{raw}' as {shape}{extra}."


_REMOVAL_WORDS = {"cut", "cutout", "cut_out", "hole", "subtract", "subtraction", "difference", "remove", "removal",
                  "boolean", "negative", "void", "opening", "bore", "drill", "drilling"}


def removal_to_hole(obj: dict[str, Any]) -> str | None:
    """
    A removal written as an object: {"shape": "cut", "params": {"shape": "cylinder", "params":
    {"radius": 0.2}}, "placement": {"relative_to": "counter", "side": "top"}} (seen with Qwen3-8B,
    "a round sink hole cut into the top"). Make it the inner shape, marked as a hole; the tools
    then turn it into a cut of the object it names. Inner shape: the one given, or a cylinder if
    it has a radius, else a box.
    """
    raw = obj.get("shape")
    if not isinstance(raw, str) or _key(raw) not in _REMOVAL_WORDS:
        return None
    params = obj.get("params") if isinstance(obj.get("params"), dict) else {}
    inner = params.get("shape") or obj.get("cut_shape") or obj.get("tool")
    inner_params = params.get("params") if isinstance(params.get("params"), dict) else {
        k: v for k, v in params.items() if k != "shape"}
    for key in [k for k in inner_params if k.endswith("diameter")]:
        inner_params[key[: -len("diameter")] + "radius"] = inner_params.pop(key) / 2
    if not isinstance(inner, str) or get_shape(_key(inner)) is None:
        inner = "cylinder" if "radius" in inner_params else "box"
    obj["shape"], obj["params"], obj["_hole"] = _key(inner), inner_params, True
    return f"Read '{raw}' as a {obj['shape']} hole."


def closest_shapes(word: str, n: int = 4) -> list[str]:
    return difflib.get_close_matches(_key(word), shape_names(), n=n, cutoff=0.4)
