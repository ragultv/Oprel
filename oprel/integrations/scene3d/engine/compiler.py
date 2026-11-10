"""
Turns a scene recipe (list of object specs) into exact geometry.

For every object:
  1. check     shared schema, the shape's own params, references (ids, parent, relative_to)
  2. build     the shape's mesh (cached by recipe), moved so its bottom-centre is the origin
  3. pattern   the copies it is repeated as (everything parented to it is repeated too)
  4. place     `position`, or `placement` resolved from the reference's exact bounds
  5. report    world bounding box, copies, triangles, warnings the model can act on

Frames: an object's `position`/`placement` is in its parent's *content frame* (the parent's
frame before the parent's own pattern), so parts of a patterned group follow every copy.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Iterable

import numpy as np
import trimesh

from oprel.integrations.scene3d.engine import attach, cuts, intent, merge, patterns, sweep
from oprel.integrations.scene3d.engine.merge import MergedSolid
from oprel.integrations.scene3d.engine.frame import (
    IDENTITY, aabb, rotation_deg, transform_points, translation, vec3,
)
from trimesh import transformations as tf
from oprel.integrations.scene3d.engine.registry import get_shape, shape_names
from oprel.integrations.scene3d.engine.schema import ALIGN_WORDS, PORT_SIDES, TOWARD, object_schema
from oprel.integrations.scene3d.engine.shapes.paths import ATTACHED, PORTED
from oprel.tools.validation import ToolArgumentError, validate_arguments

MAX_OBJECTS = 300
MAX_INSTANCES = 3000
MAX_TRIANGLES = 400_000
# Above this many points an extent is reduced to its box corners (exact enough, much faster).
_MAX_EXTENT_POINTS = 4000
_TOUCH_TOL = 0.002  # metres


class SceneError(ValueError):
    """The recipe can't be built. `errors` are written for the model to fix its call."""

    def __init__(self, errors: Iterable[str]):
        self.errors = list(dict.fromkeys(errors))  # dedupe, keep order
        super().__init__("; ".join(self.errors))


@dataclass(frozen=True)
class BuiltMesh:
    mesh: trimesh.Trimesh  # bottom-centre at the origin; shared, never mutate
    points: np.ndarray     # vertices used for exact bounds after any rotation
    notes: tuple[str, ...] = ()  # from cuts: what the model should check (a cut that misses)
    # Open ends another part can be joined to: name -> (point, outward unit direction)
    ports: tuple[tuple[str, np.ndarray, np.ndarray], ...] = ()

    def port(self, name: str) -> tuple[np.ndarray, np.ndarray] | None:
        return next(((p, d) for n, p, d in self.ports if n == name), None)


@dataclass
class CompiledObject:
    id: str
    spec: dict[str, Any]
    built: BuiltMesh | None
    position: np.ndarray                   # resolved `position` (what placement worked out to)
    local: np.ndarray                      # transform in the parent's content frame
    instances: list[np.ndarray]            # world transform of each copy of the mesh
    bbox: tuple[np.ndarray, np.ndarray]    # world bounds incl. copies and children

    @property
    def triangles(self) -> int:
        return len(self.built.mesh.faces) * len(self.instances) if self.built else 0


@dataclass
class CompiledScene:
    objects: dict[str, CompiledObject] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    merged: dict[str, MergedSolid] = field(default_factory=dict)  # merged group id -> its fused solid

    @property
    def triangles(self) -> int:
        return sum(o.triangles for o in self.objects.values())


# ---------------------------------------------------------------------------
# 1-2. check and build
# ---------------------------------------------------------------------------

@lru_cache(maxsize=512)
def _build(shape_name: str, params_json: str, cuts_json: str = "[]") -> BuiltMesh:
    sd = get_shape(shape_name)
    assert sd is not None and sd.builder is not None
    mesh = sd.builder(json.loads(params_json))
    raw_ports = mesh.metadata.pop("ports", {})
    lo, hi = mesh.bounds
    anchor = np.array([-(lo[0] + hi[0]) / 2, -lo[1], -(lo[2] + hi[2]) / 2])
    mesh.apply_translation(anchor)
    ports = tuple((name, np.asarray(p, float) + anchor, np.asarray(d, float)) for name, (p, d) in raw_ports.items())
    notes: list[str] = []
    if cuts_json != "[]":
        # Cut in the anchored frame: cut positions are from the uncut object's bottom-centre.
        mesh, notes = cuts.apply(mesh, json.loads(cuts_json), round_body=shape_name in _ROUND_SHAPES)
    points = np.asarray(mesh.vertices)
    if len(points) > _MAX_EXTENT_POINTS:
        points = _corners(*aabb(points))
    return BuiltMesh(mesh, points, tuple(notes), ports)


def _is_around(obj: dict[str, Any]) -> bool:
    return (obj.get("pattern") or {}).get("type") == "around"


def _between(obj: dict[str, Any]) -> list[str] | None:
    return (obj.get("placement") or {}).get("between")


def _spans(obj: dict[str, Any], params: dict[str, Any]) -> bool:
    """A box placed between two objects without a width: it joins them (a plate, a beam, a bridge)."""
    return obj["shape"] == "box" and bool(_between(obj)) and "width" not in params


_ROUND_SHAPES = {"cylinder", "circle", "torus", "sphere", "stool", "lathe",
                 "vase", "bottle", "cup", "bowl", "plant_pot", "wine_glass"}


def _is_round(obj: dict[str, Any]) -> bool:
    """Does the object have a round footprint (copies 'around' it go on a circle)?"""
    return obj["shape"] in _ROUND_SHAPES or (obj["shape"] == "table" and (obj.get("params") or {}).get("top") == "round")


def _corners(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def _label(obj: dict[str, Any], index: int) -> str:
    return f"'{obj['id']}'" if obj.get("id") else f"objects[{index}]"


_OBJECT_KEYS = set(object_schema()["properties"])
_HARMLESS_PARAMS = {"axis", "orientation", "direction", "toward", "rotation", "position", "color", "colour",
                    "material", "name", "label", "units", "unit"}


def _hoist_params(raw: Any) -> Any:
    """
    A shape's own params written next to "shape" instead of inside "params" ({"shape": "pipe",
    "length": 1, ...}) would be dropped as unknown keys and the shape then lacks them (seen with
    Qwen3-8B: three pipes "missing 'length'"). Move them into params.
    """
    if not isinstance(raw, dict):
        return raw
    sd = get_shape(str(raw.get("shape", "")))
    def own(k: str) -> bool:
        return sd is not None and (k in sd.params or (k.endswith("diameter")
                                                      and k[: -len("diameter")] + "radius" in sd.params))

    loose = [k for k in raw if k not in _OBJECT_KEYS and own(k)]
    if not loose:
        return raw
    out = {k: v for k, v in raw.items() if k not in loose}
    out["params"] = {**{k: raw[k] for k in loose}, **(raw.get("params") or {})}
    return out


def _from_diameters(params: dict[str, Any], known: dict[str, Any]) -> dict[str, Any]:
    """
    "diameter": 0.05 for a shape that takes a radius (users and models speak in diameters):
    radius = half. Works for any *_diameter -> *_radius pair the shape has.
    """
    for key in [k for k in params if k.endswith("diameter")]:
        target = key[: -len("diameter")] + "radius"
        value = params[key]
        if target in known and target not in params and isinstance(value, (int, float)) and key not in known:
            params[target] = value / 2
            params.pop(key)
    return params


def _check(objects: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    errors: list[str] = []
    if len(objects) > MAX_OBJECTS:
        return {}, [f"The scene has {len(objects)} objects; the limit is {MAX_OBJECTS}. Use patterns for repeats."]

    schema = object_schema()
    checked: dict[str, dict[str, Any]] = {}
    failed: set[str] = set()  # ids whose own errors are reported: don't also call them missing
    for i, raw in enumerate(objects):
        label = _label(raw, i)
        if isinstance(raw, dict) and isinstance(raw.get("shape"), str) and get_shape(raw["shape"]) is None:
            raw = dict(raw)
            intent.normalize_shape(raw)  # "counter" -> box, "mug" -> cup with a handle
            if get_shape(raw["shape"]) is None:
                if isinstance(raw.get("id"), str):
                    failed.add(raw["id"])
                near = intent.closest_shapes(raw["shape"])
                errors.append(f"{label}: there is no shape '{raw['shape']}'"
                              + (f" (closest: {', '.join(near)})" if near else "")
                              + ". Use one from the list, or build it from box / cylinder / lathe / extrude / sweep "
                                "with cuts")
                continue
        raw = _hoist_params(raw)
        if isinstance(raw.get("id"), str):
            failed.add(raw["id"])
        try:
            obj = validate_arguments(schema, raw)
        except ToolArgumentError as exc:
            errors.append(f"{label}: {exc}")
            continue
        if not obj.get("id"):
            errors.append(f"{label}: missing 'id'")
            continue
        if obj["id"] in checked:
            errors.append(f"Duplicate id '{obj['id']}'")
            continue
        sd = get_shape(obj["shape"])
        if sd is None:
            errors.append(f"{label}: unknown shape '{obj['shape']}'. Shapes: {', '.join(shape_names())}")
            continue
        params = _from_diameters(dict(obj.get("params") or {}), sd.params)
        if _spans(obj, params):
            # A box between two objects with no width reaches from one's centre to the other's;
            # the compiler works the width out once it knows where they are.
            params["width"] = 1.0
            obj["_span"] = True
        # Direction words a shape doesn't take (an elbow's "axis": joining sets its direction) are
        # harmless; warning about them sent Qwen3-8B round three more times re-sending them.
        ignored = sorted(k for k in params if k not in sd.params and k not in _HARMLESS_PARAMS)
        if ignored:
            obj["_ignored"] = ignored
        try:
            obj["params"] = sd.validate(params)
        except ToolArgumentError as exc:
            errors.append(f"{label} ({sd.name}): {exc}. Expected {sd.signature()}")
            continue
        if obj.get("merge") and sd.builder is not None:
            errors.append(f"{label}: merge is for groups: put the parts in a group with \"merge\": true "
                          f"to fuse them into one solid")
            continue
        if obj.get("cuts"):
            if sd.builder is None:
                errors.append(f"{label}: a group has no shape of its own to cut; put the cuts on one of its parts")
                continue
            try:
                obj["cuts"] = cuts.check(obj["cuts"])
            except ToolArgumentError as exc:
                errors.append(f"{label}: {exc}")
                continue
        try:
            if obj.get("pattern") and not _is_around(obj):
                patterns.copies(obj["pattern"])
        except ToolArgumentError as exc:
            errors.append(f"{label}: {exc}")
            continue
        side = obj.get("placement", {}).get("side")
        if side in PORT_SIDES and obj["shape"] in ATTACHED:
            # Seen with Qwen3-8B: a vase's handle joined to its "end". A handle goes on the side.
            obj["placement"]["side"] = side = "right"
        if obj.get("placement", {}).get("toward") and side not in PORT_SIDES:
            obj["placement"].pop("toward")  # only turns parts joined to an end; harmless elsewhere
        if side in PORT_SIDES and obj["shape"] not in PORTED:
            errors.append(f"{label}: only {', '.join(sorted(PORTED))} have ends to join; place a {obj['shape']} "
                          f"with side left/right/front/back/top instead")
            continue
        align = obj.get("placement", {}).get("align")
        if align:
            words = set(align.replace("-", " ").replace("_", " ").split())
            if not words <= ALIGN_WORDS:
                errors.append(f"{label}: placement.align must use the words {sorted(ALIGN_WORDS)}, e.g. 'front-right'")
                continue
        checked[obj["id"]] = obj
        failed.discard(obj["id"])

    for oid, obj in checked.items():
        parent = obj.get("parent")
        if parent is not None and parent not in checked:
            if parent not in failed:
                errors.append(f"'{oid}': parent '{parent}' does not exist")
        elif parent == oid:
            errors.append(f"'{oid}' cannot be its own parent")
        placement = obj.get("placement")
        between = (placement or {}).get("between")
        if between is not None:
            if len(between) != 2 or between[0] == between[1]:
                errors.append(f"'{oid}': placement.between needs two different ids, e.g. [\"post1\", \"post2\"]")
            for ref in between:
                if ref == oid:
                    errors.append(f"'{oid}' cannot be placed between itself and something else")
                elif ref not in checked:
                    if ref not in failed:
                        errors.append(f"'{oid}': placement.between '{ref}' does not exist")
                elif ref in _ancestors(oid, checked):
                    errors.append(f"'{oid}' cannot be placed between '{ref}', which contains it, and another object")
        elif placement and not placement.get("relative_to"):
            errors.append(f"'{oid}': placement needs 'relative_to' (the object to place it next to) or "
                          f"'between' (two objects to place it between)")
        elif placement:
            ref = placement["relative_to"]
            if ref == oid:
                errors.append(f"'{oid}' cannot be placed relative to itself")
            elif ref not in checked:
                if ref not in failed:  # its own error above is the one to fix
                    errors.append(f"'{oid}': placement.relative_to '{ref}' does not exist")
            elif ref != parent and ref in _ancestors(oid, checked):
                errors.append(f"'{oid}' is placed relative to '{ref}', which contains it. "
                              f"Place it relative to its direct parent or a sibling part.")
        if _is_around(obj):
            ref = obj["pattern"].get("relative_to")
            if not ref:
                errors.append(f"'{oid}': an 'around' pattern needs 'relative_to': the object to go around")
            elif ref == oid or ref not in checked:
                errors.append(f"'{oid}': pattern.relative_to '{ref}' must be another existing object")
            elif ref in _ancestors(oid, checked):
                errors.append(f"'{oid}' cannot go around '{ref}', which contains it")
    for oid in checked:
        try:
            _ancestors(oid, checked)
        except SceneError as exc:
            errors.extend(exc.errors)
    return checked, errors


def _ancestors(oid: str, objs: dict[str, dict[str, Any]]) -> list[str]:
    out: list[str] = []
    node = objs[oid].get("parent")
    while node is not None and node in objs:
        if node in out or node == oid:
            raise SceneError([f"Parent loop: '{oid}' is inside itself via {' -> '.join(out + [node])}"])
        out.append(node)
        node = objs[node].get("parent")
    return out


# ---------------------------------------------------------------------------
# 3-4. pattern and place
# ---------------------------------------------------------------------------

def _place(ref_pts: np.ndarray, own_pts: np.ndarray, placement: dict[str, Any]) -> np.ndarray:
    """Translation that puts an object (own_pts, in its own rotated frame) next to the reference."""
    rmin, rmax = aabb(ref_pts)
    omin, omax = aabb(own_pts)
    t = (rmin + rmax) / 2 - (omin + omax) / 2  # start centred on every axis
    side = placement.get("side", "center")
    gap = float(placement.get("gap", 0))

    for word in (placement.get("align") or "").replace("-", " ").replace("_", " ").split():
        if word == "left":
            t[0] = rmin[0] - omin[0]
        elif word == "right":
            t[0] = rmax[0] - omax[0]
        elif word == "back":
            t[2] = rmin[2] - omin[2]
        elif word == "front":
            t[2] = rmax[2] - omax[2]

    if side not in ("top", "bottom"):
        vertical = placement.get("vertical", "bottom")
        if vertical == "bottom":
            t[1] = rmin[1] - omin[1]
        elif vertical == "top":
            t[1] = rmax[1] - omax[1]

    if side == "right":
        t[0] = rmax[0] + gap - omin[0]
    elif side == "left":
        t[0] = rmin[0] - gap - omax[0]
    elif side == "front":
        t[2] = rmax[2] + gap - omin[2]
    elif side == "back":
        t[2] = rmin[2] - gap - omax[2]
    elif side == "top":
        t[1] = rmax[1] + gap - omin[1]
    elif side == "bottom":
        t[1] = rmin[1] - gap - omax[1]

    return t + np.array(vec3(placement.get("offset")))


def _place_between(boxes: list[tuple[np.ndarray, np.ndarray]], own_pts: np.ndarray,
                   placement: dict[str, Any]) -> np.ndarray:
    """
    Translation that centres an object midway between two others (in the ground plane), standing
    on the lower of their bottoms; vertical "top" lines its top up with the lower of their tops.
    """
    (alo, ahi), (blo, bhi) = boxes
    omin, omax = aabb(own_pts)
    mid = ((alo + ahi) / 2 + (blo + bhi) / 2) / 2
    t = mid - (omin + omax) / 2
    vertical = placement.get("vertical", "bottom")
    if vertical == "bottom":
        t[1] = min(alo[1], blo[1]) - omin[1]
    elif vertical == "top":
        t[1] = min(ahi[1], bhi[1]) - omax[1]
    return t + np.array(vec3(placement.get("offset")))


def _align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation turning unit vector a onto unit vector b (a half turn keeps up as up when it can)."""
    c = float(np.clip(np.dot(a, b), -1, 1))
    if c > 1 - 1e-9:
        return IDENTITY
    if c < -1 + 1e-9:
        axis = np.array([0.0, 1.0, 0.0]) if abs(a[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        return tf.rotation_matrix(np.pi, axis - a * np.dot(axis, a))
    return tf.rotation_matrix(np.arccos(c), np.cross(a, b))


class _Resolver:
    def __init__(self, objs: dict[str, dict[str, Any]], built: dict[str, BuiltMesh | None]):
        self.objs = objs
        self.built = built
        self.children: dict[str, list[str]] = {oid: [] for oid in objs}
        for oid, obj in objs.items():
            if obj.get("parent"):
                self.children[obj["parent"]].append(oid)
        self._fixed_copies = {oid: patterns.copies(obj.get("pattern")) for oid, obj in objs.items()
                              if not _is_around(obj)}
        self._cache: dict[tuple[str, str], Any] = {}
        self._active: set[tuple[str, str]] = set()

    def copies(self, oid: str) -> list[np.ndarray]:
        """Transforms of each copy in the object's frame. 'around' copies depend on the reference."""
        if oid in self._fixed_copies:
            return self._fixed_copies[oid]
        return self._memo("copies", oid, lambda: self._around_copies(oid))

    def _around_copies(self, oid: str) -> list[np.ndarray]:
        pattern = self.objs[oid]["pattern"]
        ref = pattern["relative_to"]
        lo, hi = aabb(self._points_in_content_frame(oid, ref, self._parent(oid)))
        olo, ohi = aabb(self._shape_points(oid))
        return patterns.around(int(pattern["count"]), lo, hi, float(ohi[2] - olo[2]),
                               float(pattern.get("gap", 0.0)), _is_round(self.objs[ref]))

    def _memo(self, kind: str, oid: str, compute):
        key = (kind, oid)
        if key in self._cache:
            return self._cache[key]
        if key in self._active:
            raise SceneError([f"Circular placement around '{oid}': objects are placed relative to each "
                              f"other in a loop. Give one of them a fixed position."])
        self._active.add(key)
        try:
            value = compute()
        finally:
            self._active.discard(key)
        self._cache[key] = value
        return value

    def _parent(self, oid: str) -> str | None:
        return self.objs[oid].get("parent")

    def _apply_copies(self, oid: str, pts: np.ndarray) -> np.ndarray:
        cps = self.copies(oid)
        if len(cps) == 1 and cps[0] is IDENTITY:
            return pts
        return np.vstack([transform_points(c, pts) for c in cps])

    def extent(self, oid: str) -> np.ndarray:
        """Points bounding the object, its copies and its children, in the object's own frame."""
        def compute() -> np.ndarray:
            parts = []
            if self.built[oid] is not None:
                parts.append(self.built[oid].points)
            for child in self.children[oid]:
                parts.append(transform_points(self.local(child), self.extent(child)))
            content = np.vstack(parts) if parts else np.zeros((1, 3))
            if len(content) * len(self.copies(oid)) > _MAX_EXTENT_POINTS:
                content = _corners(*aabb(content))
            return self._apply_copies(oid, content)
        return self._memo("extent", oid, compute)

    def _shape_points(self, oid: str) -> np.ndarray:
        """The object's own shape (or, for a group, its parts), without its pattern copies."""
        if self.built[oid] is not None:
            return self.built[oid].points
        parts = [transform_points(self.local(c), self.extent(c)) for c in self.children[oid]]
        return np.vstack(parts) if parts else np.zeros((1, 3))

    def rotation(self, oid: str) -> np.ndarray:
        """
        Turn about the shape's centre, then lift/lower it so its lowest point stays where it was:
        a lying capsule or pen still rests at its position instead of sinking into the ground.
        """
        def compute() -> np.ndarray:
            obj = self.objs[oid]
            if (obj.get("placement") or {}).get("side") in PORT_SIDES:
                return self._attach_rotation(oid)
            if _is_around(obj):
                return IDENTITY  # each copy is turned to face the reference instead
            if _between(obj) and not any(obj.get("rotation") or []):
                # Turned so its width (X) runs from the first object to the second: a plate or
                # beam joining two things that stand diagonally needs no angle from the model.
                (ca, cb) = (sum(box) / 2 for box in self.between_boxes(oid))
                dx, dz = cb[0] - ca[0], cb[2] - ca[2]
                yaw = np.degrees(np.arctan2(-dz, dx)) if abs(dx) + abs(dz) > 1e-9 else 0.0
                r = rotation_deg([0, yaw, 0])
            else:
                r = rotation_deg(obj.get("rotation"))
            if np.allclose(r, IDENTITY):
                return IDENTITY
            pts = self._shape_points(oid)
            lo, hi = aabb(pts)
            centre = (lo + hi) / 2
            m = translation(centre) @ r @ translation(-centre)
            lift = lo[1] - transform_points(m, pts)[:, 1].min()
            return translation([0, lift, 0]) @ m
        return self._memo("rotation", oid, compute)

    def position(self, oid: str) -> np.ndarray:
        """The object's position in its parent's content frame, with placement resolved."""
        def compute() -> np.ndarray:
            obj = self.objs[oid]
            if _is_around(obj):
                return np.zeros(3)  # copies are laid out in the parent's frame directly
            placement = obj.get("placement")
            if not placement:
                return np.array(vec3(obj.get("position")))
            if placement.get("side") in PORT_SIDES:
                # The part's start end on the reference's end, facing it (gap = space between them).
                start, _ = self._own_port(oid, "start")
                end, facing = self._target_port(oid)
                at = end + facing * float(placement.get("gap", 0))
                return at - transform_points(self.rotation(oid), start[None])[0] + np.array(vec3(placement.get("offset")))
            own_pts = transform_points(self.rotation(oid), self.extent(oid))
            if placement.get("between"):
                return _place_between(self.between_boxes(oid), own_pts, placement)
            ref_pts = self._points_in_content_frame(oid, placement["relative_to"], self._parent(oid))
            return _place(ref_pts, own_pts, placement)
        return self._memo("position", oid, compute)

    def mesh_in_frame(self, ref: str, frame: str | None, leave_out: str) -> trimesh.Trimesh:
        """
        `ref`'s mesh (a group: all its parts' meshes) in `frame`'s content frame, first copy of
        everything. `leave_out` (the part asking, when it is inside `ref`) is not included.
        """
        to = np.linalg.inv(self.content_frames(frame)[0])
        parts: list[trimesh.Trimesh] = []

        def add(oid: str) -> None:
            if oid == leave_out:
                return
            if self.built[oid] is not None:
                m = self.built[oid].mesh.copy()
                m.apply_transform(to @ self.frames(oid)[0] @ self.copies(oid)[0])
                parts.append(m)
            for child in self.children[oid]:
                add(child)

        add(ref)
        if not parts:
            raise SceneError([f"'{leave_out}' is attached to '{ref}', which has nothing in it to attach to"])
        return trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]

    def _own_port(self, oid: str, name: str) -> tuple[np.ndarray, np.ndarray]:
        built = self.built[oid]
        port = built.port(name) if built else None
        if port is None:
            raise SceneError([f"'{oid}' ({self.objs[oid]['shape']}) has no {name} end to join with"])
        return port

    def _target_port(self, oid: str) -> tuple[np.ndarray, np.ndarray]:
        """The end `oid` is joined to: (point, outward direction) in `oid`'s parent's content frame."""
        placement = self.objs[oid]["placement"]
        ref, name = placement["relative_to"], placement["side"]
        built = self.built[ref]
        port = built.port(name) if built else None
        if port is None:
            have = [n for n, _, _ in built.ports] if built else []
            what = (f"it only has {' and '.join(have)}" if have else
                    f"a {self.objs[ref]['shape']} has no open ends (only {', '.join(sorted(PORTED))} do)")
            raise SceneError([f"'{oid}' is joined to the {name} of '{ref}', but {what}."])
        p, d = port
        pts = np.array([p, p + d])
        frame = self._parent(oid)
        if ref != frame:
            pts = self._to_frame(ref, frame, transform_points(self.copies(ref)[0], pts))[:2]
        d2 = pts[1] - pts[0]
        return pts[0], d2 / np.linalg.norm(d2)

    def _attach_rotation(self, oid: str) -> np.ndarray:
        """
        Turn the part so its start end faces the reference's end; "toward" then rolls it about
        that line so its other end (an elbow's outlet, a tee's branch) heads that way.
        """
        _, own_dir = self._own_port(oid, "start")
        _, facing = self._target_port(oid)
        r = _align(own_dir, -facing)
        toward = self.objs[oid]["placement"].get("toward")
        built = self.built[oid]
        other = (built.port("branch") or built.port("end")) if built else None
        if toward and other is not None:
            heads = r[:3, :3] @ other[1]
            want = np.array(TOWARD[toward], float)
            heads_p = heads - facing * np.dot(heads, facing)
            want_p = want - facing * np.dot(want, facing)
            if np.linalg.norm(heads_p) > 1e-6 and np.linalg.norm(want_p) > 1e-6:
                angle = np.arctan2(np.dot(np.cross(heads_p, want_p), facing), np.dot(heads_p, want_p))
                r = tf.rotation_matrix(angle, facing) @ r
        return r

    def between_boxes(self, oid: str) -> list[tuple[np.ndarray, np.ndarray]]:
        """Bounds of the two objects `oid` is placed between, in `oid`'s parent's content frame."""
        return [aabb(self._points_in_content_frame(oid, ref, self._parent(oid)))
                for ref in _between(self.objs[oid])]

    def local(self, oid: str) -> np.ndarray:
        return translation(self.position(oid)) @ self.rotation(oid)

    def content_frames(self, oid: str | None) -> list[np.ndarray]:
        """World transforms of `oid`'s content frame, one per copy (None = the world itself)."""
        if oid is None:
            return [IDENTITY]
        return [f @ c for f in self.frames(oid) for c in self.copies(oid)]

    def frames(self, oid: str) -> list[np.ndarray]:
        """World transforms of the object's own frame, one per copy of its ancestors."""
        def compute() -> list[np.ndarray]:
            loc = self.local(oid)
            return [b @ loc for b in self.content_frames(self._parent(oid))]
        return self._memo("frames", oid, compute)

    def _points_in_content_frame(self, oid: str, ref: str, frame: str | None) -> np.ndarray:
        if ref == frame:
            # Relative to its own parent: use the parent's own mesh, not the parent's
            # children (that would include this object).
            own = self.built[ref]
            if own is None:
                raise SceneError([f"'{oid}' is placed relative to its own group '{ref}', which has no shape "
                                  f"of its own. Place it relative to a part inside the group."])
            return own.points
        return self._to_frame(ref, frame, self.extent(ref))

    def _to_frame(self, ref: str, frame: str | None, pts: np.ndarray) -> np.ndarray:
        """Points in `ref`'s own frame -> `frame`'s content frame (copies of ancestors appended)."""
        pts = transform_points(self.local(ref), pts)
        node = self._parent(ref)
        while node != frame:
            if node is None:
                # `frame` is not an ancestor of `ref`: pts are world coordinates now.
                inv = np.linalg.inv(self.content_frames(frame)[0])
                return transform_points(inv, pts)
            pts = transform_points(self.local(node), self._apply_copies(node, pts))
            node = self._parent(node)
        return pts


# ---------------------------------------------------------------------------
# 5. report
# ---------------------------------------------------------------------------

def _boxes_touch_below(box: tuple[np.ndarray, np.ndarray], other: tuple[np.ndarray, np.ndarray]) -> bool:
    """Does `other` reach up to `box`'s bottom and overlap its footprint?"""
    (lo, hi), (olo, ohi) = box, other
    if not (olo[1] - _TOUCH_TOL <= lo[1] <= ohi[1] + _TOUCH_TOL):
        return False
    return bool(olo[0] <= hi[0] + _TOUCH_TOL and lo[0] <= ohi[0] + _TOUCH_TOL
                and olo[2] <= hi[2] + _TOUCH_TOL and lo[2] <= ohi[2] + _TOUCH_TOL)


def _overhanging_supports(scene: CompiledScene) -> list[str]:
    """
    Repeated supports (legs, pillars) that reach outside the thing resting on them, e.g. five
    legs in a 4 m row under a 1.2 m table top. Only patterned objects are checked: a table top
    is meant to be wider than the vase on it.
    """
    out: list[str] = []
    meshes = [co for co in scene.objects.values() if co.built is not None]
    for support in meshes:
        if len(support.instances) < 2:
            continue
        slo, shi = support.bbox
        for held in meshes:
            if held is support:
                continue
            hlo, hhi = held.bbox
            if abs(hlo[1] - shi[1]) > _TOUCH_TOL or not _boxes_touch_below(held.bbox, support.bbox):
                continue
            margin = 0.02 * max(hhi[0] - hlo[0], hhi[2] - hlo[2]) + _TOUCH_TOL
            overhang = max(max(hlo[a] - slo[a], shi[a] - hhi[a]) for a in (0, 2))
            if overhang > margin:
                axis = max((0, 2), key=lambda a: max(hlo[a] - slo[a], shi[a] - hhi[a]))
                name = "xz"[axis // 2]
                out.append(
                    f"The {len(support.instances)} copies of '{support.id}' stick out {overhang:.2f} m past "
                    f"'{held.id}', which rests on them ({name} {slo[axis]:.2f}..{shi[axis]:.2f} vs "
                    f"{hlo[axis]:.2f}..{hhi[axis]:.2f}). Shrink the pattern by at least {overhang:.2f} m."
                )
    return out


def _box_gap(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]) -> float:
    """Distance between two boxes (0 when they touch or overlap)."""
    gaps = np.maximum(0, np.maximum(b[0] - a[1], a[0] - b[1]))
    return float(np.linalg.norm(gaps))


def _detached_parts(scene: CompiledScene, instance_boxes: dict[str, list[tuple[np.ndarray, np.ndarray]]]) -> list[str]:
    """
    Parts of a group that touch neither the ground nor anything else, e.g. a table top left
    5 cm above legs that were shortened. (Top-level objects get the 'floats' check instead.)
    """
    out: list[str] = []
    for oid, boxes in instance_boxes.items():
        if not scene.objects[oid].spec.get("parent"):
            continue
        others = [(other, box) for other, obs in instance_boxes.items() if other != oid for box in obs]
        for box in boxes:
            if box[0][1] <= _TOUCH_TOL:
                continue  # stands on the ground
            gaps = sorted((_box_gap(box, ob), other) for other, ob in others)
            if not gaps or gaps[0][0] > _TOUCH_TOL:
                nearest = f": it is {gaps[0][0]:.3f} m from '{gaps[0][1]}'" if gaps else ""
                out.append(f"'{oid}' touches nothing{nearest}. Move or resize it so the parts connect.")
                break
    return out


def _misplaced_twins(scene: CompiledScene, instance_boxes: dict[str, list[tuple[np.ndarray, np.ndarray]]]) -> list[str]:
    """
    A part identical to a sibling that stands on the ground, but which itself does not: almost
    always y written where z was meant (seen with Qwen3-8B: legs at [0, 0.5, 0] and [0, -0.5, 0]).
    """
    out: list[str] = []
    on_ground: dict[tuple[Any, str, str], str] = {}
    # Only parts of one group: two separate top-level objects (a vase on a shelf next to the
    # same vase on the floor) are not a y/z mix-up.
    # Placed parts are where the engine put them (a second elbow up a pipe run is not a mix-up).
    instance_boxes = {oid: b for oid, b in instance_boxes.items()
                      if scene.objects[oid].spec.get("parent") and not scene.objects[oid].spec.get("placement")}
    for oid, boxes in instance_boxes.items():
        spec = scene.objects[oid].spec
        key = (spec.get("parent"), spec["shape"], json.dumps(spec.get("params"), sort_keys=True))
        if min(b[0][1] for b in boxes) <= _TOUCH_TOL and key not in on_ground:
            on_ground[key] = oid
    for oid, boxes in instance_boxes.items():
        spec = scene.objects[oid].spec
        twin = on_ground.get((spec.get("parent"), spec["shape"], json.dumps(spec.get("params"), sort_keys=True)))
        lowest = min(b[0][1] for b in boxes)
        if twin and twin != oid and abs(lowest) > _TOUCH_TOL:
            out.append(
                f"'{oid}' is identical to '{twin}', which stands on the ground, but its bottom is at y={lowest:.2f}. "
                f"Positions are [x, y, z] with y = height: to move a part sideways change z, not y "
                f"(or use one object with a pattern)."
            )
    return out


# Tall vessels: wider than this many times their height no longer looks like the thing.
_MAX_WIDTH_TO_HEIGHT = {"vase": 1.6, "bottle": 1.2, "cup": 2.0, "wine_glass": 1.6}


def _odd_proportions(scene: CompiledScene) -> list[str]:
    """
    A vase 4 m wide and 1.2 m tall is a flat disc: usually a number from the user taken as
    metres when it meant cm or 'twice as wide' (seen with Qwen3-8B: "bottom radius to 2").
    """
    out: list[str] = []
    for oid, co in scene.objects.items():
        params = co.spec.get("params") or {}
        if co.spec["shape"] == "vase" and params.get("top_radius") and params.get("neck_radius"):
            # Seen with Qwen3-8B, "an amphora with a flared rim": top_radius 0.2 on a 0.05 neck and a
            # 0.15 belly, a funnel on a pot. A flared rim is a lip, about 1.3-1.6 x the neck.
            top, neck = params["top_radius"], params["neck_radius"]
            belly = max(params.get("radius") or 0, params.get("base_radius") or 0)
            if top > max(belly, 2.5 * neck):
                out.append(f"'{oid}': its rim (top_radius {top:g}) is wider than its belly ({belly:g}), so the top "
                           f"is a funnel. A flared rim is about 1.5 x the neck: top_radius {round(1.5 * neck, 3):g}. "
                           f"Keep {top:g} only if the user asked for a rim that wide.")
        limit = _MAX_WIDTH_TO_HEIGHT.get(co.spec["shape"])
        if not limit or co.built is None:
            continue
        lo, hi = aabb(co.built.points)
        width, height = float(max(hi[0] - lo[0], hi[2] - lo[2])), float(hi[1] - lo[1])
        if width > limit * height:
            radii = {k: v for k, v in (co.spec.get("params") or {}).items() if k.endswith("radius") and v}
            biggest = max(radii, key=radii.get) if radii else "radius"
            # Same ratios between the radii, scaled to a typical width (half the height for a vase).
            scale = (0.5 * limit / 1.6 * height / 2) / max(radii.values()) if radii else 1
            suggested = {k: round(v * scale, 3) for k, v in radii.items()}
            out.append(
                f"'{oid}' is {width:.2f} m wide but only {height:.2f} m tall, so it no longer looks like a "
                f"{co.spec['shape']} ({biggest} {radii.get(biggest, 0):g} m). User numbers without units are "
                f"usually relative: same ratios at a {co.spec['shape']}-like size would be params "
                f"{json.dumps(suggested)}."
            )
    return out


# Solid all the way through: one standing inside another is never meant (a vase or a table has room inside).
_SOLID_SHAPES = {"box", "cylinder", "cone", "sphere", "capsule"}


def _inside_another(scene: CompiledScene) -> list[str]:
    """
    A solid standing mostly inside a bigger one beside it. Seen with Qwen3-8B: "a thin cylinder
    with a 15 cm gap to the big one" put 15 cm between the centres, so it stood inside the big one.
    """
    out: list[str] = []
    cands = [co for co in scene.objects.values()
             if co.built is not None and co.spec["shape"] in _SOLID_SHAPES and len(co.instances) == 1
             and not _between(co.spec)]
    for small in cands:
        if small.spec["shape"] == "box":
            continue  # often a plate or beam meant to reach into things; `between` covers joining
        slo, shi = small.bbox
        for big in cands:
            blo, bhi = big.bbox
            if big is small or small.spec.get("parent") != big.spec.get("parent"):
                continue
            if np.prod(shi - slo) >= np.prod(bhi - blo):
                continue  # judge the smaller one
            overlap = np.clip(np.minimum(shi, bhi) - np.maximum(slo, blo), 0, None)
            if np.prod(overlap) < 0.5 * np.prod(shi - slo):
                continue
            sc, bc = (slo + shi) / 2, (blo + bhi) / 2
            dx, dz = sc[0] - bc[0], sc[2] - bc[2]
            side = ("right" if dx > 0 else "left") if abs(dx) >= abs(dz) else ("front" if dz > 0 else "back")
            dist = float(np.hypot(dx, dz))
            out.append(
                f"'{small.id}' stands inside '{big.id}': their centres are {dist:.3g} m apart, but '{big.id}' "
                f"alone reaches {float(max(bhi[0] - blo[0], bhi[2] - blo[2])) / 2:.3g} m from its centre. A gap is "
                f"measured between their surfaces: let the engine do it with \"placement\": {{\"relative_to\": "
                f"\"{big.id}\", \"side\": \"{side}\", \"gap\": {max(round(dist, 3), 0.01):g}}} on '{small.id}'."
            )
            break
    return out


_JOIN_WORDS = re.compile(r"join|connect|link|bridge|beam|plate|web|brace|strut|rail", re.IGNORECASE)


def _loose_connectors(scene: CompiledScene) -> list[str]:
    """
    A box named like a connector ("joining plate", "beam") that touches fewer than two things.
    Seen with Qwen3-8B: the plate meant to join two cylinders was drawn from coordinates, and
    stayed behind when one cylinder was moved.
    """
    out: list[str] = []
    for co in scene.objects.values():
        name = f"{co.id} {co.spec.get('label', '')}"
        if co.built is None or co.spec["shape"] != "box" or _between(co.spec) or not _JOIN_WORDS.search(name):
            continue
        size = co.bbox[1] - co.bbox[0]
        if size[1] < 0.25 * min(size[0], size[2]):
            continue  # a flat slab (a base plate things stand on), not a web or beam between things
        others = sorted(
            (_box_gap(co.bbox, other.bbox), other.id) for other in scene.objects.values()
            if other is not co and other.built is not None and other.spec.get("parent") == co.spec.get("parent")
        )
        if len(others) < 2 or sum(gap <= _TOUCH_TOL for gap, _ in others) >= 2:
            continue
        (_, a), (_, b) = others[:2]
        touching = [oid for gap, oid in others if gap <= _TOUCH_TOL]
        state = f"touches only '{touching[0]}'" if touching else "touches nothing"
        out.append(
            f"'{co.id}' {state}. To join two objects let the engine size and place it: \"placement\": "
            f"{{\"between\": [\"{a}\", \"{b}\"]}} and leave out its width."
        )
    return out


_HANDLE_WORDS = re.compile(r"handle|grip", re.IGNORECASE)


def _stray_handles(scene: CompiledScene) -> list[str]:
    """
    A box or cylinder named "handle" that touches nothing. Seen with Qwen3-8B, "a curvy vase with a
    box handle": a 5 cm box on the floor beside the vase. The handle shape attaches itself.
    """
    out: list[str] = []
    for co in scene.objects.values():
        name = f"{co.id} {co.spec.get('label', '')}"
        if co.built is None or co.spec["shape"] == "handle" or not _HANDLE_WORDS.search(name):
            continue
        others = sorted((_box_gap(co.bbox, o.bbox), o.id) for o in scene.objects.values()
                        if o is not co and o.built is not None)
        if not others or others[0][0] <= _TOUCH_TOL:
            continue
        style = "box" if co.spec["shape"] in ("box", "rectangle") else "round"
        out.append(f"'{co.id}' touches nothing ({others[0][0]:.3f} m from '{others[0][1]}'). For a handle use "
                   f"{{\"id\": \"{co.id}\", \"shape\": \"handle\", \"params\": {{\"style\": \"{style}\"}}, "
                   f"\"placement\": {{\"relative_to\": \"{others[0][1]}\", \"side\": \"right\"}}}}: it attaches "
                   f"to the surface and is sized to the object (delete '{co.id}' first, or use a new id).")
    return out


def _loose_ends(scene: CompiledScene) -> list[str]:
    """
    Two open ends close together, facing each other, but not joined (a pipe run with a gap), and
    two parts joined to the same end (they sit inside each other).
    """
    ends = []
    for oid, co in scene.objects.items():
        if co.built is None or not co.built.ports or len(co.instances) != 1:
            continue
        m = co.instances[0]
        size = float(np.max(co.bbox[1] - co.bbox[0]))
        ends += [(oid, n, transform_points(m, p[None])[0], m[:3, :3] @ d, size) for n, p, d in co.built.ports]
    out: list[str] = []
    on_end: dict[tuple[str, str], list[str]] = {}
    for oid, co in scene.objects.items():
        pl = co.spec.get("placement") or {}
        if pl.get("side") in PORT_SIDES:
            on_end.setdefault((pl["relative_to"], pl["side"]), []).append(oid)
    for (ref, side), parts in on_end.items():
        if len(parts) > 1:
            out.append(f"{' and '.join(repr(p) for p in parts)} are all joined to the {side} of '{ref}', so they "
                       f"overlap. Join each part to the one before it, e.g. '{parts[1]}' to the end of '{parts[0]}'.")
    for i, (oa, na, pa, da, sa) in enumerate(ends):
        for ob, nb, pb, db, sb in ends[i + 1:]:
            dist = float(np.linalg.norm(pa - pb))
            if oa == ob or dist <= _TOUCH_TOL or dist > min(0.1, 0.25 * max(sa, sb)) or np.dot(da, db) > -0.5:
                continue
            if nb == "start":
                fix = f': put "placement": {{"relative_to": "{oa}", "side": "{na}"}} on \'{ob}\''
            elif na == "start":
                fix = f': put "placement": {{"relative_to": "{ob}", "side": "{nb}"}} on \'{oa}\''
            else:
                fix = '. Join them with placement side "end" so they meet exactly'
            out.append(f"The {na} of '{oa}' is {dist:.3f} m from the {nb} of '{ob}' but they are not joined{fix}.")
    return out


def _below_ground(oid: str, lowest: float, spec: dict[str, Any]) -> str:
    placement = spec.get("placement") or {}
    if placement.get("side") == "bottom" and placement.get("relative_to"):
        # Seen with Qwen3-8B: side "bottom" used to mean "at the bottom, next to it".
        return (f"'{oid}' goes {-lowest:.3f} m below the ground: side \"bottom\" puts it underneath "
                f"'{placement['relative_to']}'. To stand it on the ground next to it use side \"left\", \"right\", "
                f"\"front\" or \"back\"; to join two objects use \"placement\": {{\"between\": [\"a\", \"b\"]}}.")
    return f"'{oid}' goes {-lowest:.3f} m below the ground (y=0)."


def _warnings(scene: CompiledScene, objs: dict[str, dict[str, Any]]) -> list[str]:
    out: list[str] = []
    instance_boxes: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    has_parts = {obj.get("parent") for obj in objs.values()}
    for oid, co in scene.objects.items():
        if co.spec.get("_ignored"):
            # Seen with Qwen3-8B: "handles": 2 on a plant_pot (which had none) silently dropped.
            sd = get_shape(co.spec["shape"])
            out.append(f"'{oid}': a {co.spec['shape']} has no {', '.join(repr(k) for k in co.spec['_ignored'])}, so "
                       f"{'it was' if len(co.spec['_ignored']) == 1 else 'they were'} ignored. Its params: "
                       f"{sd.signature() if sd else ''}.")
        if co.spec.get("placement") and co.spec.get("position"):
            out.append(f"'{oid}': both position and placement were given; position was ignored.")
        if co.built is None:
            if oid not in has_parts:
                out.append(f"'{oid}' is an empty group: set \"parent\": \"{oid}\" on its parts "
                           f"so they move together.")
            continue
        out.extend(f"'{oid}': {note}" for note in co.built.notes)
        boxes = [aabb(transform_points(m, co.built.points)) for m in co.instances]
        instance_boxes[oid] = boxes
        lowest = min(b[0][1] for b in boxes)
        if lowest < -_TOUCH_TOL:
            out.append(_below_ground(oid, lowest, co.spec))

    out.extend(_overhanging_supports(scene))
    out.extend(_inside_another(scene))
    out.extend(_loose_connectors(scene))
    out.extend(_detached_parts(scene, instance_boxes))
    out.extend(_misplaced_twins(scene, instance_boxes))
    out.extend(_odd_proportions(scene))
    out.extend(_loose_ends(scene))
    out.extend(_stray_handles(scene))

    def subtree(root: str) -> set[str]:
        ids = {root}
        changed = True
        while changed:
            changed = False
            for oid, obj in objs.items():
                if oid not in ids and obj.get("parent") in ids:
                    ids.add(oid)
                    changed = True
        return ids

    for oid, co in scene.objects.items():
        if co.spec.get("parent"):
            continue  # parts are judged together with their top-level object
        lo = co.bbox[0]
        if lo[1] <= _TOUCH_TOL:
            continue
        own = subtree(oid)
        if not own & instance_boxes.keys():
            continue  # empty group
        supported = any(
            _boxes_touch_below(co.bbox, box)
            for other, boxes in instance_boxes.items() if other not in own
            for box in boxes
        )
        if not supported:
            out.append(f"'{oid}' floats {lo[1]:.3f} m above the ground with nothing under it.")
    return out


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def _is_attached(obj: dict[str, Any]) -> bool:
    """A handle placed beside an object: built onto that object's surface."""
    pl = obj.get("placement") or {}
    return obj["shape"] in ATTACHED and bool(pl.get("relative_to")) and pl.get("side") in attach.SIDES


def _attached_mesh(resolver: _Resolver, oid: str) -> trimesh.Trimesh:
    """A handle built onto the surface of the object it is placed beside, in its parent's content frame."""
    obj = resolver.objs[oid]
    pl, p = obj["placement"], obj.get("params") or {}
    ref, side = pl["relative_to"], pl["side"]
    target = resolver.mesh_in_frame(ref, resolver._parent(oid), oid)
    orientation = "horizontal" if str(p.get("orientation", "")).lower().startswith("h") else "vertical"
    f = attach.frame_for(side, orientation, target.bounds)
    off = np.array(vec3(pl.get("offset")))
    extent = float(abs(np.ptp(target.bounds, axis=0) @ f.U))
    span, thickness, reach = attach.defaults(p.get("height"), p.get("thickness"), p.get("reach"), extent,
                                             orientation, side)
    u_hi, u_lo, level = attach.ends(target.bounds, f, span, side, orientation, pl.get("vertical"), float(off @ f.U))
    f.k0 = level + float(off @ f.K)
    return attach.build(target, f, u_hi, u_lo, reach=reach, thickness=thickness,
                        style=sweep.handle_style(p.get("style")), ear=orientation == "vertical" and side != "top",
                        what=f"'{ref}'")


def compile_scene(objects: list[dict[str, Any]]) -> CompiledScene:
    """Build every object. Raises SceneError listing everything the caller must fix."""
    objs, errors = _check(objects)
    if errors:
        raise SceneError(errors)

    built: dict[str, BuiltMesh | None] = {}
    for oid, obj in objs.items():
        sd = get_shape(obj["shape"])
        if sd is None or sd.builder is None:
            built[oid] = None
            continue
        if _is_attached(obj):
            # A stand-in until the third pass builds it onto its object (its sizes are checked there).
            built[oid] = _build(sd.name, json.dumps(sd.validate({}), sort_keys=True))
            continue
        try:
            built[oid] = _build(sd.name, json.dumps(obj["params"], sort_keys=True),
                                json.dumps(obj.get("cuts") or [], sort_keys=True))
        except ToolArgumentError as exc:
            errors.append(f"'{oid}' ({sd.name}): {exc}")
    if errors:
        raise SceneError(errors)

    resolver = _Resolver(objs, built)
    spans = [oid for oid, obj in objs.items() if obj.get("_span")]
    if spans:
        # Second pass: now that the objects they join are placed, give spanning boxes their width
        # (centre to centre, so the box reaches into both) and rebuild with the real sizes.
        for oid in spans:
            try:
                (alo, ahi), (blo, bhi) = resolver.between_boxes(oid)
            except SceneError as exc:
                errors.extend(exc.errors)
                continue
            d = (blo + bhi) / 2 - (alo + ahi) / 2
            width = float(np.hypot(d[0], d[2]))
            a, b = _between(objs[oid])
            if width < 1e-4:
                errors.append(f"'{oid}' is between '{a}' and '{b}', but they stand on the same spot, so there is "
                              f"nothing to span. Give it a width, or move one of them.")
                continue
            objs[oid]["params"]["width"] = round(width, 6)
            built[oid] = _build("box", json.dumps(objs[oid]["params"], sort_keys=True),
                                json.dumps(objs[oid].get("cuts") or [], sort_keys=True))
        if errors:
            raise SceneError(errors)
        resolver = _Resolver(objs, built)
    attached = [oid for oid, obj in objs.items() if _is_attached(obj)]
    if attached:
        # Third pass: handles are built onto the real surface of what they are attached to, now that
        # it is placed (engine/attach.py), then fixed where they were built.
        for oid in attached:
            try:
                mesh = _attached_mesh(resolver, oid)
            except ToolArgumentError as exc:
                errors.append(f"'{oid}' ({objs[oid]['shape']}): {exc}")
                continue
            except SceneError as exc:
                errors.extend(exc.errors)
                continue
            lo, hi = mesh.bounds
            anchor = np.array([(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2])
            mesh.apply_translation(-anchor)
            pts = np.asarray(mesh.vertices)
            built[oid] = BuiltMesh(mesh, _corners(*aabb(pts)) if len(pts) > _MAX_EXTENT_POINTS else pts)
            spec = {k: v for k, v in objs[oid].items() if k not in ("placement", "rotation")}
            objs[oid] = {**spec, "position": anchor.tolist(), "_attached_to": objs[oid]["placement"]["relative_to"]}
        if errors:
            raise SceneError(errors)
        resolver = _Resolver(objs, built)
    scene = CompiledScene()
    instance_total = 0
    for oid, obj in objs.items():
        try:
            frames = resolver.frames(oid)
            extent = resolver.extent(oid)
        except SceneError as exc:
            errors.extend(exc.errors)
            continue
        instances = [f @ c for f in frames for c in resolver.copies(oid)] if built[oid] else []
        instance_total += len(instances)
        world = np.vstack([transform_points(f, extent) for f in frames])
        scene.objects[oid] = CompiledObject(
            oid, obj, built[oid], resolver.position(oid), resolver.local(oid), instances, aabb(world)
        )
    if errors:
        raise SceneError(errors)

    if instance_total > MAX_INSTANCES:
        raise SceneError([f"The scene has {instance_total} mesh copies; the limit is {MAX_INSTANCES}."])
    if scene.triangles > MAX_TRIANGLES:
        raise SceneError([f"The scene has {scene.triangles} triangles; the limit is {MAX_TRIANGLES}. "
                          f"Lower 'segments' or the number of copies."])
    try:
        scene.merged = merge.merge_groups(scene.objects)
    except ToolArgumentError as exc:
        raise SceneError([f"Could not merge: {exc}"]) from exc
    scene.warnings = _warnings(scene, objs) + _split_merges(scene)
    return scene


def _split_merges(scene: CompiledScene) -> list[str]:
    """A merged group that still comes out as several bodies: some part touches nothing else in it."""
    out: list[str] = []
    for gid, solid in scene.merged.items():
        if solid.bodies <= 1:
            continue
        boxes = {oid: [aabb(transform_points(m, scene.objects[oid].built.points)) for m in scene.objects[oid].instances]
                 for oid in solid.parts}
        loose = [oid for oid, own in boxes.items()
                 if not any(_box_gap(a, b) <= _TOUCH_TOL for a in own
                            for other, theirs in boxes.items() if other != oid for b in theirs)]
        which = f": {', '.join(repr(o) for o in loose)} touch{'es' if len(loose) == 1 else ''} nothing else in it" \
            if loose else ""
        out.append(f"'{gid}' is merged but comes out as {solid.bodies} separate bodies{which}. Move the parts so "
                   f"they touch or overlap, or take the loose ones out of the group.")
    return out
