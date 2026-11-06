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
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Iterable

import numpy as np
import trimesh

from oprel.integrations.scene3d.engine import patterns
from oprel.integrations.scene3d.engine.frame import (
    IDENTITY, aabb, rotation_deg, transform_points, translation, vec3,
)
from oprel.integrations.scene3d.engine.registry import get_shape, shape_names
from oprel.integrations.scene3d.engine.schema import ALIGN_WORDS, object_schema
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

    @property
    def triangles(self) -> int:
        return sum(o.triangles for o in self.objects.values())


# ---------------------------------------------------------------------------
# 1-2. check and build
# ---------------------------------------------------------------------------

@lru_cache(maxsize=512)
def _build(shape_name: str, params_json: str) -> BuiltMesh:
    sd = get_shape(shape_name)
    assert sd is not None and sd.builder is not None
    mesh = sd.builder(json.loads(params_json))
    lo, hi = mesh.bounds
    mesh.apply_translation([-(lo[0] + hi[0]) / 2, -lo[1], -(lo[2] + hi[2]) / 2])
    points = np.asarray(mesh.vertices)
    if len(points) > _MAX_EXTENT_POINTS:
        points = _corners(*aabb(points))
    return BuiltMesh(mesh, points)


def _is_around(obj: dict[str, Any]) -> bool:
    return (obj.get("pattern") or {}).get("type") == "around"


_ROUND_SHAPES = {"cylinder", "circle", "torus", "sphere", "stool", "lathe",
                 "vase", "bottle", "cup", "bowl", "plant_pot", "wine_glass"}


def _is_round(obj: dict[str, Any]) -> bool:
    """Does the object have a round footprint (copies 'around' it go on a circle)?"""
    return obj["shape"] in _ROUND_SHAPES or (obj["shape"] == "table" and (obj.get("params") or {}).get("top") == "round")


def _corners(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def _label(obj: dict[str, Any], index: int) -> str:
    return f"'{obj['id']}'" if obj.get("id") else f"objects[{index}]"


def _check(objects: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    errors: list[str] = []
    if len(objects) > MAX_OBJECTS:
        return {}, [f"The scene has {len(objects)} objects; the limit is {MAX_OBJECTS}. Use patterns for repeats."]

    schema = object_schema()
    checked: dict[str, dict[str, Any]] = {}
    for i, raw in enumerate(objects):
        label = _label(raw, i)
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
        try:
            obj["params"] = sd.validate(obj.get("params"))
        except ToolArgumentError as exc:
            errors.append(f"{label} ({sd.name}): {exc}. Expected {sd.signature()}")
            continue
        try:
            if obj.get("pattern") and not _is_around(obj):
                patterns.copies(obj["pattern"])
        except ToolArgumentError as exc:
            errors.append(f"{label}: {exc}")
            continue
        align = obj.get("placement", {}).get("align")
        if align:
            words = set(align.replace("-", " ").replace("_", " ").split())
            if not words <= ALIGN_WORDS:
                errors.append(f"{label}: placement.align must use the words {sorted(ALIGN_WORDS)}, e.g. 'front-right'")
                continue
        checked[obj["id"]] = obj

    for oid, obj in checked.items():
        parent = obj.get("parent")
        if parent is not None and parent not in checked:
            errors.append(f"'{oid}': parent '{parent}' does not exist")
        elif parent == oid:
            errors.append(f"'{oid}' cannot be its own parent")
        placement = obj.get("placement")
        if placement:
            ref = placement["relative_to"]
            if ref == oid:
                errors.append(f"'{oid}' cannot be placed relative to itself")
            elif ref not in checked:
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
            if _is_around(self.objs[oid]):
                return IDENTITY  # each copy is turned to face the reference instead
            r = rotation_deg(self.objs[oid].get("rotation"))
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
            ref_pts = self._points_in_content_frame(oid, placement["relative_to"], self._parent(oid))
            own_pts = transform_points(self.rotation(oid), self.extent(oid))
            return _place(ref_pts, own_pts, placement)
        return self._memo("position", oid, compute)

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
        pts = transform_points(self.local(ref), self.extent(ref))
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
    instance_boxes = {oid: b for oid, b in instance_boxes.items() if scene.objects[oid].spec.get("parent")}
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


def _warnings(scene: CompiledScene, objs: dict[str, dict[str, Any]]) -> list[str]:
    out: list[str] = []
    instance_boxes: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    has_parts = {obj.get("parent") for obj in objs.values()}
    for oid, co in scene.objects.items():
        if co.spec.get("placement") and co.spec.get("position"):
            out.append(f"'{oid}': both position and placement were given; position was ignored.")
        if co.built is None:
            if oid not in has_parts:
                out.append(f"'{oid}' is an empty group: set \"parent\": \"{oid}\" on its parts "
                           f"so they move together.")
            continue
        boxes = [aabb(transform_points(m, co.built.points)) for m in co.instances]
        instance_boxes[oid] = boxes
        lowest = min(b[0][1] for b in boxes)
        if lowest < -_TOUCH_TOL:
            out.append(f"'{oid}' goes {-lowest:.3f} m below the ground (y=0).")

    out.extend(_overhanging_supports(scene))
    out.extend(_detached_parts(scene, instance_boxes))
    out.extend(_misplaced_twins(scene, instance_boxes))
    out.extend(_odd_proportions(scene))

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
        try:
            built[oid] = _build(sd.name, json.dumps(obj["params"], sort_keys=True))
        except ToolArgumentError as exc:
            errors.append(f"'{oid}' ({sd.name}): {exc}")
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
    scene.warnings = _warnings(scene, objs)
    return scene
