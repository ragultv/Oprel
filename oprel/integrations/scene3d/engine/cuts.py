"""
Row 4: holes and cuts. Any object can carry `cuts`: shapes removed from its mesh (drilled
holes, slots, pockets, notches, windows), done with exact mesh booleans (manifold3d).

A cut is {shape, params, position, rotation, side, pattern}, in the object's own frame
(its bottom-centre is [0, 0, 0]):
  position  the cut's centre (default: the middle of the object)
  length    leave out a cylinder's height / a box's size / an extrude's thickness and the cut
            goes right through the object along that direction
  side      words from top/bottom/left/right/front/back: the cut starts flush with that face
            of the object (a pocket from the top, a notch in the front edge)
  pattern   linear / grid / radial copies of the cut, laid out in the object's frame

Faces of a cut that lie flush with the object's outside are pushed out a hair, so the cut
comes out clean instead of leaving a paper-thin skin.
"""

from __future__ import annotations

from typing import Any

import manifold3d as m3
import numpy as np
import trimesh

from oprel.integrations.scene3d.engine import patterns
from oprel.integrations.scene3d.engine.frame import rotation_deg, transform_points, translation, vec3
from oprel.integrations.scene3d.engine.registry import get_shape
from oprel.tools.validation import ToolArgumentError, validate_arguments

MAX_CUTS = 20
MAX_CUT_COPIES = 200

_VEC3 = {"type": "array", "items": {"type": "number"}, "maxItems": 3}
# Sent to the model inside every object schema, so kept loose (the system hint shows the
# fields with an example; spelling them out here cost ~100 tokens of a 4k context).
SCHEMA: dict[str, Any] = {"type": "array", "items": {"type": "object", "additionalProperties": True}}
# What check() actually enforces.
_FULL_SCHEMA: dict[str, Any] = {
    "type": "array",
    "maxItems": MAX_CUTS,
    "items": {
        "type": "object",
        "properties": {
            "shape": {"type": "string"},
            "params": {"type": "object", "additionalProperties": True},
            "position": _VEC3,
            "rotation": _VEC3,
            "side": {"type": "string"},
            "pattern": {"type": "object", "additionalProperties": True},
        },
        "required": ["shape"],
    },
}

# Shapes a cut can be: plain solids (cutting a chair out of something is never what was meant).
CUTTERS = ("cylinder", "box", "sphere", "cone", "capsule", "torus", "extrude", "lathe", "sweep")
# Lengths that, when left out, make the cut go right through the object.
THROUGH = {"cylinder": ("height",), "box": ("width", "height", "depth"), "extrude": ("thickness",)}
# side word -> (axis, direction)
SIDES = {"left": (0, -1), "right": (0, 1), "bottom": (1, -1), "top": (1, 1), "back": (2, -1), "front": (2, 1)}
_PLACEHOLDER = 1.0


def _words(side: str | None) -> list[str]:
    return (side or "").replace("-", " ").replace("_", " ").replace(",", " ").split()


def check(cuts: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Validate the cuts of one object; returns them with defaults filled. Raises ToolArgumentError."""
    if not cuts:
        return []
    cuts = validate_arguments({"type": "object", "properties": {"cuts": _FULL_SCHEMA}}, {"cuts": cuts})["cuts"]
    out, total = [], 0
    for i, cut in enumerate(cuts):
        label = f"cuts[{i}]"
        sd = get_shape(cut["shape"])
        if sd is None or sd.name not in CUTTERS:
            raise ToolArgumentError(f"{label}: '{cut['shape']}' can't be cut out; use one of {', '.join(CUTTERS)}")
        through = [k for k in THROUGH.get(sd.name, ()) if k not in (cut.get("params") or {})]
        try:
            params = sd.validate({**(cut.get("params") or {}), **{k: _PLACEHOLDER for k in through}})
        except ToolArgumentError as exc:
            raise ToolArgumentError(f"{label} ({sd.name}): {exc}. Expected {sd.signature()}") from exc
        for k in through:
            params.pop(k)
        bad = [w for w in _words(cut.get("side")) if w not in SIDES]
        if bad:
            raise ToolArgumentError(f"{label}: side must use the words {sorted(SIDES)}, e.g. 'top' or 'top front'")
        pattern = cut.get("pattern")
        if pattern:
            # Seen with Qwen3-8B: the start position put inside the pattern, where it was dropped.
            if "position" in pattern and cut.get("position") is None:
                cut = {**cut, "position": pattern["position"]}
            pattern = validate_arguments(patterns.SCHEMA, {k: v for k, v in pattern.items() if k != "position"})
            if pattern["type"] == "around":
                raise ToolArgumentError(f"{label}: cuts can repeat as linear, grid or radial patterns, not 'around'")
            try:
                total += len(patterns.copies(pattern))
            except ToolArgumentError as exc:
                raise ToolArgumentError(f"{label}: {exc}") from exc
        else:
            total += 1
        entry = {**cut, "shape": sd.name, "params": params, **({"pattern": pattern} if pattern else {})}
        if entry in out:
            # Seen with Qwen3-8B: a wall's door moved onto its window, leaving one opening.
            raise ToolArgumentError(f"{label} is the same as cuts[{out.index(entry)}]: give each cut its own "
                                    f"position (or size), or leave the copy out")
        out.append(entry)
    if total > MAX_CUT_COPIES:
        raise ToolArgumentError(f"the cuts make {total} copies; the limit is {MAX_CUT_COPIES}")
    return out


# ---------------------------------------------------------------------------
# meshes <-> manifolds
# ---------------------------------------------------------------------------

def _manifold(mesh: trimesh.Trimesh, what: str) -> m3.Manifold:
    m = m3.Manifold(m3.Mesh(vert_properties=np.asarray(mesh.vertices, np.float32),
                            tri_verts=np.asarray(mesh.faces, np.uint32)))
    if m.status() != m3.Error.NoError:
        raise ToolArgumentError(f"{what} can't be used in a cut: its surface isn't closed")
    return m


def as_solid(mesh: trimesh.Trimesh, what: str) -> m3.Manifold:
    """A ready-made object is several overlapping closed parts: merge them into one solid."""
    whole = _manifold(mesh, what)
    parts = whole.decompose()
    return whole if len(parts) <= 1 else m3.Manifold.batch_boolean(parts, m3.OpType.Add)


def to_trimesh(m: m3.Manifold) -> trimesh.Trimesh:
    out = m.to_mesh()
    mesh = trimesh.Trimesh(np.asarray(out.vert_properties)[:, :3].astype(float),
                           np.asarray(out.tri_verts), process=True)
    if mesh.volume < 0:
        mesh.invert()
    return mesh


# ---------------------------------------------------------------------------
# applying
# ---------------------------------------------------------------------------

def _cutter(cut: dict[str, Any], through_len: float) -> trimesh.Trimesh:
    """The cut's own shape, centred on the origin and turned by its rotation."""
    sd = get_shape(cut["shape"])
    assert sd is not None and sd.builder is not None
    params = dict(cut["params"])
    for k in THROUGH.get(sd.name, ()):
        params.setdefault(k, through_len)
    mesh = sd.builder(params).copy()
    lo, hi = mesh.bounds
    mesh.apply_translation(-(lo + hi) / 2)
    return mesh.apply_transform(rotation_deg(cut.get("rotation")))


def _push_flush_faces(verts: np.ndarray, blo: np.ndarray, bhi: np.ndarray, tol: float, margin: float) -> None:
    """Move cutter faces lying on the object's outer bounds outward (in place)."""
    clo, chi = verts.min(axis=0), verts.max(axis=0)
    for a in range(3):
        if abs(clo[a] - blo[a]) < tol:
            verts[verts[:, a] <= clo[a] + tol, a] -= margin
        if abs(chi[a] - bhi[a]) < tol:
            verts[verts[:, a] >= chi[a] - tol, a] += margin


def _breaks_edge(verts: np.ndarray, blo: np.ndarray, bhi: np.ndarray, tol: float,
                 sided: set[int]) -> tuple[int, str, float] | None:
    """
    (axis, side word, edge) where the cutter sticks out of the object on one side only: a hole
    meant for the middle that was measured from a corner (seen with Qwen3-8B: a 30 cm plate's
    "hole in the middle" at [0.15, 0, 0.15]). Going right through (out both sides) is fine, and
    so is any side the cut names.
    """
    clo, chi = verts.min(axis=0), verts.max(axis=0)
    for a in range(3):
        if a in sided:
            continue
        out_lo = clo[a] < blo[a] - tol and chi[a] > blo[a] + tol
        out_hi = chi[a] > bhi[a] + tol and clo[a] < bhi[a] - tol
        if out_lo != out_hi:
            word = next(w for w, (axis, sign) in SIDES.items() if axis == a and (sign > 0) == out_hi)
            return a, word, float(bhi[a] if out_hi else blo[a])
    return None


def _outside(centre: np.ndarray, blo: np.ndarray, bhi: np.ndarray, positioned: list[tuple[int, list[float]]]) -> str:
    """
    Where a missed cut is and the number that likely fixes it. Seen with Qwen3-8B: a 4 m wall's
    window at x = 2.5, measured from the wall's left end instead of its middle. `positioned`:
    (index, position) of every cut with a position, so the fix names each one's new number (with
    one number for "this one", it changed the wrong cut).
    """
    for a, name, end in ((0, "x", "left end"), (2, "z", "back")):
        if blo[a] - 1e-9 <= centre[a] <= bhi[a] + 1e-9:
            continue
        half = (bhi[a] - blo[a]) / 2
        text = (f"{name} = {centre[a]:.3g} is outside it; it runs from {name} = {blo[a]:.3g} to {bhi[a]:.3g} "
                f"({name} = 0 is its middle). ")
        shifted = centre[a] - half if centre[a] > bhi[a] else centre[a] + half
        if blo[a] <= shifted <= bhi[a]:
            sign = "subtract" if centre[a] > bhi[a] else "add"
            word = "from" if sign == "subtract" else "to"
            delta = -half if sign == "subtract" else half
            each = ", ".join(f"cuts[{j}] {name} = {pos[a] + delta:.3g}" for j, pos in positioned)
            text += (f"If you measured from its {end if sign == 'subtract' else 'other end'}, {sign} {half:.3g} "
                     f"{word} every cut's {name}: {each}.")
        else:
            text += f"Use an {name} between {blo[a]:.3g} and {bhi[a]:.3g}."
        return text
    return (f"the cut is centred at {_fmt(centre)}, but the object spans {_fmt(blo)} to {_fmt(bhi)} "
            f"(cut positions are from its bottom-centre [0, 0, 0]). Move the cut inside it.")


def _report(i: int, cut: dict[str, Any], centre: np.ndarray, placed: list[np.ndarray], missed: list[np.ndarray],
            opened: tuple[int, str, float, float] | None, blo: np.ndarray, bhi: np.ndarray,
            round_body: bool, positioned: list[tuple[int, list[float]]]) -> list[str]:
    """Notes for one cut entry whose copies miss the object or break open through its sides."""
    label = f"cuts[{i}] ({cut['shape']})"
    if len(placed) > 1 and (missed or opened):
        # Seen with Qwen3-8B: 3 peg holes as a linear pattern from the board's middle, so the row
        # ran off its end. A linear pattern starts at the position; the fix is to start earlier.
        mid, middle = np.mean(placed, axis=0), (blo + bhi) / 2
        start = centre.copy()
        start[[0, 2]] -= (mid - middle)[[0, 2]]
        shifted = np.array(placed) + (start - centre)
        off_centre = float(np.abs((mid - middle)[[0, 2]]).max()) > 1e-6 + 0.01 * float(np.max(bhi - blo))
        if not off_centre and opened and opened[0] != 1:
            # Centred already, just spread too wide (seen with Qwen3-8B: bolt holes 0.1 apart across a
            # 0.1 m flange, so each row sat on an edge): name the spacing that fits.
            axis, word, edge, _ = opened
            name = "xyz"[axis]
            p = cut["params"]
            half = float(p.get("radius") or max(p.get("width", 0), p.get("depth", 0)) / 2)
            levels = len(np.unique(np.round(np.array(placed)[:, axis], 6)))
            room = float(bhi[axis] - blo[axis]) - 2.4 * half
            fit = (f"spacing along {name} of at most {room / (levels - 1):.3g}" if levels > 1 and room > 0
                   else f"fewer copies along {name}")
            return [f"the copies of {label} are spread too wide along {name}: the outer ones break through the "
                    f"object's {word} face ({name} = {edge:.3g}). Use a {fit}."]
        if off_centre and np.all(shifted[:, [0, 2]] >= blo[[0, 2]]) and np.all(shifted[:, [0, 2]] <= bhi[[0, 2]]):
            return [f"the copies of {label} run off the object (they are centred at {_fmt(mid)}, the object's "
                    f"middle is {_fmt(middle)}). A linear pattern starts at the cut's position: to centre them use "
                    f"\"position\": {_fmt(start)}."]
    notes = []
    if opened:
        axis, word, edge, at = opened
        name, mid_a = "xyz"[axis], (blo[axis] + bhi[axis]) / 2
        which = f"a copy of {label}" if len(placed) > 1 else label
        # The fix first: Qwen3-8B took the side option from this message for a hole put at y = 0.
        notes.append(
            f"{which} breaks open through the object's {word} face ({name} = {edge:.3g}) because its centre is at "
            f"{name} = {at:.3g}. For a hole inside the object use {name} = {mid_a:.3g} in its position (positions "
            f"are from the bottom-centre: x = z = 0 is the middle, not a corner; y is the height). Only for a notch "
            f"or a door that opens onto that face, add \"side\": \"{word}\" instead."
        )
    if missed and len(placed) == 1:
        notes.append(f"{label} misses the object: " + _outside(centre, blo, bhi, positioned))
    elif missed:
        where = ", ".join(_fmt(p) for p in missed[:3]) + (", ..." if len(missed) > 3 else "")
        fix = "Make the pattern smaller so every copy is inside."
        if round_body and cut["pattern"]["type"] != "radial":
            ring = round(0.7 * (bhi[0] - blo[0]) / 2, 3)
            fix = (f"For holes in a ring use \"pattern\": {{\"type\": \"radial\", \"count\": {len(placed)}, "
                   f"\"radius\": {ring:g}}}.")
        notes.append(f"{len(missed)} of the {len(placed)} copies of {label} miss the object: they land at {where}, "
                     f"but the object spans {_fmt(blo)} to {_fmt(bhi)}. {fix}")
    return notes


def _fmt(v: np.ndarray) -> str:
    return "[" + ", ".join(f"{x:.3g}" for x in v) + "]"


def apply(body: trimesh.Trimesh, cuts: list[dict[str, Any]],
          round_body: bool = False) -> tuple[trimesh.Trimesh, list[str]]:
    """
    Remove the cuts from `body` (bottom-centre at the origin). Returns the new mesh and notes
    for the model (cuts that miss, pieces that fell apart). The object keeps its frame: cutting
    away its bottom doesn't move what is left. `round_body`: a round footprint, where repeated
    holes almost always mean a ring (a flange's bolt holes).
    """
    blo, bhi = body.bounds
    diag = float(np.linalg.norm(bhi - blo))
    tol, margin = max(1e-6, 1e-4 * diag), max(1e-5, 2e-3 * diag)
    solid = as_solid(body, "The object")
    volume = solid.volume()
    pieces_before = len(solid.decompose())

    notes: list[str] = []
    all_cutters: list[m3.Manifold] = []
    positioned = [(j, vec3(c["position"])) for j, c in enumerate(cuts) if c.get("position") is not None]
    for i, cut in enumerate(cuts):
        mesh = _cutter(cut, 2 * diag + 2 * margin)
        clo, chi = mesh.bounds
        centre = (blo + bhi) / 2 if cut.get("position") is None else np.array(vec3(cut["position"]))
        for word in _words(cut.get("side")):
            axis, sign = SIDES[word]
            centre[axis] = (bhi[axis] - chi[axis]) if sign > 0 else (blo[axis] - clo[axis])
        copies = patterns.copies(cut.get("pattern"))
        sided = {SIDES[w][0] for w in _words(cut.get("side"))}
        missed: list[np.ndarray] = []
        placed: list[np.ndarray] = []
        opened = None
        for c in copies:
            place = translation(centre) @ c
            placed.append(place[:3, 3])
            verts = transform_points(place, np.asarray(mesh.vertices))
            edge = _breaks_edge(verts, blo, bhi, tol, sided)
            _push_flush_faces(verts, blo, bhi, tol, margin)
            cutter = _manifold(trimesh.Trimesh(verts, mesh.faces, process=False), f"cuts[{i}] ({cut['shape']})")
            if (solid ^ cutter).volume() <= 1e-12:
                missed.append(place[:3, 3])
            elif opened is None and edge:
                opened = (*edge, float(place[edge[0], 3]))
            all_cutters.append(cutter)
        notes.extend(_report(i, cut, centre, placed, missed, opened, blo, bhi, round_body, positioned))

    result = solid - m3.Manifold.batch_boolean(all_cutters, m3.OpType.Add) if all_cutters else solid
    if result.is_empty() or result.volume() < 1e-9 * max(volume, 1e-12):
        raise ToolArgumentError("the cuts remove the whole object. Make them smaller or move them to its edge")
    pieces = len(result.decompose())
    if pieces > pieces_before:
        notes.append(f"the cuts split it into {pieces} separate pieces. If it should stay in one piece, "
                     f"make the cuts shorter or narrower.")
    return to_trimesh(result), notes
