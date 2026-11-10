"""
Row 6: standard sizes, so the model names a part ("M10", "DN50", 20 teeth module 2) and the
engine knows every dimension. Values in millimetres as the standards give them; the shape
builders convert to metres.

  ISO 261/262   metric coarse thread pitch
  ISO 4014/4017 hex bolt head (across flats s, height k); ISO 4032 nut height m
  ISO 4762      socket head cap screw (head diameter, key size); ISO 10642 countersunk
  ISO 7089      plain washer; ISO 273 clearance holes (medium)
  EN 1092-1     PN16 plate flanges (type 01)

A size the model writes loosely ("m10", "M10x1.25", 10, 0.01, "10mm") is read here; a size
not in the table is an error naming the nearest ones.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from oprel.integrations.scene3d.engine.registry import ToolArgumentError


@dataclass(frozen=True)
class Metric:
    d: float              # nominal diameter
    pitch: float          # coarse pitch
    s: float              # hex across flats (bolt head and nut)
    k: float              # hex head height
    m: float              # nut height
    washer: tuple[float, float, float]  # inner d1, outer d2, thickness h
    clearance: float      # medium clearance hole
    socket: tuple[float, float]         # socket head diameter dk, hex key size

    @property
    def name(self) -> str:
        return f"M{self.d:g}"

    @property
    def tap_drill(self) -> float:
        return round(self.d - self.pitch, 2)


_METRIC_ROWS = [
    # d, pitch, s, k, m, washer (d1, d2, h), clearance, socket (dk, key)
    (2, 0.4, 4, 1.4, 1.6, (2.2, 5, 0.3), 2.4, (3.8, 1.5)),
    (2.5, 0.45, 5, 1.7, 2.0, (2.7, 6, 0.5), 2.9, (4.5, 2)),
    (3, 0.5, 5.5, 2, 2.4, (3.2, 7, 0.5), 3.4, (5.5, 2.5)),
    (4, 0.7, 7, 2.8, 3.2, (4.3, 9, 0.8), 4.5, (7, 3)),
    (5, 0.8, 8, 3.5, 4.7, (5.3, 10, 1), 5.5, (8.5, 4)),
    (6, 1.0, 10, 4, 5.2, (6.4, 12, 1.6), 6.6, (10, 5)),
    (8, 1.25, 13, 5.3, 6.8, (8.4, 16, 1.6), 9, (13, 6)),
    (10, 1.5, 16, 6.4, 8.4, (10.5, 20, 2), 11, (16, 8)),
    (12, 1.75, 18, 7.5, 10.8, (13, 24, 2.5), 13.5, (18, 10)),
    (14, 2, 21, 8.8, 12.8, (15, 28, 2.5), 15.5, (21, 12)),
    (16, 2, 24, 10, 14.8, (17, 30, 3), 17.5, (24, 14)),
    (20, 2.5, 30, 12.5, 18, (21, 37, 3), 22, (30, 17)),
    (24, 3, 36, 15, 21.5, (25, 44, 4), 26, (36, 19)),
    (30, 3.5, 46, 18.7, 25.6, (31, 56, 4), 33, (45, 22)),
    (36, 4, 55, 22.5, 31, (37, 66, 5), 39, (54, 27)),
]
METRIC = {row[0]: Metric(*row) for row in _METRIC_ROWS}
METRIC_NAMES = ", ".join(m.name for m in METRIC.values())

_METRIC_WORD = re.compile(r"^\s*m?\s*(\d+(?:\.\d+)?)\s*(?:mm)?\s*(?:[x×]\s*(\d+(?:\.\d+)?))?\s*$", re.IGNORECASE)


def metric(size: object) -> tuple[Metric, float]:
    """
    (table row, pitch) for "M10", "m10", "M10x1.25" (fine pitch), 10 (mm) or 0.01 (m).
    Raises with the nearest sizes when it isn't a standard one.
    """
    d = pitch = None
    if isinstance(size, (int, float)) and not isinstance(size, bool):
        d = float(size) * (1000 if size < 0.1 else 1)  # 0.01 is metres, 10 is millimetres
    elif isinstance(size, str):
        hit = _METRIC_WORD.match(size)
        if hit:
            d = float(hit.group(1))
            pitch = float(hit.group(2)) if hit.group(2) else None
    if d is None:
        raise ToolArgumentError(f"'size' must be a metric size like \"M10\" (one of {METRIC_NAMES})")
    row = METRIC.get(round(d, 1)) or METRIC.get(int(d) if d == int(d) else d)
    if row is None:
        near = sorted(METRIC, key=lambda k: abs(k - d))[:2]
        raise ToolArgumentError(f"M{d:g} is not a standard ISO size; use {' or '.join(f'M{n:g}' for n in sorted(near))}")
    if pitch is not None and not 0.2 <= pitch <= row.pitch:
        raise ToolArgumentError(f"pitch {pitch:g} mm doesn't fit {row.name} (coarse {row.pitch:g}, fine pitches are smaller)")
    return row, pitch or row.pitch


def thread_length(d: float, length: float) -> float:
    """ISO 4014 thread length b for a bolt of `length` (mm); the rest is plain shank."""
    b = 2 * d + (6 if length <= 125 else 12 if length <= 200 else 25)
    return length if length - b < d else b  # short bolts are threaded all the way (ISO 4017)


@dataclass(frozen=True)
class Flange:
    dn: int
    pipe_od: float        # outside diameter of the pipe it fits
    outer: float          # flange diameter D
    bolt_circle: float    # K
    holes: int
    hole: float           # bolt hole diameter L
    bolt: str
    thickness: float      # C (type 01 plate flange)


_FLANGE_ROWS = [
    # DN, pipe OD, D, K, holes, L, bolt, C   (EN 1092-1 PN16)
    (15, 21.3, 95, 65, 4, 14, "M12", 14), (20, 26.9, 105, 75, 4, 14, "M12", 16),
    (25, 33.7, 115, 85, 4, 14, "M12", 16), (32, 42.4, 140, 100, 4, 18, "M16", 18),
    (40, 48.3, 150, 110, 4, 18, "M16", 18), (50, 60.3, 165, 125, 4, 18, "M16", 20),
    (65, 76.1, 185, 145, 8, 18, "M16", 20), (80, 88.9, 200, 160, 8, 18, "M16", 20),
    (100, 114.3, 220, 180, 8, 18, "M16", 22), (125, 139.7, 250, 210, 8, 18, "M16", 22),
    (150, 168.3, 285, 240, 8, 22, "M20", 24), (200, 219.1, 340, 295, 12, 22, "M20", 24),
    (250, 273.0, 405, 355, 12, 26, "M24", 26), (300, 323.9, 460, 410, 12, 26, "M24", 28),
]
FLANGES = {row[0]: Flange(*row) for row in _FLANGE_ROWS}
DN_NAMES = ", ".join(f"DN{n}" for n in FLANGES)
# Pipe size in inches (NPS) -> DN
_INCH_TO_DN = {0.5: 15, 0.75: 20, 1: 25, 1.25: 32, 1.5: 40, 2: 50, 2.5: 65, 3: 80, 4: 100, 5: 125,
               6: 150, 8: 200, 10: 250, 12: 300}
_DN_WORD = re.compile(r"^\s*(dn|nps)?\s*(\d+(?:\.\d+)?)\s*(\"|in|inch|inches|mm)?\s*$", re.IGNORECASE)


def nominal(size: object) -> Flange:
    """The DN row for "DN50", 50, "2 inch" / "2in" / "NPS 2", or a pipe's outer diameter in metres (0.0603)."""
    dn = None
    if isinstance(size, (int, float)) and not isinstance(size, bool):
        if size < 1:  # metres: the pipe's outer diameter, or the bore
            mm = size * 1000
            dn = min(FLANGES, key=lambda n: min(abs(FLANGES[n].pipe_od - mm), abs(n - mm)))
        else:
            dn = int(size)
    elif isinstance(size, str):
        hit = _DN_WORD.match(size)
        if hit:
            value, prefix, unit = float(hit.group(2)), (hit.group(1) or "").lower(), (hit.group(3) or "").lower()
            if prefix == "nps" or unit in ('"', "in", "inch", "inches"):
                dn = _INCH_TO_DN.get(value)
                if dn is None:
                    raise ToolArgumentError(f"{value:g} inch is not a standard pipe size; use one of "
                                            f"{', '.join(f'{k:g}' for k in _INCH_TO_DN)} inch")
            else:
                dn = int(value)
    if dn is None:
        raise ToolArgumentError(f"'size' must be a nominal pipe size like \"DN50\" (one of {DN_NAMES})")
    if dn not in FLANGES:
        near = sorted(FLANGES, key=lambda n: abs(n - dn))[:2]
        raise ToolArgumentError(f"DN{dn} is not in the table; use {' or '.join(f'DN{n}' for n in sorted(near))}")
    return FLANGES[dn]
