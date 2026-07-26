"""Length parsing and conversion.

Everything inside the renderer is in PostScript points (1/72 inch), which is
what ReportLab's coordinate system uses. Config files may use any of the
supported suffixes; a bare number is interpreted as millimetres because that
is what people reach for when describing paper.
"""

from __future__ import annotations

import re

MM = 72.0 / 25.4
CM = 10 * MM
IN = 72.0
PT = 1.0

_UNITS = {"mm": MM, "cm": CM, "in": IN, '"': IN, "pt": PT}

_LENGTH_RE = re.compile(r'^([+-]?(?:\d+\.?\d*|\.\d+))\s*(mm|cm|in|pt|")?$')


class UnitError(ValueError):
    """Raised when a length cannot be parsed."""


def to_points(value, *, default_unit: str = "mm") -> float:
    """Convert a length to points.

    Accepts numbers (interpreted in ``default_unit``) and strings such as
    ``"12mm"``, ``"0.5in"``, ``"36pt"``, ``"1.5cm"``.
    """
    if isinstance(value, bool):  # bool is an int subclass; almost certainly a mistake
        raise UnitError(f"expected a length, got boolean {value!r}")
    if isinstance(value, (int, float)):
        return float(value) * _UNITS[default_unit]
    if not isinstance(value, str):
        raise UnitError(f"expected a length, got {type(value).__name__}: {value!r}")

    match = _LENGTH_RE.match(value.strip().lower())
    if not match:
        raise UnitError(
            f"cannot parse length {value!r}; use e.g. '12mm', '0.5in', '36pt'"
        )
    number, unit = match.groups()
    return float(number) * _UNITS[unit or default_unit]


def points_to(value: float, unit: str) -> float:
    """Convert points back to ``unit`` (for reporting and manifests)."""
    try:
        return value / _UNITS[unit]
    except KeyError:  # pragma: no cover - guarded by callers
        raise UnitError(f"unknown unit {unit!r}") from None


def format_mm(value: float, places: int = 2) -> float:
    """Round a point value to millimetres for human-facing output."""
    return round(points_to(value, "mm"), places)
