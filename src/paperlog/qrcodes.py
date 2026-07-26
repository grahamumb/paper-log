"""QR symbol generation and vector drawing.

Symbols are drawn as filled rectangles straight into the PDF rather than as an
embedded raster. That keeps them resolution-independent -- they print as
crisply as the printer can manage -- and keeps the file small. Horizontally
adjacent dark modules are merged into single rectangles, which cuts the object
count of a typical symbol by roughly half.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Sequence, Tuple

import segno

from .ids import PageRef, render_payload


class QrError(RuntimeError):
    """Raised when a payload cannot be encoded."""


@lru_cache(maxsize=512)
def build_matrix(payload: str, error_correction: str = "m") -> Tuple[Tuple[int, ...], ...]:
    """Encode ``payload`` and return its module matrix, without a quiet zone."""
    try:
        symbol = segno.make(payload, error=error_correction, micro=False)
    except Exception as exc:  # segno raises a family of DataOverflow/Mode errors
        raise QrError(f"cannot encode QR payload {payload!r}: {exc}") from exc
    return tuple(tuple(row) for row in symbol.matrix)


def symbol_modules(payload: str, error_correction: str = "m", quiet_zone: int = 4) -> int:
    """Total width of the symbol in modules, including the quiet zone."""
    return len(build_matrix(payload, error_correction)) + 2 * quiet_zone


def worst_case_payload(config) -> str:
    """The longest payload this journal will produce.

    Page numbers grow a character at a time, so the last page is the dense one;
    sizing every symbol against it keeps all the codes on a run identical in
    module pitch.
    """
    ref = PageRef(
        notebook=config.notebook_id,
        page=max(config.pages, 1),
        side="F",
        corner=(config.qr.corners[0] if config.qr.corners else "TL"),
    )
    return render_payload(config.qr.payload, ref)


def module_size(config) -> float:
    """Printed size of one QR module, in points, for the densest page."""
    modules = symbol_modules(
        worst_case_payload(config), config.qr.error_correction, config.qr.quiet_zone
    )
    return config.qr.size / modules


def _runs(row: Sequence[int]) -> List[Tuple[int, int]]:
    """Collapse a matrix row into ``(start, length)`` runs of dark modules."""
    out: List[Tuple[int, int]] = []
    start = None
    for index, value in enumerate(row):
        if value and start is None:
            start = index
        elif not value and start is not None:
            out.append((start, index - start))
            start = None
    if start is not None:
        out.append((start, len(row) - start))
    return out


def draw_qr(
    canvas,
    x: float,
    y: float,
    size: float,
    payload: str,
    *,
    error_correction: str = "m",
    quiet_zone: int = 4,
    color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    fixed_modules: int = 0,
) -> float:
    """Draw a QR symbol with its bottom-left corner at ``(x, y)``.

    ``size`` covers the whole footprint including the quiet zone, so the
    caller can reason about clearance from the paper edge without knowing how
    dense the symbol turned out.

    ``fixed_modules`` pins the module pitch to a symbol of that many modules
    (see :func:`worst_case_payload`); the drawn symbol is then centred in the
    footprint. Pass 0 to fill the footprint.

    Returns the module size actually used.
    """
    matrix = build_matrix(payload, error_correction)
    modules = len(matrix) + 2 * quiet_zone
    pitch = size / max(fixed_modules, modules)
    drawn = modules * pitch
    origin_x = x + (size - drawn) / 2
    origin_y = y + (size - drawn) / 2

    canvas.saveState()
    canvas.setFillColorRGB(*color)
    canvas.setStrokeColorRGB(*color)
    path = canvas.beginPath()
    for row_index, row in enumerate(matrix):
        # PDF y grows upward; matrix row 0 is the top of the symbol.
        top = origin_y + drawn - (quiet_zone + row_index) * pitch
        for start, length in _runs(row):
            left = origin_x + (quiet_zone + start) * pitch
            path.rect(left, top - pitch, length * pitch, pitch)
    canvas.drawPath(path, stroke=0, fill=1)
    canvas.restoreState()
    return pitch
