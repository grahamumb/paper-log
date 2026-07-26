"""Drawing the writing surface: lines, dots, grids.

Every ruling is anchored to the *top* of the writing rectangle and stops when
it runs out of room, so a page with a header and a page without one still have
their first line in a consistent place relative to whatever sits above it.
Horizontal grids and dot fields are centred in the leftover width so the
margins stay visually even.
"""

from __future__ import annotations

from typing import Tuple

from .config import Ruling

Rect = Tuple[float, float, float, float]  # x, y, width, height


def _horizontal_lines(canvas, rect: Rect, spacing: float, offset: float) -> None:
    x, y, width, height = rect
    path = canvas.beginPath()
    position = y + height - offset - spacing
    while position >= y - 1e-6:
        path.moveTo(x, position)
        path.lineTo(x + width, position)
        position -= spacing
    canvas.drawPath(path, stroke=1, fill=0)


def _vertical_lines(canvas, rect: Rect, spacing: float) -> None:
    x, y, width, height = rect
    count = int(width // spacing)
    inset = (width - count * spacing) / 2
    path = canvas.beginPath()
    for index in range(count + 1):
        position = x + inset + index * spacing
        path.moveTo(position, y)
        path.lineTo(position, y + height)
    canvas.drawPath(path, stroke=1, fill=0)


def _dots(canvas, rect: Rect, spacing: float, offset: float, radius: float) -> None:
    x, y, width, height = rect
    columns = int(width // spacing)
    inset = (width - columns * spacing) / 2
    path = canvas.beginPath()
    row_y = y + height - offset
    while row_y >= y - 1e-6:
        for index in range(columns + 1):
            dot_x = x + inset + index * spacing
            path.circle(dot_x, row_y, radius)
        row_y -= spacing
    canvas.drawPath(path, stroke=0, fill=1)


def draw(canvas, rect: Rect, ruling: Ruling) -> None:
    """Render ``ruling`` inside ``rect`` (the writing area, in points)."""
    x, y, width, height = rect
    if width <= 0 or height <= 0:
        return
    # "blank" still gets the margin rule below, it just has no body ruling.
    style = ruling.style

    canvas.saveState()
    canvas.setStrokeColorRGB(*ruling.color)
    canvas.setFillColorRGB(*ruling.color)
    canvas.setLineWidth(ruling.line_width)
    canvas.setLineCap(0)

    if style == "ruled":
        _horizontal_lines(canvas, rect, ruling.spacing, ruling.first_line_offset)
    elif style == "grid":
        _horizontal_lines(canvas, rect, ruling.spacing, ruling.first_line_offset)
        _vertical_lines(canvas, rect, ruling.spacing)
    elif style == "dotted":
        _dots(canvas, rect, ruling.spacing, ruling.first_line_offset, ruling.dot_radius)
    elif style == "cornell":
        _draw_cornell(canvas, rect, ruling)

    if ruling.margin_rule and style != "cornell":
        canvas.setStrokeColorRGB(*ruling.margin_rule_color)
        canvas.setLineWidth(ruling.line_width)
        rule_x = x + ruling.margin_rule_offset
        if x < rule_x < x + width:
            canvas.line(rule_x, y, rule_x, y + height)

    canvas.restoreState()


def _draw_cornell(canvas, rect: Rect, ruling: Ruling) -> None:
    """Cue column on the left, note area on the right, summary band below."""
    x, y, width, height = rect
    summary = min(ruling.summary_band, height * 0.4)
    cue = min(ruling.cue_column, width * 0.5)

    notes_rect = (x + cue, y + summary, width - cue, height - summary)
    _horizontal_lines(canvas, notes_rect, ruling.spacing, ruling.first_line_offset)

    canvas.saveState()
    canvas.setStrokeColorRGB(*ruling.margin_rule_color)
    canvas.setLineWidth(ruling.line_width * 1.6)
    canvas.line(x + cue, y + summary, x + cue, y + height)
    canvas.line(x, y + summary, x + width, y + summary)
    canvas.restoreState()
