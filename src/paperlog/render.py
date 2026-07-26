"""Page composition and PDF output.

:func:`draw_page` renders one journal page in page-local coordinates with its
origin at the bottom-left, which is what lets the booklet imposition place two
of them on one sheet with nothing more than a translate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas as pdfcanvas

from . import rulings
from .config import JournalConfig
from .ids import PageRef, render_payload
from .imposition import booklet_sheets
from .manifest import Page
from .qrcodes import draw_qr, symbol_modules, worst_case_payload
from .units import MM

Rect = Tuple[float, float, float, float]


def build_pages(config: JournalConfig) -> List[Page]:
    """The journal's pages in reading order."""
    pages: List[Page] = []
    for number in range(1, config.pages + 1):
        side = "F" if (not config.duplex or number % 2 == 1) else "B"
        ref = PageRef(
            notebook=config.notebook_id, page=number, side=side, corner="TL"
        )
        pages.append(Page(number=number, side=side, ref=ref))
    return pages


# --------------------------------------------------------------------------
# corner marks
# --------------------------------------------------------------------------


def corner_origins(config: JournalConfig, size: float, inset: float) -> Dict[str, Tuple[float, float]]:
    """Bottom-left origin of a ``size``-square box tucked into each corner."""
    return {
        corner: (box[0], box[1])
        for corner, box in config.corner_boxes(size, inset).items()
    }


def draw_corner_codes(canvas, config: JournalConfig, page: Page) -> None:
    if not config.qr.enabled or not config.qr.corners:
        return
    spec = config.qr
    origins = corner_origins(config, spec.size, spec.inset)
    # Pin every symbol on the run to the pitch of the densest page so the codes
    # look uniform and a scanner tuned to one page works on all of them.
    fixed = symbol_modules(worst_case_payload(config), spec.error_correction, spec.quiet_zone)

    for corner in spec.corners:
        x, y = origins[corner]
        payload = render_payload(spec.payload, page.ref.for_corner(corner), spec.token_format)
        draw_qr(
            canvas,
            x,
            y,
            spec.size,
            payload,
            error_correction=spec.error_correction,
            quiet_zone=spec.quiet_zone,
            color=spec.color,
            fixed_modules=fixed,
        )
        if spec.caption:
            _draw_caption(canvas, config, corner, x, y, spec.size, page)


#: A caption smaller than this is illegible, so it is dropped instead.
MIN_CAPTION_PT = 3.0


def _caption_width_limit(config: JournalConfig, corner: str, x: float, size: float, side: str) -> float:
    """How wide a corner caption may be before it runs into something.

    Never wider than the code it labels. A caption under a *top* corner also
    shares a line with the header, so it additionally has to stop at the edge
    of the text block.
    """
    limit = size
    if not corner.startswith("T"):
        return limit
    block_x, _, block_width, _ = config.text_block(side)
    gap = (block_x - x) if corner.endswith("L") else ((x + size) - (block_x + block_width))
    # Leave a millimetre so the caption never sits flush against the header.
    return min(limit, gap - 1 * MM) if gap > 0 else limit


def _draw_caption(canvas, config: JournalConfig, corner: str, x: float, y: float, size: float, page: Page) -> None:
    """A tiny human-readable label so a person can file a page by eye too."""
    spec = config.qr
    font = config.furniture.font
    limit = _caption_width_limit(config, corner, x, size, page.side)
    font_size = spec.caption_size

    # Prefer the full id, but a bare page number is far more useful than a
    # caption shrunk to the point of illegibility, or none at all.
    text = f"{page.ref.notebook}·{page.number}"
    width = pdfmetrics.stringWidth(text, font, font_size)
    if width > limit:
        text = str(page.number)
        width = pdfmetrics.stringWidth(text, font, font_size)
    if width > limit:
        font_size *= limit / width
    if font_size < MIN_CAPTION_PT:
        return

    canvas.saveState()
    canvas.setFont(font, font_size)
    canvas.setFillColorRGB(*config.furniture.color)
    gap = font_size * 0.4
    text_y = (y - font_size - gap) if corner.startswith("T") else (y + size + gap)
    if corner.endswith("L"):
        canvas.drawString(x, text_y, text)
    else:
        canvas.drawRightString(x + size, text_y, text)
    canvas.restoreState()


def draw_fiducials(canvas, config: JournalConfig, corners: List[str]) -> None:
    marks = config.fiducials
    if marks.style == "none" or not corners:
        return
    size = marks.size
    origins = corner_origins(config, size, marks.inset)

    canvas.saveState()
    canvas.setStrokeColorRGB(*marks.color)
    canvas.setFillColorRGB(*marks.color)
    canvas.setLineWidth(marks.thickness)
    canvas.setLineCap(0)

    for corner in corners:
        x, y = origins[corner]
        if marks.style == "square":
            canvas.rect(x, y, size, size, stroke=0, fill=1)
            continue
        if marks.style == "cross":
            mid_x, mid_y = x + size / 2, y + size / 2
            canvas.line(x, mid_y, x + size, mid_y)
            canvas.line(mid_x, y, mid_x, y + size)
            continue
        # bracket: an L hugging the outside of the corner
        corner_x = x if corner.endswith("L") else x + size
        corner_y = y + size if corner.startswith("T") else y
        horizontal = size if corner.endswith("L") else -size
        vertical = -size if corner.startswith("T") else size
        path = canvas.beginPath()
        path.moveTo(corner_x + horizontal, corner_y)
        path.lineTo(corner_x, corner_y)
        path.lineTo(corner_x, corner_y + vertical)
        canvas.drawPath(path, stroke=1, fill=0)

    canvas.restoreState()


# --------------------------------------------------------------------------
# furniture
# --------------------------------------------------------------------------


def header_height(config: JournalConfig) -> float:
    furniture = config.furniture
    if not (furniture.date_line or furniture.page_number or (furniture.title and furniture.title_text)):
        return 0.0
    return furniture.size * 2.6


def draw_furniture(canvas, config: JournalConfig, page: Page, block: Rect) -> None:
    furniture = config.furniture
    height = header_height(config)
    if height <= 0:
        return
    x, y, width, block_height = block
    baseline = y + block_height - furniture.size

    canvas.saveState()
    canvas.setFont(furniture.font, furniture.size)
    canvas.setFillColorRGB(*furniture.color)
    canvas.setStrokeColorRGB(*furniture.color)

    left_text = ""
    if furniture.title and furniture.title_text:
        left_text = furniture.title_text
    if furniture.date_line:
        left_text = f"{left_text}   {furniture.date_label}" if left_text else furniture.date_label
    if left_text:
        canvas.drawString(x, baseline, left_text)

    if furniture.page_number:
        label = furniture.page_number_format.format(
            page=page.number, notebook=page.ref.notebook, side=page.side
        )
        canvas.drawRightString(x + width, baseline, label)

    if furniture.header_rule:
        rule_y = y + block_height - height + furniture.size * 0.6
        canvas.setLineWidth(config.ruling.line_width)
        canvas.line(x, rule_y, x + width, rule_y)

    canvas.restoreState()


# --------------------------------------------------------------------------
# whole pages
# --------------------------------------------------------------------------


def draw_page(canvas, config: JournalConfig, page: Optional[Page]) -> None:
    """Draw one journal page with its origin at ``(0, 0)``.

    ``page`` of ``None`` is a padding leaf in a booklet -- deliberately left
    completely blank, since it is not part of the notebook and should not
    carry an identifier.
    """
    if page is None:
        return

    block = config.text_block(page.side)
    x, y, width, height = block
    writing = (x, y, width, max(height - header_height(config), 0.0))

    rulings.draw(canvas, writing, config.ruling)
    draw_furniture(canvas, config, page, block)
    draw_corner_codes(canvas, config, page)
    draw_fiducials(canvas, config, config.fiducials.resolve_corners(
        config.qr.corners if config.qr.enabled else []
    ))


def _draw_fold_guides(canvas, sheet_width: float, sheet_height: float) -> None:
    """A dashed spine line plus trim ticks, for booklet sheets."""
    middle = sheet_width / 2
    canvas.saveState()
    canvas.setStrokeColorRGB(0.75, 0.75, 0.75)
    canvas.setLineWidth(0.3)
    canvas.setDash(2, 4)
    canvas.line(middle, 4 * MM, middle, sheet_height - 4 * MM)
    canvas.restoreState()

    canvas.saveState()
    canvas.setStrokeColorRGB(0.4, 0.4, 0.4)
    canvas.setLineWidth(0.4)
    tick = 3 * MM
    for edge_y in (0.0, sheet_height):
        direction = 1 if edge_y == 0.0 else -1
        canvas.line(middle, edge_y, middle, edge_y + direction * tick)
    canvas.restoreState()


def render(config: JournalConfig, output: Path) -> List[Page]:
    """Write the journal PDF and return the pages that went into it."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pages = build_pages(config)
    page_width, page_height = config.size

    if config.imposition == "booklet":
        sheet_size = (page_width * 2, page_height)
    else:
        sheet_size = (page_width, page_height)

    canvas = pdfcanvas.Canvas(str(output), pagesize=sheet_size)
    canvas.setTitle(config.title or f"paper-log {config.notebook_id}")
    canvas.setSubject(f"paper-log notebook {config.notebook_id}")
    canvas.setCreator("paper-log")
    canvas.setKeywords([f"paperlog:{config.notebook_id}"])

    if config.imposition == "booklet":
        by_number = {page.number: page for page in pages}
        for left, right in booklet_sheets(len(pages)):
            for slot, number in ((0.0, left), (page_width, right)):
                if number is None:
                    continue
                canvas.saveState()
                canvas.translate(slot, 0)
                draw_page(canvas, config, by_number[number])
                canvas.restoreState()
            if config.crop_marks:
                _draw_fold_guides(canvas, sheet_size[0], sheet_size[1])
            canvas.showPage()
    else:
        for page in pages:
            draw_page(canvas, config, page)
            canvas.showPage()

    canvas.save()
    return pages
