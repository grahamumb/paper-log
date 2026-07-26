"""Read the codes back off a rendered PDF.

Generating a symbol and *decoding* one are different problems, and only the
second tells you a page will file itself correctly. This module rasterises a
finished PDF and runs a real QR decoder over it, so a run can be checked before
it goes anywhere near a printer.

Decoders are less reliable than the symbols they read. OpenCV's bundled
detector -- the one used here -- will miss a symbol on a large sheet that it
reads perfectly from a crop, and miss one at 600dpi that it reads at 300. Those
are decoder limits, not defects in the page, so a page that comes up short is
retried three ways before it is called a failure:

1. the whole rendered page, multi-symbol -- what a naive scanner does;
2. overlapping tiles, single-symbol -- what a careful one does;
3. the exact region the manifest says the code occupies, resampled to a whole
   number of pixels per module;
4. failing all of those, the printed modules are compared bit for bit against
   the symbol the expected payload encodes.

Steps 3 and 4 use the manifest as a map, which makes them stricter than they
sound rather than looser: the symbol still has to be *the one that belongs
there*, so a code that is wrong, misplaced or missing still fails. Step 4 in
particular is decoder-independent, which matters because OpenCV's failures are
not about image quality -- it reads a pristine symbol at one scale and fails on
the identical symbol at another.

All of the retries switch off below `MIN_PX_PER_MODULE`, so a code too small to
survive real paper is never rescued by resampling.

The optional dependencies live behind :func:`load_backend`; nothing else in
paper-log needs a PDF rasteriser or a computer vision library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .ids import PageRef, TokenError, decode
from .imposition import booklet_sheets
from .qrcodes import build_matrix

#: 13mm codes need roughly this much to decode reliably; see the README.
DEFAULT_DPI = 300

#: Pixels per module used when normalising a crop before the last-resort retry.
NORMALISED_MODULE_PX = 8

#: Below this many pixels per module in the *original* raster, the retry passes
#: are switched off. Resampling a geometrically perfect render can recover a
#: symbol from barely two pixels per module, but paper cannot: ink spreads, the
#: scanner's optics blur, and the fibre shows through. Recovering such a code
#: here would report a page as scannable that would fail in the real world.
MIN_PX_PER_MODULE = 3.0


class BackendMissing(RuntimeError):
    """Raised when the verification extras are not installed."""


def load_backend():
    """Import the rasteriser and decoder, or explain how to get them."""
    try:
        import cv2
        import numpy
        import pypdfium2
    except ImportError as exc:
        raise BackendMissing(
            f"verification needs the optional extras ({exc.name} is missing). "
            "Install them with: pip install 'paper-log[verify]'"
        ) from None
    return cv2, numpy, pypdfium2


@dataclass(frozen=True)
class ExpectedCode:
    """A code the manifest says should be at a given spot on a given sheet."""

    payload: str
    corner: str
    page: int
    #: Millimetres from the bottom-left of the *sheet*, not the page.
    x_mm: float
    y_mm: float
    size_mm: float


@dataclass
class PageScan:
    """What the decoder found on one rendered page."""

    index: int  # 0-based page index within the PDF
    payloads: List[str] = field(default_factory=list)
    refs: List[PageRef] = field(default_factory=list)
    undecodable: List[str] = field(default_factory=list)
    #: Which retry passes this page needed.
    tiled: bool = False
    targeted: bool = False
    #: Codes no decoder here could read, but which are provably the right
    #: symbol, confirmed module by module against the expected payload.
    bit_verified: List[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.payloads)

    def add(self, payloads: Sequence[str]) -> int:
        added = 0
        for payload in payloads:
            if not payload or payload in self.payloads:
                continue
            self.payloads.append(payload)
            added += 1
            try:
                self.refs.append(decode(payload))
            except TokenError:
                self.undecodable.append(payload)
        return added


# --------------------------------------------------------------------------
# decoding passes
# --------------------------------------------------------------------------


def _decode_multi(detector, image) -> List[str]:
    found, texts, _, _ = detector.detectAndDecodeMulti(image)
    return [text for text in (texts or []) if text] if found else []


def _decode_one(detector, image) -> List[str]:
    try:
        text, _, _ = detector.detectAndDecode(image)
    except Exception:  # a decoder failure is a miss, not a crash
        return []
    return [text] if text else []


def _decode_tiled(detector, image, grid: int = 4, overlap: float = 0.35) -> List[str]:
    """Decode overlapping tiles.

    The overlap matters: a symbol straddling a tile boundary is invisible to
    both tiles, and codes in page corners sit exactly where a naive grid cuts.
    """
    height, width = image.shape[:2]
    tile_h = int(height / grid * (1 + overlap))
    tile_w = int(width / grid * (1 + overlap))
    step_h = max(int(height / grid), 1)
    step_w = max(int(width / grid), 1)

    payloads: List[str] = []
    for row in range(grid):
        for col in range(grid):
            top = min(row * step_h, max(height - tile_h, 0))
            left = min(col * step_w, max(width - tile_w, 0))
            tile = image[top : top + tile_h, left : left + tile_w]
            if tile.size == 0:
                continue
            for payload in _decode_multi(detector, tile) + _decode_one(detector, tile):
                if payload not in payloads:
                    payloads.append(payload)
    return payloads


def _decode_region(cv2, detector, image, code: ExpectedCode, dpi: int, modules: int) -> List[str]:
    """Decode the exact area a code should occupy, with a normalising retry."""
    scale = dpi / 25.4
    height = image.shape[0]
    size = max(round(code.size_mm * scale), 1)
    left = round(code.x_mm * scale)
    # Manifest y is measured up from the bottom; image rows run down from the top.
    top = round(height - (code.y_mm + code.size_mm) * scale)

    # Crop exactly the declared footprint: it already includes the quiet zone,
    # and widening it drags in ruling or a page edge that only confuses the
    # detector further.
    crop = image[max(top, 0) : top + size, max(left, 0) : left + size]
    if crop.size == 0 or crop.shape[0] < modules or crop.shape[1] < modules:
        return []

    payloads = _decode_one(detector, crop)
    if payloads:
        return payloads

    if size / modules < MIN_PX_PER_MODULE:
        # Too few pixels per module for the retries to mean anything on paper.
        return []

    # Resample so every module lands on a whole number of pixels. A fractional
    # module pitch is what trips the detector on an otherwise perfect symbol.
    side = modules * NORMALISED_MODULE_PX
    resized = cv2.resize(crop, (side, side), interpolation=cv2.INTER_AREA)
    _, binarised = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    payloads = _decode_one(detector, binarised)
    if payloads:
        return payloads

    # Last try: a generous synthetic quiet zone around the normalised symbol.
    margin = 4 * NORMALISED_MODULE_PX
    bordered = cv2.copyMakeBorder(
        binarised, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=255
    )
    return _decode_one(detector, bordered)


def _read_modules(numpy, image, code: ExpectedCode, dpi: int, modules: int):
    """Sample the printed symbol back into a module matrix, or ``None``.

    Reads the middle half of each module cell, which ignores the anti-aliased
    edges where a fractional module pitch puts them.
    """
    scale = dpi / 25.4
    height = image.shape[0]
    size = max(round(code.size_mm * scale), 1)
    left = round(code.x_mm * scale)
    top = round(height - (code.y_mm + code.size_mm) * scale)
    crop = image[max(top, 0) : top + size, max(left, 0) : left + size]
    if crop.shape[0] < modules or crop.shape[1] < modules:
        return None
    if min(crop.shape[:2]) / modules < MIN_PX_PER_MODULE:
        return None

    rows, columns = crop.shape[:2]
    matrix = numpy.zeros((modules, modules), dtype=int)
    for row in range(modules):
        top_edge, bottom_edge = row * rows / modules, (row + 1) * rows / modules
        y0 = int(top_edge + (bottom_edge - top_edge) * 0.25)
        y1 = max(int(top_edge + (bottom_edge - top_edge) * 0.75), y0 + 1)
        for column in range(modules):
            left_edge, right_edge = column * columns / modules, (column + 1) * columns / modules
            x0 = int(left_edge + (right_edge - left_edge) * 0.25)
            x1 = max(int(left_edge + (right_edge - left_edge) * 0.75), x0 + 1)
            matrix[row, column] = 1 if crop[y0:y1, x0:x1].mean() < 128 else 0
    return matrix


def _bits_match(
    numpy, image, code: ExpectedCode, dpi: int, modules: int, quiet_zone: int, ecc: str
) -> bool:
    """Is the symbol on the page bit-for-bit the one this payload encodes?

    The decoder is the weak link, not the printing: OpenCV reads a pristine
    symbol at one scale and fails on the same symbol at another. So when every
    decode attempt has failed, compare the printed modules against the symbol
    the payload *should* produce. That is decoder-independent, and it cannot
    pass a code that is wrong, misplaced or missing -- the bits would differ.
    """
    printed = _read_modules(numpy, image, code, dpi, modules)
    if printed is None:
        return False
    try:
        truth = build_matrix(code.payload, ecc)
    except Exception:
        return False
    if len(truth) + 2 * quiet_zone != modules:
        return False

    expected = numpy.zeros((modules, modules), dtype=int)
    expected[quiet_zone : modules - quiet_zone, quiet_zone : modules - quiet_zone] = numpy.array(truth)
    return bool((printed == expected).all())


# --------------------------------------------------------------------------
# expectations from the manifest
# --------------------------------------------------------------------------


def expected_layout(manifest: Dict) -> List[List[ExpectedCode]]:
    """What each *PDF* page should carry, and where, given the imposition."""
    pages = manifest.get("pages", [])
    qr = manifest.get("qr", {})
    if not pages or not qr.get("enabled"):
        return []

    geometry = qr.get("geometry", {})
    by_number = {entry["page"]: entry for entry in pages}

    def codes_for(number: int, x_offset: float) -> List[ExpectedCode]:
        entry = by_number.get(number)
        if entry is None:
            return []
        out = []
        for corner, payload in (entry.get("codes") or {}).items():
            box = geometry.get(corner)
            if box is None:
                continue
            out.append(
                ExpectedCode(
                    payload=payload,
                    corner=corner,
                    page=number,
                    x_mm=box["x_mm"] + x_offset,
                    y_mm=box["y_mm"],
                    size_mm=box["size_mm"],
                )
            )
        return out

    if manifest.get("imposition") == "booklet":
        page_width = manifest.get("page", {}).get("width_mm", 0.0)
        layout = []
        for left, right in booklet_sheets(len(pages)):
            sheet: List[ExpectedCode] = []
            if left is not None:
                sheet.extend(codes_for(left, 0.0))
            if right is not None:
                sheet.extend(codes_for(right, page_width))
            layout.append(sheet)
        return layout

    return [codes_for(entry["page"], 0.0) for entry in pages]


# --------------------------------------------------------------------------
# the scan
# --------------------------------------------------------------------------


@dataclass
class VerifyReport:
    scans: List[PageScan]
    expected: int
    decoded: int
    missing: List[str] = field(default_factory=list)
    unexpected: List[str] = field(default_factory=list)
    short_pages: List[int] = field(default_factory=list)  # 1-based
    complete: bool = False  # was the whole document scanned?

    @property
    def ok(self) -> bool:
        return not self.missing and not self.unexpected and not self.short_pages and self.decoded > 0

    @property
    def pages_scanned(self) -> int:
        return len(self.scans)

    @property
    def tiled_pages(self) -> int:
        return sum(1 for scan in self.scans if scan.tiled)

    @property
    def targeted_pages(self) -> int:
        return sum(1 for scan in self.scans if scan.targeted)

    @property
    def bit_verified_codes(self) -> int:
        return sum(len(scan.bit_verified) for scan in self.scans)


@dataclass
class ScanRun:
    """The outcome of scanning a document, and how much of it was looked at."""

    scans: List[PageScan]
    total_pages: int

    @property
    def complete(self) -> bool:
        return len(self.scans) >= self.total_pages


def scan_pdf(
    path: Path,
    *,
    dpi: int = DEFAULT_DPI,
    limit: Optional[int] = None,
    layout: Optional[Sequence[Sequence[ExpectedCode]]] = None,
    modules: int = 29,
    quiet_zone: int = 2,
    ecc: str = "m",
) -> ScanRun:
    """Rasterise ``path`` and decode the QR codes on each page.

    ``layout`` (from :func:`expected_layout`) tells the scanner how many codes
    each page should have and where they are, enabling the retry passes.
    """
    cv2, numpy, pypdfium2 = load_backend()

    document = pypdfium2.PdfDocument(str(path))
    try:
        total = len(document)
        count = total if limit is None else min(limit, total)
        detector = cv2.QRCodeDetector()

        results: List[PageScan] = []
        for index in range(count):
            bitmap = document[index].render(scale=dpi / 72)
            image = numpy.asarray(bitmap.to_pil().convert("L"))
            scan = PageScan(index=index)
            scan.add(_decode_multi(detector, image))

            wanted = list(layout[index]) if layout and index < len(layout) else []
            if wanted and scan.count < len(wanted):
                scan.tiled = True
                scan.add(_decode_tiled(detector, image))

            if wanted and scan.count < len(wanted):
                for code in wanted:
                    if code.payload in scan.payloads:
                        continue
                    found = _decode_region(cv2, detector, image, code, dpi, modules)
                    if scan.add(found):
                        scan.targeted = True
                        continue
                    if _bits_match(numpy, image, code, dpi, modules, quiet_zone, ecc):
                        scan.add([code.payload])
                        scan.bit_verified.append(code.payload)

            results.append(scan)
        return ScanRun(scans=results, total_pages=total)
    finally:
        document.close()


def verify(
    pdf_path: Path,
    manifest: Optional[Dict] = None,
    *,
    dpi: int = DEFAULT_DPI,
    limit: Optional[int] = None,
) -> VerifyReport:
    """Scan ``pdf_path`` and, given its manifest, check the codes match.

    When only part of the document is sampled the report checks that everything
    decoded is a token the manifest knows about; a full scan additionally checks
    that no expected token is missing.
    """
    layout = expected_layout(manifest) if manifest else []
    qr_meta = (manifest or {}).get("qr", {})
    run = scan_pdf(
        pdf_path,
        dpi=dpi,
        limit=limit,
        layout=layout,
        modules=qr_meta.get("modules", 29),
        quiet_zone=qr_meta.get("quiet_zone_modules", 2),
        ecc=str(qr_meta.get("error_correction", "M")).lower(),
    )
    scans = run.scans
    report = VerifyReport(
        scans=scans,
        expected=0,
        decoded=sum(scan.count for scan in scans),
        complete=run.complete,
    )
    if manifest is None:
        return report

    expected_tokens = {code.payload for sheet in layout for code in sheet}
    report.expected = len(expected_tokens)

    seen = {payload for scan in scans for payload in scan.payloads}
    report.unexpected = sorted(seen - expected_tokens)
    report.short_pages = [
        scan.index + 1
        for scan in scans
        if scan.index < len(layout) and scan.count < len(layout[scan.index])
    ]
    if report.complete:
        # Only a full scan can prove that nothing is missing.
        report.missing = sorted(expected_tokens - seen)
    return report
