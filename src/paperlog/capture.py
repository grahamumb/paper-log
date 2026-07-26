"""Turn a photograph of a page into a flat, identified page image.

This is the read side of paper-log. You photograph a page with a phone --
hand-held, at an angle, under whatever light is going -- and this works out
which page it is and undoes the perspective, producing an image as square-on
as a flatbed scan.

How it works, and why each step is there:

**Finding the codes.** A whole-frame QR search fails on a page photograph:
the symbols are a few percent of the frame, and detectors downsample before
they look. Searching overlapping tiles at full resolution finds four codes
where a whole-frame search finds one. That single change is worth roughly a
1.5x reduction in the resolution you need.

**Flattening.** Every detected symbol yields four corner points, and OpenCV
returns them in the symbol's *own* frame -- index 0 is the symbol's top-left
however the photo is rotated. Since the token says which page corner the
symbol occupies, and the manifest says where that corner sits in millimetres,
each code contributes four point correspondences straight into a homography.
Four codes give sixteen points spread to the sheet's extremes, which is a
well-conditioned fit; one code gives four clustered points, which is a poor
one, and is reported as such.

**Orientation comes free.** The corner is part of the token, so a page
photographed upside down produces a homography that simply includes the
180-degree rotation. There is no separate "which way up" guess to get wrong.

Needs the optional extras: ``pip install 'paper-log[verify]'``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .ids import CORNERS, PageRef, TokenError, decode
from .scancheck import BackendMissing, load_backend

#: Output resolution that matches what a flatbed would give you.
DEFAULT_DPI = 300

#: Tiling used to hunt for codes. 3x3 with heavy overlap keeps every corner of
#: the page comfortably inside at least one tile.
TILE_GRID = 3
TILE_OVERLAP = 0.4

#: Above this reprojection error the fit is suspect -- usually a misdetected
#: code or a page bent enough that a flat homography cannot describe it.
RESIDUAL_WARN_MM = 1.5


class CaptureError(RuntimeError):
    """Raised when a photograph cannot be turned into a page."""


@dataclass(frozen=True)
class DetectedCode:
    """One QR symbol found in a photograph."""

    payload: str
    ref: PageRef
    #: The symbol's four corners in photo pixels, in the symbol's own frame:
    #: top-left, top-right, bottom-right, bottom-left.
    quad: Tuple[Tuple[float, float], ...]

    @property
    def corner(self) -> str:
        return self.ref.corner

    @property
    def center(self) -> Tuple[float, float]:
        xs = [point[0] for point in self.quad]
        ys = [point[1] for point in self.quad]
        return (sum(xs) / len(xs), sum(ys) / len(ys))


@dataclass
class Flattened:
    """A page recovered from a photograph."""

    image: object  # numpy array, HxW grayscale
    ref: PageRef  # notebook/page/side; corner is not meaningful here
    codes: List[DetectedCode] = field(default_factory=list)
    dpi: int = DEFAULT_DPI
    #: RMS reprojection error of the fit, in millimetres on the page.
    residual_mm: float = 0.0
    warnings: List[str] = field(default_factory=list)

    @property
    def confident(self) -> bool:
        return len(self.codes) >= 3 and self.residual_mm <= RESIDUAL_WARN_MM

    @property
    def name(self) -> str:
        """A stable filename stem that sorts into reading order."""
        return f"{self.ref.notebook}-p{self.ref.page:04d}{self.ref.side}"


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------


def _as_gray(cv2, numpy, image):
    array = numpy.asarray(image)
    if array.ndim == 3:
        return cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    return array


def _detect_in(detector, image, offset: Tuple[int, int]) -> List[Tuple[str, object]]:
    """Decode symbols in one image, returning payloads with absolute quads.

    Both detector calls are guarded. OpenCV does not merely fail to find a
    symbol on awkward input -- it raises out of its own internals (a convexHull
    assertion, for one) on perhaps one photograph in six. A detector that
    cannot read a tile is a miss, not a reason to abandon the photograph, and
    the other tiles usually find the code anyway.
    """
    found: List[Tuple[str, object]] = []
    ox, oy = offset

    try:
        ok, texts, points, _ = detector.detectAndDecodeMulti(image)
        if ok and texts is not None and points is not None:
            for text, quad in zip(texts, points):
                if text:
                    found.append((text, quad.reshape(-1, 2) + (ox, oy)))
    except Exception:
        pass

    # The multi-detector misses symbols the single one reads; ask both.
    try:
        text, quad, _ = detector.detectAndDecode(image)
        if text and quad is not None:
            found.append((text, quad.reshape(-1, 2) + (ox, oy)))
    except Exception:
        pass
    return found


def detect_codes(image, *, grid: int = TILE_GRID, overlap: float = TILE_OVERLAP) -> List[DetectedCode]:
    """Find and decode every paper-log code in a photograph.

    Searches the whole frame and then overlapping tiles, because the tiles are
    what actually find small symbols. Duplicates across tiles are collapsed,
    keeping the first quad seen for each payload.
    """
    cv2, numpy, _ = load_backend()
    gray = _as_gray(cv2, numpy, image)
    detector = cv2.QRCodeDetector()

    raw: List[Tuple[str, object]] = list(_detect_in(detector, gray, (0, 0)))

    height, width = gray.shape[:2]
    tile_h = int(height / grid * (1 + overlap))
    tile_w = int(width / grid * (1 + overlap))
    step_h = max(height // grid, 1)
    step_w = max(width // grid, 1)
    for row in range(grid):
        for column in range(grid):
            top = min(row * step_h, max(height - tile_h, 0))
            left = min(column * step_w, max(width - tile_w, 0))
            tile = gray[top : top + tile_h, left : left + tile_w]
            if tile.size:
                raw.extend(_detect_in(detector, tile, (left, top)))

    out: List[DetectedCode] = []
    seen = set()
    for payload, quad in raw:
        if payload in seen:
            continue
        try:
            ref = decode(payload)
        except TokenError:
            continue  # some other QR code that happens to be in shot
        seen.add(payload)
        out.append(
            DetectedCode(
                payload=payload,
                ref=ref,
                quad=tuple((float(x), float(y)) for x, y in quad),
            )
        )
    return out


# --------------------------------------------------------------------------
# geometry from the manifest
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PageGeometry:
    """Where the codes live on the printed page, in millimetres."""

    width_mm: float
    height_mm: float
    #: corner -> (x, y, size) with the origin at the bottom-left of the page.
    #: ``size`` is the printed footprint, quiet zone included.
    boxes: Dict[str, Tuple[float, float, float]]
    #: Width of the quiet zone in millimetres. A detector reports the bounds of
    #: the *dark module area*, not the footprint, so this has to come off every
    #: edge before photo points can be matched to page coordinates. Forgetting
    #: it costs a systematic error of exactly one quiet zone -- 1.3mm at the
    #: default settings, which is far more than the fit's real noise.
    quiet_mm: float = 0.0

    @classmethod
    def from_manifest(cls, manifest: Dict) -> "PageGeometry":
        page = manifest.get("page") or {}
        qr = manifest.get("qr") or {}
        geometry = qr.get("geometry") or {}
        if not geometry:
            raise CaptureError(
                "this manifest records no QR geometry, so there is nothing to "
                "align a photograph against"
            )
        boxes = {
            corner: (float(box["x_mm"]), float(box["y_mm"]), float(box["size_mm"]))
            for corner, box in geometry.items()
        }
        modules = int(qr.get("modules", 0))
        quiet_modules = int(qr.get("quiet_zone_modules", 0))
        any_size = next(iter(boxes.values()))[2]
        quiet_mm = (any_size / modules) * quiet_modules if modules else 0.0
        return cls(
            width_mm=float(page["width_mm"]),
            height_mm=float(page["height_mm"]),
            boxes=boxes,
            quiet_mm=quiet_mm,
        )

    def quad_mm(self, corner: str) -> List[Tuple[float, float]]:
        """The dark symbol's four corners in page millimetres.

        Ordered to match what the detector returns: the symbol's top-left,
        top-right, bottom-right, bottom-left. Page y grows upward, so the
        symbol's *top* is the larger y.
        """
        x, y, size = self.boxes[corner]
        low_x, low_y = x + self.quiet_mm, y + self.quiet_mm
        high_x, high_y = x + size - self.quiet_mm, y + size - self.quiet_mm
        return [(low_x, high_y), (high_x, high_y), (high_x, low_y), (low_x, low_y)]


def load_manifests(paths: Iterable[Path]) -> Dict[str, Dict]:
    """Index manifests by notebook id, so a photo can find its own."""
    import json

    out: Dict[str, Dict] = {}
    for path in paths:
        path = Path(path)
        candidates = sorted(path.glob("*.manifest.json")) if path.is_dir() else [path]
        for candidate in candidates:
            try:
                manifest = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise CaptureError(f"{candidate}: cannot read manifest ({exc})") from None
            notebook = (manifest.get("notebook") or {}).get("id")
            if notebook:
                out[str(notebook).upper()] = manifest
    return out


# --------------------------------------------------------------------------
# flattening
# --------------------------------------------------------------------------


def _fit_transform(cv2, numpy, source, target):
    """Best mapping from photo pixels to page millimetres.

    Falls back down the ladder as points run out: a homography needs four,
    an affine fit three, and a similarity two. Fewer points means a transform
    that cannot express as much distortion -- an affine fit cannot undo
    perspective at all -- which is why the caller reports how many codes it had.
    """
    source = numpy.asarray(source, dtype=numpy.float32)
    target = numpy.asarray(target, dtype=numpy.float32)

    if len(source) >= 8:
        matrix, _ = cv2.findHomography(source, target, cv2.RANSAC, 3.0)
        if matrix is not None:
            return matrix, "homography"
    if len(source) >= 4:
        matrix = cv2.getPerspectiveTransform(source[:4], target[:4]) if len(source) == 4 else None
        if matrix is None:
            matrix, _ = cv2.findHomography(source, target, 0)
        if matrix is not None:
            return matrix, "homography"
    if len(source) >= 3:
        affine, _ = cv2.estimateAffine2D(source, target)
        if affine is not None:
            return numpy.vstack([affine, [0, 0, 1]]).astype(numpy.float64), "affine"
    if len(source) >= 2:
        partial, _ = cv2.estimateAffinePartial2D(source, target)
        if partial is not None:
            return numpy.vstack([partial, [0, 0, 1]]).astype(numpy.float64), "similarity"
    raise CaptureError("not enough points to work out the page's position")


def _residual_mm(numpy, matrix, source, target) -> float:
    points = numpy.asarray(source, dtype=numpy.float64).reshape(-1, 1, 2)
    import cv2

    projected = cv2.perspectiveTransform(points, matrix).reshape(-1, 2)
    errors = numpy.linalg.norm(projected - numpy.asarray(target, dtype=numpy.float64), axis=1)
    return float(math.sqrt(float((errors ** 2).mean())))


def flatten_page(
    image,
    codes: Sequence[DetectedCode],
    geometry: PageGeometry,
    *,
    dpi: int = DEFAULT_DPI,
) -> Flattened:
    """Warp one page out of a photograph, given the codes found on it."""
    cv2, numpy, _ = load_backend()
    gray = _as_gray(cv2, numpy, image)
    if not codes:
        raise CaptureError("no codes to align against")

    reference = codes[0].ref
    source: List[Tuple[float, float]] = []
    target: List[Tuple[float, float]] = []
    used: List[DetectedCode] = []
    for code in codes:
        if code.corner not in geometry.boxes:
            continue
        source.extend(code.quad)
        target.extend(geometry.quad_mm(code.corner))
        used.append(code)

    if not used:
        raise CaptureError(
            "the codes in this photo sit in corners the manifest does not "
            "describe; is it the right notebook?"
        )

    matrix, method = _fit_transform(cv2, numpy, source, target)
    residual = _residual_mm(numpy, matrix, source, target)

    # Compose photo->mm with mm->pixels to warp straight to the output.
    scale = dpi / 25.4
    out_w = int(round(geometry.width_mm * scale))
    out_h = int(round(geometry.height_mm * scale))
    # Page y grows upward; image rows grow downward.
    to_pixels = numpy.array(
        [[scale, 0.0, 0.0], [0.0, -scale, geometry.height_mm * scale], [0.0, 0.0, 1.0]]
    )
    warp = to_pixels @ matrix

    flat = cv2.warpPerspective(
        gray, warp, (out_w, out_h), flags=cv2.INTER_CUBIC, borderValue=255
    )

    notes: List[str] = []
    if len(used) < 3:
        notes.append(
            f"only {len(used)} code(s) found, so the page was straightened with a "
            f"{method} fit — rotation and scale are corrected but tilt may remain. "
            "Shoot more square-on, or with all four corners in frame."
        )
    if residual > RESIDUAL_WARN_MM:
        notes.append(
            f"the codes did not fit a flat page well (RMS {residual:.1f}mm). "
            "The page may be curled, or a code was misread."
        )

    return Flattened(
        image=flat,
        ref=PageRef(reference.notebook, reference.page, reference.side, "TL"),
        codes=used,
        dpi=dpi,
        residual_mm=residual,
        warnings=notes,
    )


def group_by_page(codes: Sequence[DetectedCode]) -> Dict[Tuple[str, int], List[DetectedCode]]:
    """Bucket detected codes by the page they belong to.

    A photograph of an open notebook shows two pages at once, and both are
    worth recovering, so this does not assume a single page per frame.
    """
    groups: Dict[Tuple[str, int], List[DetectedCode]] = {}
    for code in codes:
        groups.setdefault((code.ref.notebook, code.ref.page), []).append(code)
    for group in groups.values():
        group.sort(key=lambda code: CORNERS.index(code.corner))
    return groups


def flatten(
    image,
    manifests: Dict[str, Dict],
    *,
    dpi: int = DEFAULT_DPI,
    min_codes: int = 1,
) -> List[Flattened]:
    """Recover every page visible in one photograph.

    ``manifests`` maps notebook id to manifest, as :func:`load_manifests`
    returns. Pages whose notebook is unknown are skipped with a clear error
    rather than guessed at.
    """
    codes = detect_codes(image)
    if not codes:
        raise CaptureError(
            "no paper-log codes found in this image. Check it is in focus, that "
            "the page corners are in frame, and that the photo is at least "
            "~2000 pixels across."
        )

    out: List[Flattened] = []
    for (notebook, _page), group in sorted(group_by_page(codes).items()):
        if len(group) < min_codes:
            continue
        manifest = manifests.get(notebook.upper())
        if manifest is None:
            raise CaptureError(
                f"found pages from notebook {notebook}, but no manifest for it. "
                "Pass its .manifest.json with --manifest."
            )
        geometry = PageGeometry.from_manifest(manifest)
        out.append(flatten_page(image, group, geometry, dpi=dpi))
    return out


ENHANCEMENTS = ("none", "flatten", "scan")


def enhance(image, mode: str = "flatten"):
    """Even out the lighting on a flattened page.

    A photograph carries the lamp with it: one corner bright, the opposite one
    in shadow, which looks wrong next to a flatbed scan and wrecks any later
    thresholding. Dividing by a heavily blurred copy of the page estimates that
    illumination and cancels it, leaving paper uniformly white and ink where it
    was.

    ``flatten`` keeps the greys, which suits handwriting and pencil. ``scan``
    pushes on to near black-and-white for the smallest files and the crispest
    look, at the cost of faint strokes.
    """
    if mode == "none":
        return image
    if mode not in ENHANCEMENTS:
        raise CaptureError(f"unknown enhancement {mode!r}; use one of {', '.join(ENHANCEMENTS)}")

    cv2, numpy, _ = load_backend()
    gray = image.astype(numpy.float32)

    # Estimate the paper level by closing away everything darker than the
    # kernel -- handwriting, ruling, print. A plain blur would do, but it gets
    # dragged down by the corner codes and leaves bright haloes around them.
    kernel_size = max(int(min(gray.shape[:2]) * 0.02) | 1, 15)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    paper = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    paper = cv2.GaussianBlur(paper, (0, 0), kernel_size)
    # The codes are far wider than any kernel that would leave ink alone, so
    # clamp the estimate: without this the page brightens around each symbol.
    paper = numpy.maximum(paper, 0.6 * float(numpy.median(paper)))

    # Dividing by the paper level puts white at a known place, which is the
    # whole point -- so there is no black point left to stretch. Stretching one
    # anyway is what turns a blank page grey and grainy: on a page that is 98%
    # paper, the 2nd percentile *is* paper, and rescaling from it amplifies
    # nothing but sensor noise.
    normalised = numpy.clip(gray / numpy.maximum(paper, 1.0) * 250.0, 0, 255)

    if mode == "scan":
        blurred = cv2.GaussianBlur(normalised.astype(numpy.uint8), (0, 0), 1.0)
        return cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 12
        )

    # A mild gamma deepens ink without touching paper: 250 stays 250, while a
    # mid-grey pencil stroke drops usefully darker.
    curved = 255.0 * numpy.power(normalised / 255.0, 1.4)
    return numpy.clip(curved, 0, 255).astype(numpy.uint8)


def read_image(path: Path):
    """Load a photograph as grayscale."""
    cv2, _, _ = load_backend()
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise CaptureError(f"{path}: not an image this build of OpenCV can read")
    return image


def write_image(image, path: Path) -> Path:
    cv2, _, _ = load_backend()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise CaptureError(f"{path}: could not write the flattened page")
    return path


# --------------------------------------------------------------------------
# the batch workflow
# --------------------------------------------------------------------------


@dataclass
class CaptureRecord:
    """One page recovered from one photograph, and where it was written."""

    source: Path
    page: Flattened
    path: Optional[Path] = None
    replaced: Optional[Path] = None  # a worse shot of the same page

    @property
    def quality(self) -> Tuple[int, float]:
        """Sort key for picking the better of two shots of the same page."""
        return (len(self.page.codes), -self.page.residual_mm)


@dataclass
class CaptureRun:
    records: List[CaptureRecord] = field(default_factory=list)
    failures: List[Tuple[Path, str]] = field(default_factory=list)
    superseded: List[CaptureRecord] = field(default_factory=list)

    def by_notebook(self) -> Dict[str, List[CaptureRecord]]:
        out: Dict[str, List[CaptureRecord]] = {}
        for record in self.records:
            out.setdefault(record.page.ref.notebook, []).append(record)
        for group in out.values():
            group.sort(key=lambda record: record.page.ref.page)
        return out

    def missing_pages(self, manifests: Dict[str, Dict]) -> Dict[str, List[int]]:
        """Pages a notebook has but this run never saw -- the ones to reshoot."""
        out: Dict[str, List[int]] = {}
        for notebook, records in self.by_notebook().items():
            manifest = manifests.get(notebook.upper())
            if not manifest:
                continue
            expected = {entry["page"] for entry in manifest.get("pages", [])}
            seen = {record.page.ref.page for record in records}
            gaps = sorted(expected - seen)
            if gaps:
                out[notebook] = gaps
        return out


def process(
    paths: Sequence[Path],
    manifests: Dict[str, Dict],
    out_dir: Optional[Path] = None,
    *,
    dpi: int = DEFAULT_DPI,
    enhancement: str = "flatten",
    suffix: str = ".png",
) -> CaptureRun:
    """Flatten a pile of photographs into named page images.

    Shooting the same page twice is normal -- you reshoot when you think one
    came out badly -- so a repeat does not become a second file. The capture
    that used more codes, and fit them better, wins; the other is reported as
    superseded rather than silently dropped.
    """
    run = CaptureRun()
    best: Dict[Tuple[str, int], CaptureRecord] = {}

    for path in paths:
        path = Path(path)
        try:
            image = read_image(path)
            pages = flatten(image, manifests, dpi=dpi)
        except CaptureError as exc:
            run.failures.append((path, str(exc)))
            continue

        for page in pages:
            record = CaptureRecord(source=path, page=page)
            key = (page.ref.notebook, page.ref.page)
            previous = best.get(key)
            if previous is None:
                best[key] = record
            elif record.quality > previous.quality:
                record.replaced = previous.source
                best[key] = record
                run.superseded.append(previous)
            else:
                run.superseded.append(record)

    run.records = sorted(
        best.values(), key=lambda r: (r.page.ref.notebook, r.page.ref.page)
    )

    if out_dir is not None:
        out_dir = Path(out_dir)
        for record in run.records:
            image = enhance(record.page.image, enhancement)
            target = out_dir / record.page.ref.notebook / f"{record.page.name}{suffix}"
            record.path = write_image(image, target)
    return run


def assemble_pdf(records: Sequence[CaptureRecord], output: Path, manifest: Dict) -> Path:
    """Bind flattened pages back into a PDF, in reading order.

    The point of the exercise: a pile of phone photos comes out the other end
    as the notebook, in order, at its original physical size.
    """
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas as pdfcanvas

    from .units import MM

    if not records:
        raise CaptureError("no pages to assemble")
    missing = [record for record in records if record.path is None]
    if missing:
        raise CaptureError("assemble_pdf needs pages that were written to disk")

    page = manifest.get("page") or {}
    width = float(page.get("width_mm", 210)) * MM
    height = float(page.get("height_mm", 297)) * MM

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pdf = pdfcanvas.Canvas(str(output), pagesize=(width, height))
    notebook = (manifest.get("notebook") or {}).get("id", "")
    pdf.setTitle(f"paper-log {notebook}")
    pdf.setCreator("paper-log")

    for record in sorted(records, key=lambda r: r.page.ref.page):
        pdf.drawImage(
            ImageReader(str(record.path)), 0, 0, width=width, height=height,
            preserveAspectRatio=True, anchor="c",
        )
        pdf.showPage()
    pdf.save()
    return output


__all__ = [
    "BackendMissing",
    "CaptureError",
    "CaptureRecord",
    "CaptureRun",
    "assemble_pdf",
    "enhance",
    "process",
    "DetectedCode",
    "Flattened",
    "PageGeometry",
    "detect_codes",
    "flatten",
    "flatten_page",
    "group_by_page",
    "load_manifests",
    "read_image",
    "write_image",
]
