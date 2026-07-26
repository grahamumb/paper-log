"""Shared fixtures, including a simulated phone camera.

The capture tests need photographs, and real ones cannot live in a test suite.
:func:`photograph` fakes one: it takes a rendered page and applies the things a
hand-held phone actually does to it -- perspective from not being square-on,
in-plane rotation, a background around the sheet, uneven lighting, downscaling,
focus blur, sensor noise and JPEG artefacts.

It is a simulation, so it proves the geometry and the decoding rather than the
optics. Numbers measured against it are quoted in the README as what they are:
simulated captures, not photographs of paper.
"""

from __future__ import annotations

import pytest

from paperlog.library import HOME_VARIABLE


@pytest.fixture(autouse=True)
def isolated_library(tmp_path, monkeypatch):
    """Never let a test write into the real ~/.paperlog.

    ``paperlog build`` files a copy of every manifest it makes, which is the
    right behaviour for a person and quite wrong for a test suite -- without
    this, running the tests litters the developer's home directory with
    notebooks that do not exist.
    """
    monkeypatch.setenv(HOME_VARIABLE, str(tmp_path / "paperlog-home"))


def _backend():
    """Import the capture backend, skipping the test if it is not installed.

    Done inside fixtures rather than at module scope: an ``importorskip`` at
    the top of a conftest aborts collection for the whole suite, including the
    many tests that need nothing more than reportlab.
    """
    cv2 = pytest.importorskip("cv2", reason="needs the [verify] extras")
    numpy = pytest.importorskip("numpy", reason="needs the [verify] extras")
    pdfium = pytest.importorskip("pypdfium2", reason="needs the [verify] extras")
    return cv2, numpy, pdfium


@pytest.fixture
def render_page():
    """Rasterise one page of a PDF, as the printer would put it on paper."""
    _, numpy, pdfium = _backend()

    def _render(pdf_path, index=0, dpi=300):
        document = pdfium.PdfDocument(str(pdf_path))
        try:
            return numpy.asarray(
                document[index].render(scale=dpi / 72).to_pil().convert("L")
            )
        finally:
            document.close()

    return _render


def photograph(
    page,
    *,
    width=2400,
    tilt=0.12,
    rotation=6.0,
    blur=1.2,
    noise=5.0,
    jpeg=70,
    shade=0.28,
    seed=0,
    margin=0.10,
):
    """Fake a hand-held phone photo of a printed page.

    ``width`` is the output width in pixels -- the single most important knob,
    since it decides how many pixels each QR module gets. ``tilt`` is how far
    off square-on the camera is, as a fraction of the page.
    """
    cv2, numpy, _ = _backend()
    rng = numpy.random.default_rng(seed)
    height, page_width = page.shape
    source = numpy.float32(
        [[0, 0], [page_width, 0], [page_width, height], [0, height]]
    )

    # Shove each corner around to fake a camera that is not parallel to the page.
    d = tilt
    target = numpy.float32([
        [page_width * d * rng.uniform(0.3, 1.0), height * d * rng.uniform(0.3, 1.0)],
        [page_width * (1 - d * rng.uniform(0.0, 0.6)), height * d * rng.uniform(0.0, 0.6)],
        [page_width * (1 - d * rng.uniform(0.3, 1.0)), height * (1 - d * rng.uniform(0.3, 1.0))],
        [page_width * d * rng.uniform(0.0, 0.6), height * (1 - d * rng.uniform(0.0, 0.6))],
    ])

    angle = numpy.deg2rad(rotation)
    cos, sin = numpy.cos(angle), numpy.sin(angle)
    cx, cy = page_width / 2, height / 2
    rotate = numpy.float32(
        [[cos, -sin, cx - cos * cx + sin * cy], [sin, cos, cy - sin * cx - cos * cy]]
    )
    target = (target @ rotate[:, :2].T) + rotate[:, 2]

    pad = int(max(page_width, height) * margin)
    target += pad
    canvas_w, canvas_h = page_width + 2 * pad, height + 2 * pad
    background = numpy.full((canvas_h, canvas_w), 150, numpy.uint8)
    background = cv2.add(
        background,
        rng.normal(0, 12, background.shape).astype(numpy.int16).clip(-40, 40).astype(numpy.uint8),
    )

    matrix = cv2.getPerspectiveTransform(source, target.astype(numpy.float32))
    warped = cv2.warpPerspective(
        page, matrix, (canvas_w, canvas_h), flags=cv2.INTER_AREA,
        borderMode=cv2.BORDER_TRANSPARENT, dst=background.copy(),
    )

    # A lamp off to one side, plus lens vignetting.
    yy, xx = numpy.mgrid[0:canvas_h, 0:canvas_w].astype(numpy.float32)
    gradient = 1.0 - shade * (xx / canvas_w) - shade * 0.5 * (yy / canvas_h)
    radius = numpy.sqrt(
        ((xx - canvas_w / 2) / canvas_w) ** 2 + ((yy - canvas_h / 2) / canvas_h) ** 2
    )
    gradient *= 1.0 - 0.35 * radius
    lit = numpy.clip(warped.astype(numpy.float32) * gradient, 0, 255)

    scale = width / canvas_w
    small = cv2.resize(
        lit, (width, int(canvas_h * scale)), interpolation=cv2.INTER_AREA
    )
    if blur > 0:
        small = cv2.GaussianBlur(small, (0, 0), blur)
    small = numpy.clip(small + rng.normal(0, noise, small.shape), 0, 255).astype(numpy.uint8)

    if jpeg:
        ok, encoded = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, jpeg])
        if ok:
            small = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    return small


@pytest.fixture
def phone():
    return photograph
