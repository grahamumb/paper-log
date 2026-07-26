"""The read side: photograph a page, get a flat identified page back.

Every test here goes through a simulated phone camera (see ``conftest.py``) --
tilted, rotated, unevenly lit, downscaled, blurred and JPEG-compressed. That is
the whole point: the printing side can be proved with pixel comparisons, but
"will this survive a photograph" can only be answered by taking one.
"""

import pytest

from paperlog import JournalConfig, build

pytest.importorskip("cv2", reason="needs the [verify] extras")

from paperlog.capture import (  # noqa: E402
    CaptureError,
    PageGeometry,
    assemble_pdf,
    detect_codes,
    enhance,
    flatten,
    group_by_page,
    load_manifests,
    process,
)


@pytest.fixture
def notebook(tmp_path):
    config = JournalConfig.from_dict(
        {"page_size": "a5", "pages": 6, "notebook_id": "K7M2QX"}
    )
    result = build(config, tmp_path / "journal.pdf")
    manifests = load_manifests([result.manifest_path])
    return config, result, manifests


def test_a_photograph_identifies_its_page(notebook, render_page, phone):
    _, result, manifests = notebook
    picture = phone(render_page(result.pdf_path, 2), seed=1)

    pages = flatten(picture, manifests)
    assert len(pages) == 1
    page = pages[0]
    assert page.ref.notebook == "K7M2QX"
    assert page.ref.page == 3
    assert page.ref.side == "F"
    assert page.name == "K7M2QX-p0003F"


def test_the_flattened_page_is_square_on_and_the_right_size(notebook, render_page, phone):
    _, result, manifests = notebook
    picture = phone(render_page(result.pdf_path, 0), seed=2)
    page = flatten(picture, manifests, dpi=300)[0]

    # A5 at 300dpi, whatever shape the photograph was.
    height, width = page.image.shape[:2]
    assert width == pytest.approx(148 / 25.4 * 300, abs=2)
    assert height == pytest.approx(210 / 25.4 * 300, abs=2)


def test_the_fit_is_accurate_to_a_fraction_of_a_millimetre(notebook, render_page, phone):
    """Residual is the honest measure of the flattening.

    It also catches the failure that matters most and is invisible by eye: a
    systematic offset in the geometry, such as matching the detector's symbol
    bounds against the footprint including its quiet zone. That mistake costs
    about 1.3mm, which still produces a plausible-looking flat page.
    """
    _, result, manifests = notebook
    for seed in range(8):
        page = flatten(phone(render_page(result.pdf_path, 0), seed=seed), manifests)[0]
        assert len(page.codes) == 4, f"seed {seed}: only {len(page.codes)} codes"
        assert page.residual_mm < 0.6, f"seed {seed}: {page.residual_mm}mm"
        assert page.confident


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_a_page_photographed_any_way_up_comes_out_upright(
    notebook, render_page, phone, rotation
):
    """The corner is inside the token, so orientation needs no guessing."""
    import cv2

    _, result, manifests = notebook
    picture = phone(render_page(result.pdf_path, 0), seed=5)
    if rotation:
        codes = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
        picture = cv2.rotate(picture, codes[rotation])

    page = flatten(picture, manifests)[0]
    assert page.ref.page == 1
    assert page.residual_mm < 0.6
    # Portrait out, regardless of how the photo went in.
    height, width = page.image.shape[:2]
    assert height > width


def test_both_pages_of_an_open_spread_are_recovered(notebook, render_page, phone):
    """Photographing an open notebook catches two pages; keep both."""
    import numpy

    _, result, manifests = notebook
    left = render_page(result.pdf_path, 1)
    right = render_page(result.pdf_path, 2)
    spread = numpy.hstack([left, right])
    picture = phone(spread, width=3600, seed=6, tilt=0.05, rotation=2.0)

    pages = flatten(picture, manifests)
    assert {page.ref.page for page in pages} == {2, 3}


def test_a_photo_with_no_codes_says_so_usefully(notebook, phone):
    import numpy

    _, _, manifests = notebook
    blank = numpy.full((1200, 900), 200, numpy.uint8)
    with pytest.raises(CaptureError, match="no paper-log codes"):
        flatten(blank, manifests)


def test_an_unknown_notebook_is_named_not_guessed(notebook, render_page, phone):
    _, result, manifests = notebook
    picture = phone(render_page(result.pdf_path, 0), seed=7)
    with pytest.raises(CaptureError, match="no manifest"):
        flatten(picture, {})


def test_a_detector_crash_on_one_tile_does_not_lose_the_photograph(
    notebook, render_page, phone
):
    """OpenCV raises out of its own internals on some inputs -- a convexHull
    assertion, roughly one photograph in six with this simulator. A tile the
    detector cannot cope with has to count as a miss, not an exception."""
    _, result, manifests = notebook
    picture = phone(render_page(result.pdf_path, 0), seed=3)
    page = flatten(picture, manifests)[0]
    assert page.ref.page == 1
    assert page.confident


def test_detection_finds_all_four_corners(notebook, render_page, phone):
    _, result, manifests = notebook
    codes = detect_codes(phone(render_page(result.pdf_path, 0), seed=8))
    assert {code.corner for code in codes} == {"TL", "TR", "BL", "BR"}
    assert all(len(code.quad) == 4 for code in codes)


def test_group_by_page_separates_notebooks_and_pages(notebook, render_page, phone):
    _, result, manifests = notebook
    codes = detect_codes(phone(render_page(result.pdf_path, 3), seed=9))
    groups = group_by_page(codes)
    assert list(groups) == [("K7M2QX", 4)]


def test_geometry_accounts_for_the_quiet_zone(notebook):
    """The detector reports the dark symbol, not the printed footprint."""
    _, result, _ = notebook
    geometry = PageGeometry.from_manifest(result.manifest)
    assert geometry.quiet_mm > 0
    x, y, size = geometry.boxes["TL"]
    quad = geometry.quad_mm("TL")
    # The quad sits strictly inside the footprint, by the quiet zone all round.
    assert min(point[0] for point in quad) == pytest.approx(x + geometry.quiet_mm)
    assert max(point[0] for point in quad) == pytest.approx(x + size - geometry.quiet_mm)


def test_enhancement_whitens_paper_without_greying_it(notebook, render_page, phone):
    """A blank page must come out white, not mid-grey noise.

    Stretching to a low percentile does exactly that on a mostly-blank page,
    because the low percentile *is* paper.
    """
    import numpy

    _, result, manifests = notebook
    page = flatten(phone(render_page(result.pdf_path, 0), seed=10), manifests)[0]

    lit = enhance(page.image, "flatten")
    assert float(numpy.median(lit)) > 200
    # Ink survives: the corner codes are still properly dark somewhere.
    assert float(numpy.percentile(lit, 0.5)) < 120

    scanned = enhance(page.image, "scan")
    assert set(numpy.unique(scanned)) <= {0, 255}
    assert float(numpy.median(scanned)) == 255


def test_enhancement_evens_out_the_lighting(notebook, render_page, phone):
    import numpy

    _, result, manifests = notebook
    page = flatten(phone(render_page(result.pdf_path, 0), seed=11, shade=0.4), manifests)[0]

    def corner_spread(image):
        h, w = image.shape[:2]
        box = min(h, w) // 6
        # Sample the middle of each side, not the corners, which hold codes.
        mids = [
            image[h // 2 - box // 2 : h // 2 + box // 2, :box],
            image[h // 2 - box // 2 : h // 2 + box // 2, -box:],
        ]
        means = [float(numpy.mean(p)) for p in mids]
        return max(means) - min(means)

    assert corner_spread(enhance(page.image, "flatten")) < corner_spread(page.image)


def test_unknown_enhancement_is_rejected(notebook, render_page, phone):
    _, result, manifests = notebook
    page = flatten(phone(render_page(result.pdf_path, 0), seed=12), manifests)[0]
    with pytest.raises(CaptureError, match="unknown enhancement"):
        enhance(page.image, "sparkle")


# -- the batch workflow ----------------------------------------------------


def test_a_pile_of_photos_becomes_an_ordered_archive(notebook, render_page, phone, tmp_path):
    _, result, manifests = notebook
    shots = tmp_path / "shots"
    shots.mkdir()
    import cv2

    # Photograph pages 4, 1 and 3, in that order, at different angles.
    for index, (page_number, seed, rotation) in enumerate([(4, 1, 9.0), (1, 2, -5.0), (3, 3, 3.0)]):
        picture = phone(render_page(result.pdf_path, page_number - 1), seed=seed, rotation=rotation)
        cv2.imwrite(str(shots / f"IMG_{index}.jpg"), picture)

    run = process(sorted(shots.glob("*.jpg")), manifests, tmp_path / "pages")
    assert not run.failures
    # Sorted into reading order regardless of the order they were shot.
    assert [record.page.ref.page for record in run.records] == [1, 3, 4]
    assert all(record.path and record.path.exists() for record in run.records)
    assert run.missing_pages(manifests) == {"K7M2QX": [2, 5, 6]}


def test_reshooting_a_page_keeps_the_better_photograph(notebook, render_page, phone, tmp_path):
    """Reshooting is normal; it should not produce two files."""
    import cv2

    _, result, manifests = notebook
    shots = tmp_path / "shots"
    shots.mkdir()
    page = render_page(result.pdf_path, 0)
    cv2.imwrite(str(shots / "poor.jpg"), phone(page, width=2000, seed=4, blur=1.5, tilt=0.16))
    cv2.imwrite(str(shots / "good.jpg"), phone(page, width=3000, seed=5, blur=0.8))

    run = process(sorted(shots.glob("*.jpg")), manifests, tmp_path / "pages")
    assert len(run.records) == 1
    assert len(run.superseded) == 1
    kept = run.records[0]
    assert kept.page.residual_mm <= 0.6
    assert len(list((tmp_path / "pages" / "K7M2QX").iterdir())) == 1


def test_unreadable_photos_are_reported_not_fatal(notebook, render_page, phone, tmp_path):
    import cv2
    import numpy

    _, result, manifests = notebook
    shots = tmp_path / "shots"
    shots.mkdir()
    cv2.imwrite(str(shots / "good.jpg"), phone(render_page(result.pdf_path, 0), seed=6))
    cv2.imwrite(str(shots / "lens-cap.jpg"), numpy.full((900, 700), 30, numpy.uint8))

    run = process(sorted(shots.glob("*.jpg")), manifests, tmp_path / "pages")
    assert len(run.records) == 1
    assert len(run.failures) == 1
    assert run.failures[0][0].name == "lens-cap.jpg"


def test_recovered_pages_bind_back_into_a_pdf(notebook, render_page, phone, tmp_path):
    import cv2

    _, result, manifests = notebook
    shots = tmp_path / "shots"
    shots.mkdir()
    for index, page_number in enumerate([3, 1]):
        cv2.imwrite(
            str(shots / f"IMG_{index}.jpg"),
            phone(render_page(result.pdf_path, page_number - 1), seed=index),
        )

    run = process(sorted(shots.glob("*.jpg")), manifests, tmp_path / "pages")
    out = assemble_pdf(run.records, tmp_path / "book.pdf", result.manifest)
    assert out.exists()

    import pypdfium2

    document = pypdfium2.PdfDocument(str(out))
    try:
        assert len(document) == 2
        width, height = document[0].get_size()
        assert width == pytest.approx(148 / 25.4 * 72, abs=1)
        assert height == pytest.approx(210 / 25.4 * 72, abs=1)
    finally:
        document.close()
