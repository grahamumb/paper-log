"""End-to-end proof: the codes we print can be read back off the page.

These tests rasterise real PDFs and run a real QR decoder, so they are the
slowest in the suite -- and the only ones that actually check the premise of
the project. Page counts are kept small deliberately.
"""

import pytest

from paperlog import JournalConfig, build

pytest.importorskip("cv2", reason="needs the [verify] extras")
pytest.importorskip("pypdfium2", reason="needs the [verify] extras")

from paperlog.scancheck import expected_layout, verify  # noqa: E402


def journal(tmp_path, name="j.pdf", **overrides):
    settings = {"pages": 4, "notebook_id": "K7M2QX"}
    settings.update(overrides)
    config = JournalConfig.from_dict(settings)
    return config, build(config, tmp_path / name)


def test_every_printed_code_reads_back(tmp_path):
    _, result = journal(tmp_path)
    report = verify(result.pdf_path, result.manifest)
    assert report.ok, f"missing={report.missing} unexpected={report.unexpected}"
    assert report.decoded == report.expected == 16  # 4 pages x 4 corners


def test_decoded_tokens_name_the_right_page_side_and_corner(tmp_path):
    _, result = journal(tmp_path)
    report = verify(result.pdf_path, result.manifest)
    for scan in report.scans:
        expected_page = scan.index + 1
        assert {ref.page for ref in scan.refs} == {expected_page}
        assert {ref.side for ref in scan.refs} == {"F" if expected_page % 2 else "B"}
        assert {ref.corner for ref in scan.refs} == {"TL", "TR", "BL", "BR"}
        assert {ref.notebook for ref in scan.refs} == {"K7M2QX"}


def test_two_corners_still_work_for_people_who_want_less_ink(tmp_path):
    _, result = journal(tmp_path, qr={"corners": "TL,BR"})
    report = verify(result.pdf_path, result.manifest)
    assert report.ok
    assert report.decoded == 8
    assert {ref.corner for scan in report.scans for ref in scan.refs} == {"TL", "BR"}


def test_booklet_sheets_read_back(tmp_path):
    _, result = journal(tmp_path, pages=8, imposition="booklet")
    report = verify(result.pdf_path, result.manifest)
    assert report.ok, f"missing={report.missing}"
    assert report.pages_scanned == 4  # 8 pages -> 2 folded sheets -> 4 sides
    assert report.decoded == 32  # 8 pages x 4 corners


def test_url_payloads_read_back(tmp_path):
    _, result = journal(tmp_path, qr={"payload": "https://n.test/p/{token}"})
    report = verify(result.pdf_path, result.manifest)
    assert report.ok
    assert all(payload.startswith("https://n.test/p/") for scan in report.scans for payload in scan.payloads)


def test_undersized_codes_are_reported_as_unreadable(tmp_path):
    """The check has to be able to fail, or it is worth nothing.

    4mm codes carry the right data -- resampling a perfect render can even
    recover it -- but at well under a tenth of a millimetre per module they
    would never survive a real scan, so verification must reject them.
    """
    _, result = journal(tmp_path, qr={"size": "4mm", "inset": "2mm"})
    report = verify(result.pdf_path, result.manifest)
    assert not report.ok
    assert report.missing


def test_mismatched_manifest_is_detected(tmp_path):
    _, printed = journal(tmp_path, "printed.pdf", notebook_id="AAAAAA")
    _, other = journal(tmp_path, "other.pdf", notebook_id="BBBBBB")
    report = verify(printed.pdf_path, other.manifest)
    assert not report.ok
    assert report.missing and report.unexpected


def test_low_resolution_scans_fail_rather_than_silently_passing(tmp_path):
    """The check has to have a floor, or it proves nothing.

    The default 16mm codes are deliberately forgiving -- they still read at
    150dpi, where the old 13mm ECC-M codes did not -- so the floor has moved
    down rather than away. At 100dpi a module is under three pixels and no
    amount of processing gets it back.
    """
    _, result = journal(tmp_path)
    assert verify(result.pdf_path, result.manifest, dpi=150).ok
    assert not verify(result.pdf_path, result.manifest, dpi=100).ok


def test_bigger_codes_survive_a_lower_resolution_scan(tmp_path):
    _, result = journal(
        tmp_path,
        qr={"size": "17mm", "inset": "4mm"},
        margins={"top": "24mm", "bottom": "24mm", "inner": "24mm", "outer": "22mm"},
    )
    assert verify(result.pdf_path, result.manifest, dpi=200).ok


def test_sampling_only_part_of_the_document(tmp_path):
    _, result = journal(tmp_path, pages=8)
    report = verify(result.pdf_path, result.manifest, limit=2)
    assert report.pages_scanned == 2
    assert report.ok
    assert not report.complete  # a partial scan cannot prove nothing is missing


def test_expected_layout_places_booklet_codes_on_the_right_half(tmp_path):
    config, result = journal(tmp_path, pages=8, imposition="booklet")
    layout = expected_layout(result.manifest)
    page_width = result.manifest["page"]["width_mm"]
    # First sheet is (8, 1): page 8 on the left half, page 1 on the right.
    left = [code for code in layout[0] if code.page == 8]
    right = [code for code in layout[0] if code.page == 1]
    assert left and right
    assert all(code.x_mm < page_width for code in left)
    assert all(code.x_mm >= page_width for code in right)


# -- the decoder-independent bit check -------------------------------------


def render_page(pdf_path, index=0, dpi=300):
    import numpy
    import pypdfium2

    document = pypdfium2.PdfDocument(str(pdf_path))
    try:
        return numpy.asarray(document[index].render(scale=dpi / 72).to_pil().convert("L"))
    finally:
        document.close()


def test_bit_check_confirms_a_correctly_printed_symbol(tmp_path):
    import numpy

    from paperlog.scancheck import _bits_match

    _, result = journal(tmp_path)
    image = render_page(result.pdf_path, 0)
    code = expected_layout(result.manifest)[0][0]
    qr = result.manifest["qr"]
    assert _bits_match(
        numpy, image, code, 300, qr["modules"], qr["quiet_zone_modules"], "m"
    )


def test_bit_check_rejects_a_different_payload(tmp_path):
    """The check compares against ground truth, so it cannot rubber-stamp:
    point it at the right place with the wrong payload and it must say no."""
    import dataclasses

    import numpy

    from paperlog.scancheck import _bits_match

    _, result = journal(tmp_path)
    image = render_page(result.pdf_path, 0)
    code = expected_layout(result.manifest)[0][0]
    wrong = dataclasses.replace(code, payload="PL1:ZZZZZZ:9:B:BR:0000")
    qr = result.manifest["qr"]
    assert not _bits_match(
        numpy, image, wrong, 300, qr["modules"], qr["quiet_zone_modules"], "m"
    )


def test_bit_check_rejects_a_blank_region(tmp_path):
    import dataclasses

    import numpy

    from paperlog.scancheck import _bits_match

    _, result = journal(tmp_path)
    image = render_page(result.pdf_path, 0)
    code = expected_layout(result.manifest)[0][0]
    # Aim at the middle of the page, where there is no code at all.
    empty = dataclasses.replace(code, x_mm=60.0, y_mm=100.0)
    qr = result.manifest["qr"]
    assert not _bits_match(
        numpy, image, empty, 300, qr["modules"], qr["quiet_zone_modules"], "m"
    )


def test_symbols_this_decoder_cannot_read_still_verify(tmp_path):
    """Notebook 4APCMD page 5 produces a symbol OpenCV refuses to decode --
    even from a pristine render at some scales. It is a correct symbol, so
    verification has to pass it rather than blaming the page."""
    _, result = journal(tmp_path, pages=8, notebook_id="4APCMD")
    report = verify(result.pdf_path, result.manifest)
    assert report.ok, f"missing={report.missing}"
