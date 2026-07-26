import pytest

from paperlog import JournalConfig, build
from paperlog.imposition import padded_count
from paperlog.qrcodes import _runs, build_matrix, module_size, symbol_modules, worst_case_payload
from paperlog.render import (
    _caption_width_limit,
    build_pages,
    corner_origins,
    header_height,
)
from paperlog.units import MM

pdfium = pytest.importorskip("pypdfium2", reason="needs the [verify] extras")


def page_sizes(path):
    document = pdfium.PdfDocument(str(path))
    try:
        return [tuple(round(value, 1) for value in document[i].get_size()) for i in range(len(document))]
    finally:
        document.close()


@pytest.fixture
def config():
    return JournalConfig.from_dict(
        {"page_size": "a5", "pages": 8, "notebook_id": "K7M2QX4A"}
    )


def test_build_pages_numbers_and_sides(config):
    pages = build_pages(config)
    assert [page.number for page in pages] == list(range(1, 9))
    assert [page.side for page in pages] == ["F", "B"] * 4
    assert pages[0].ref.notebook == "K7M2QX4A"


def test_pdf_has_one_sheet_per_page(config, tmp_path):
    result = build(config, tmp_path / "j.pdf")
    sizes = page_sizes(result.pdf_path)
    assert len(sizes) == 8
    assert sizes[0] == (round(148 * MM, 1), round(210 * MM, 1))


def test_booklet_pdf_has_half_as_many_sheets_and_is_twice_as_wide(tmp_path):
    config = JournalConfig.from_dict({"page_size": "a5", "pages": 10, "imposition": "booklet"})
    result = build(config, tmp_path / "b.pdf")
    sizes = page_sizes(result.pdf_path)
    # 10 pages pads to 12, which is 3 folded sheets, which is 6 printed sides.
    assert padded_count(10) == 12
    assert len(sizes) == 6
    assert sizes[0] == (round(296 * MM, 1), round(210 * MM, 1))


def test_landscape_swaps_the_page_dimensions(tmp_path):
    config = JournalConfig.from_dict({"page_size": "a5", "pages": 2, "landscape": True})
    result = build(config, tmp_path / "l.pdf")
    assert page_sizes(result.pdf_path)[0] == (round(210 * MM, 1), round(148 * MM, 1))


def test_manifest_lists_every_page_with_unique_tokens(config, tmp_path):
    result = build(config, tmp_path / "j.pdf")
    manifest = result.manifest
    assert len(manifest["pages"]) == 8
    tokens = [entry["token"] for entry in manifest["pages"]]
    assert len(set(tokens)) == 8
    assert manifest["notebook"]["id"] == "K7M2QX4A"
    assert manifest["page"]["width_mm"] == pytest.approx(148, abs=0.01)


def test_manifest_records_a_payload_per_corner(config, tmp_path):
    result = build(config, tmp_path / "j.pdf")
    entry = result.manifest["pages"][0]
    assert set(entry["codes"]) == {"TL", "BR"}
    # Each corner carries a distinct payload, which is what lets a scanner work
    # out the page orientation from any single code it manages to read.
    assert len(set(entry["codes"].values())) == 2
    assert all(":TL:" in payload for payload in [entry["codes"]["TL"]])


def test_manifest_geometry_matches_the_renderer(config, tmp_path):
    result = build(config, tmp_path / "j.pdf")
    geometry = result.manifest["qr"]["geometry"]
    origins = corner_origins(config, config.qr.size, config.qr.inset)
    for corner, box in geometry.items():
        x, y = origins[corner]
        assert box["x_mm"] == pytest.approx(x / MM, abs=0.01)
        assert box["y_mm"] == pytest.approx(y / MM, abs=0.01)


def test_manifest_written_next_to_the_pdf(config, tmp_path):
    result = build(config, tmp_path / "nested" / "j.pdf")
    assert result.manifest_path == tmp_path / "nested" / "j.manifest.json"
    assert result.manifest_path.exists()


def test_manifest_can_be_skipped(config, tmp_path):
    result = build(config, tmp_path / "j.pdf", write_manifest=False)
    assert result.manifest_path is None
    assert not (tmp_path / "j.manifest.json").exists()
    assert result.manifest["pages"]  # still built in memory


def test_disabling_codes_leaves_them_out_of_the_manifest(tmp_path):
    config = JournalConfig.from_dict({"pages": 2, "qr": {"enabled": False}})
    result = build(config, tmp_path / "j.pdf")
    assert result.manifest["qr"]["enabled"] is False
    assert "codes" not in result.manifest["pages"][0]


@pytest.mark.parametrize("style", ["blank", "ruled", "dotted", "grid", "cornell"])
def test_every_ruling_renders(style, tmp_path):
    config = JournalConfig.from_dict({"pages": 2, "ruling": {"style": style}})
    result = build(config, tmp_path / f"{style}.pdf")
    assert result.pdf_path.stat().st_size > 0
    assert len(page_sizes(result.pdf_path)) == 2


@pytest.mark.parametrize("style", ["bracket", "square", "cross", "none"])
def test_every_fiducial_style_renders(style, tmp_path):
    config = JournalConfig.from_dict({"pages": 1, "fiducials": {"style": style}})
    assert build(config, tmp_path / f"{style}.pdf").pdf_path.exists()


def test_header_height_is_zero_when_nothing_is_printed():
    bare = JournalConfig.from_dict(
        {"furniture": {"date_line": False, "page_number": False, "title": False}}
    )
    assert header_height(bare) == 0.0
    assert header_height(JournalConfig.from_dict({})) > 0.0


def test_url_payloads_are_carried_through(tmp_path):
    config = JournalConfig.from_dict(
        {"pages": 2, "notebook_id": "K7M2QX4A", "qr": {"payload": "https://n.test/p/{token}"}}
    )
    result = build(config, tmp_path / "j.pdf")
    payload = result.manifest["pages"][0]["codes"]["TL"]
    assert payload.startswith("https://n.test/p/PL1:K7M2QX4A:1:F:TL:")


def test_warnings_are_reported_on_the_result(tmp_path):
    config = JournalConfig.from_dict({"pages": 2, "qr": {"size": "5mm"}})
    result = build(config, tmp_path / "j.pdf")
    assert any("below the" in note for note in result.warnings)


def test_captions_render(tmp_path):
    config = JournalConfig.from_dict({"pages": 2, "qr": {"caption": True, "corners": "all"}})
    assert build(config, tmp_path / "j.pdf").pdf_path.exists()


def test_top_captions_stop_short_of_the_text_block():
    """A caption under a top corner shares a line with the header, so it has
    to stop before the text block starts. A bottom one has the margin to
    itself and may always use the code's full width."""
    tight = JournalConfig.from_dict(
        {
            "page_size": "pocket",
            "margins": {"top": "16mm", "bottom": "16mm", "inner": "14mm", "outer": "9mm"},
            "qr": {"corners": "all", "size": "12mm", "inset": "3mm"},
        }
    )
    boxes = tight.qr_boxes()
    size = tight.qr.size
    assert _caption_width_limit(tight, "BL", boxes["BL"][0], size, "F") == size
    # The top-left code reaches to 15mm; the front text block starts at 14mm.
    assert _caption_width_limit(tight, "TL", boxes["TL"][0], size, "F") < size


def test_generous_margins_leave_top_captions_unclamped():
    roomy = JournalConfig.from_dict({"page_size": "a5", "qr": {"corners": "all"}})
    boxes = roomy.qr_boxes()
    assert _caption_width_limit(roomy, "TL", boxes["TL"][0], roomy.qr.size, "F") == roomy.qr.size


def test_caption_limit_never_exceeds_the_code_width():
    config = JournalConfig.from_dict({"page_size": "a5", "margins": {"all": "40mm"}})
    for corner, box in config.corner_boxes(config.qr.size, config.qr.inset).items():
        for side in ("F", "B"):
            limit = _caption_width_limit(config, corner, box[0], config.qr.size, side)
            assert limit <= config.qr.size


# -- QR helpers ------------------------------------------------------------


def test_runs_collapse_adjacent_modules():
    assert _runs([0, 0, 0]) == []
    assert _runs([1, 1, 0, 1]) == [(0, 2), (3, 1)]
    assert _runs([1, 1, 1]) == [(0, 3)]
    assert _runs([0, 1, 1, 0, 0, 1]) == [(1, 2), (5, 1)]


def test_matrix_is_square_and_deterministic():
    matrix = build_matrix("PL1:K7M2QX4A:1:F:TL:0000", "m")
    assert len(matrix) == len(matrix[0])
    assert build_matrix("PL1:K7M2QX4A:1:F:TL:0000", "m") == matrix


def test_symbol_modules_includes_the_quiet_zone():
    payload = "PL1:K7M2QX4A:1:F:TL:0000"
    bare = len(build_matrix(payload, "m"))
    assert symbol_modules(payload, "m", quiet_zone=2) == bare + 4
    assert symbol_modules(payload, "m", quiet_zone=0) == bare


def test_module_size_shrinks_as_error_correction_rises():
    low = JournalConfig.from_dict({"qr": {"error_correction": "l"}})
    high = JournalConfig.from_dict({"qr": {"error_correction": "h"}})
    assert module_size(low) > module_size(high)


def test_worst_case_payload_uses_the_last_page():
    config = JournalConfig.from_dict({"pages": 128, "notebook_id": "K7M2QX4A"})
    assert ":128:" in worst_case_payload(config)


def test_all_pages_share_one_module_pitch(tmp_path):
    """Page 9 and page 100 must print the same size symbol, or a scanner tuned
    to one page stops working halfway through the notebook."""
    config = JournalConfig.from_dict({"pages": 120, "notebook_id": "K7M2QX4A"})
    fixed = symbol_modules(worst_case_payload(config), "m", config.qr.quiet_zone)
    for page in (1, 9, 10, 99, 100, 120):
        payload = f"PL1:K7M2QX4A:{page}:F:TL:0000"
        assert symbol_modules(payload, "m", config.qr.quiet_zone) <= fixed
