import json

import pytest

from paperlog.config import (
    ConfigError,
    JournalConfig,
    Margins,
    load_config,
    parse_color,
    parse_page_size,
)
from paperlog.units import MM, UnitError, to_points


def test_length_parsing():
    assert to_points(10) == pytest.approx(10 * MM)
    assert to_points("10mm") == pytest.approx(10 * MM)
    assert to_points("1cm") == pytest.approx(10 * MM)
    assert to_points("1in") == pytest.approx(72)
    assert to_points("36pt") == pytest.approx(36)
    assert to_points("0.5in") == pytest.approx(36)
    with pytest.raises(UnitError):
        to_points("about yea big")
    with pytest.raises(UnitError):
        to_points(True)


def test_page_size_parsing():
    assert parse_page_size("a4") == pytest.approx((210 * MM, 297 * MM))
    assert parse_page_size("148x210mm") == pytest.approx((148 * MM, 210 * MM))
    assert parse_page_size("5.5in x 8.5in") == pytest.approx((5.5 * 72, 8.5 * 72))
    assert parse_page_size([100, 200]) == pytest.approx((100 * MM, 200 * MM))


def test_unknown_page_size_suggests_a_real_one():
    with pytest.raises(ConfigError, match="a4"):
        parse_page_size("a44")


def test_colour_parsing():
    assert parse_color("#ffffff") == (1.0, 1.0, 1.0)
    assert parse_color("#000") == (0.0, 0.0, 0.0)
    assert parse_color(0.5) == (0.5, 0.5, 0.5)
    with pytest.raises(ConfigError):
        parse_color("chartreuse")


def test_unknown_key_is_an_error_with_a_suggestion():
    with pytest.raises(ConfigError, match="did you mean 'spacing'"):
        JournalConfig.from_dict({"ruling": {"spaceing": "7mm"}})
    with pytest.raises(ConfigError, match="unknown journal setting"):
        JournalConfig.from_dict({"colour_scheme": "blue"})


def test_margins_accept_left_right_and_all():
    margins = Margins.from_dict({"all": "10mm", "left": "25mm"})
    assert margins.inner == pytest.approx(25 * MM)
    assert margins.outer == pytest.approx(10 * MM)
    assert margins.top == pytest.approx(10 * MM)


def test_text_block_mirrors_on_duplex():
    config = JournalConfig.from_dict({"page_size": "a5", "duplex": True})
    front = config.text_block("F")
    back = config.text_block("B")
    assert front[0] == pytest.approx(config.margins.inner)
    assert back[0] == pytest.approx(config.margins.outer)
    # Same size block, just shifted across the gutter.
    assert front[2] == pytest.approx(back[2])
    assert front[3] == pytest.approx(back[3])


def test_text_block_does_not_mirror_when_single_sided():
    config = JournalConfig.from_dict({"page_size": "a5", "duplex": False})
    assert config.text_block("F") == config.text_block("B")


def test_impossible_margins_are_rejected():
    with pytest.raises(ConfigError, match="no room to write"):
        JournalConfig.from_dict({"page_size": "a6", "margins": {"all": "60mm"}})


def test_defaults_keep_every_corner_clear_of_the_text_block():
    config = JournalConfig.from_dict({"page_size": "a5", "qr": {"corners": "all"}})
    assert config.warnings() == []


def test_overlapping_code_is_reported_per_corner():
    config = JournalConfig.from_dict(
        {"page_size": "a5", "qr": {"corners": ["TR"], "size": "20mm", "inset": "8mm"}}
    )
    notes = config.warnings()
    assert any("TR code overlaps" in note for note in notes)


def test_small_modules_are_reported():
    config = JournalConfig.from_dict({"page_size": "a5", "qr": {"size": "6mm"}})
    assert any("below the" in note for note in config.warnings())


def test_odd_page_count_warns_about_the_blank_back():
    config = JournalConfig.from_dict({"pages": 7, "duplex": True})
    assert any("odd" in note for note in config.warnings())


def test_booklet_pads_to_a_multiple_of_four():
    config = JournalConfig.from_dict({"pages": 30, "imposition": "booklet"})
    assert any("multiple of 4" in note for note in config.warnings())


def test_booklet_rejects_landscape():
    with pytest.raises(ConfigError, match="booklet"):
        JournalConfig.from_dict({"imposition": "booklet", "landscape": True})


def test_load_config_from_yaml(tmp_path):
    path = tmp_path / "journal.yaml"
    path.write_text("page_size: a6\npages: 12\nruling:\n  style: dotted\n")
    config = load_config(path)
    assert config.pages == 12
    assert config.ruling.style == "dotted"


def test_load_config_from_json_with_overrides(tmp_path):
    path = tmp_path / "journal.json"
    path.write_text(json.dumps({"pages": 12, "ruling": {"style": "dotted", "spacing": "6mm"}}))
    config = load_config(path, {"pages": 20, "ruling": {"style": "grid"}})
    assert config.pages == 20
    assert config.ruling.style == "grid"
    # The override merges into the section rather than replacing it wholesale.
    assert config.ruling.spacing == pytest.approx(6 * MM)


def test_corners_accept_strings_lists_and_all():
    assert JournalConfig.from_dict({"qr": {"corners": "tl, br"}}).qr.corners == ["TL", "BR"]
    assert JournalConfig.from_dict({"qr": {"corners": "all"}}).qr.corners == ["TL", "TR", "BL", "BR"]
    assert JournalConfig.from_dict({"qr": {"corners": ["top-left"]}}).qr.corners == ["TL"]
    with pytest.raises(ConfigError):
        JournalConfig.from_dict({"qr": {"corners": ["nowhere"]}})


def test_fiducials_default_to_the_corners_without_codes():
    config = JournalConfig.from_dict({"qr": {"corners": ["TL", "BR"]}})
    assert config.fiducials.resolve_corners(config.qr.corners) == ["TR", "BL"]
    off = JournalConfig.from_dict({"fiducials": {"style": "none"}})
    assert off.fiducials.resolve_corners([]) == []


def test_unknown_font_is_rejected_at_config_time():
    """Better a clear error here than a KeyError from deep inside the renderer."""
    with pytest.raises(ConfigError, match="unknown font"):
        JournalConfig.from_dict({"furniture": {"font": "Comic Sans"}})


def test_font_names_are_case_insensitive():
    config = JournalConfig.from_dict({"furniture": {"font": "helvetica"}})
    assert config.furniture.font == "Helvetica"
