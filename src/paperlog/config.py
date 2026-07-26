"""Journal configuration: the dataclasses, their defaults, and strict loading.

Config comes from YAML/JSON files, CLI flags, or Python. Loading is strict --
an unknown key is an error with a suggestion rather than a silently ignored
setting, because a typo in ``line_spacing`` should not quietly print you 200
pages of the wrong notebook.
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .ids import CORNERS, new_notebook_id, normalise_notebook_id
from .units import MM, to_points

Color = Tuple[float, float, float]


class ConfigError(ValueError):
    """Raised for invalid configuration, with a message aimed at the author."""


# --------------------------------------------------------------------------
# paper sizes (portrait, in points)
# --------------------------------------------------------------------------

PAGE_SIZES: Dict[str, Tuple[float, float]] = {
    "a3": (297 * MM, 420 * MM),
    "a4": (210 * MM, 297 * MM),
    "a5": (148 * MM, 210 * MM),
    "a6": (105 * MM, 148 * MM),
    "b5": (176 * MM, 250 * MM),
    "b6": (125 * MM, 176 * MM),
    "letter": (8.5 * 72, 11 * 72),
    "half-letter": (5.5 * 72, 8.5 * 72),
    "legal": (8.5 * 72, 14 * 72),
    "junior-legal": (5 * 72, 8 * 72),
    "pocket": (3.5 * 72, 5.5 * 72),
    "travelers": (110 * MM, 210 * MM),
    "travelers-passport": (89 * MM, 124 * MM),
}

RULING_STYLES = ("blank", "ruled", "dotted", "grid", "cornell")
FIDUCIAL_STYLES = ("bracket", "square", "cross", "none")
ECC_LEVELS = ("l", "m", "q", "h")
IMPOSITIONS = ("none", "booklet")


def parse_color(value: Any) -> Color:
    """Accept ``"#aabbcc"``, ``"#abc"``, a 0-1 gray float, or an ``[r,g,b]``."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        gray = float(value)
        if not 0.0 <= gray <= 1.0:
            raise ConfigError(f"gray levels run from 0 (black) to 1 (white), got {value}")
        return (gray, gray, gray)
    if isinstance(value, (list, tuple)):
        if len(value) != 3:
            raise ConfigError(f"colour lists must have 3 components, got {value!r}")
        out = tuple(float(component) for component in value)
        if not all(0.0 <= component <= 1.0 for component in out):
            raise ConfigError(f"colour components run from 0 to 1, got {value!r}")
        return out  # type: ignore[return-value]
    if isinstance(value, str):
        text = value.strip().lstrip("#")
        if len(text) == 3:
            text = "".join(char * 2 for char in text)
        if len(text) == 6 and re.fullmatch(r"[0-9a-fA-F]{6}", text):
            return tuple(int(text[i : i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]
    raise ConfigError(f"cannot parse colour {value!r}; use '#7a8fa6', 0.6, or [r, g, b]")


def _reject_unknown(data: Dict[str, Any], cls, context: str) -> None:
    known = {item.name for item in fields(cls)}
    for key in data:
        if key not in known:
            close = difflib.get_close_matches(key, known, n=1)
            hint = f"; did you mean {close[0]!r}?" if close else ""
            raise ConfigError(f"unknown {context} setting {key!r}{hint}")


def _one_of(value: Any, allowed: Sequence[str], name: str) -> str:
    text = str(value).strip().lower()
    if text not in allowed:
        raise ConfigError(f"{name} must be one of {', '.join(allowed)}; got {value!r}")
    return text


def _positive(value: float, name: str) -> float:
    if value <= 0:
        raise ConfigError(f"{name} must be greater than zero, got {value}")
    return value


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------


@dataclass
class Margins:
    """Page margins in points.

    ``inner`` is the binding edge and ``outer`` the open edge. On a duplex
    journal they swap sides every page so the gutter always lands at the spine.
    """

    #: The defaults leave every corner clear of the text block for a QR code
    #: at the default size and inset -- see ``JournalConfig.warnings``.
    top: float = 18 * MM
    bottom: float = 20 * MM
    inner: float = 20 * MM
    outer: float = 12 * MM

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Margins":
        data = dict(data or {})
        # left/right are friendlier for single-sided printing.
        if "left" in data:
            data["inner"] = data.pop("left")
        if "right" in data:
            data["outer"] = data.pop("right")
        if "all" in data:
            everywhere = data.pop("all")
            for side in ("top", "bottom", "inner", "outer"):
                data.setdefault(side, everywhere)
        _reject_unknown(data, cls, "margins")
        values = {key: to_points(value) for key, value in data.items()}
        for name, value in values.items():
            if value < 0:
                raise ConfigError(f"margin {name} cannot be negative, got {value}")
        return cls(**values)


@dataclass
class Ruling:
    """The writing surface itself."""

    style: str = "ruled"
    spacing: float = 7 * MM
    line_width: float = 0.4
    color: Color = (0.62, 0.68, 0.75)
    dot_radius: float = 0.35
    #: Extra blank space between the top of the text block and the first line.
    first_line_offset: float = 0.0
    #: Vertical rule down the binding side, legal-pad style.
    margin_rule: bool = False
    margin_rule_offset: float = 12 * MM
    margin_rule_color: Color = (0.85, 0.55, 0.55)
    #: Cornell only: width of the left-hand cue column and height of the
    #: summary band across the bottom.
    cue_column: float = 40 * MM
    summary_band: float = 45 * MM

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Ruling":
        data = dict(data or {})
        _reject_unknown(data, cls, "ruling")
        kwargs: Dict[str, Any] = {}
        if "style" in data:
            kwargs["style"] = _one_of(data["style"], RULING_STYLES, "ruling.style")
        for key in ("spacing", "first_line_offset", "margin_rule_offset", "cue_column", "summary_band"):
            if key in data:
                kwargs[key] = to_points(data[key])
        for key in ("line_width", "dot_radius"):
            if key in data:
                kwargs[key] = float(data[key])
        for key in ("color", "margin_rule_color"):
            if key in data:
                kwargs[key] = parse_color(data[key])
        if "margin_rule" in data:
            kwargs["margin_rule"] = bool(data["margin_rule"])
        ruling = cls(**kwargs)
        _positive(ruling.spacing, "ruling.spacing")
        return ruling


@dataclass
class QrSpec:
    """The corner codes -- the part that makes the pages machine-filable."""

    enabled: bool = True
    #: Which corners carry a code. Four gives a scanner an anchor at every
    #: corner (best deskew, most ink); two diagonal corners is a good default;
    #: one is enough to identify a page that was scanned squarely.
    corners: List[str] = field(default_factory=lambda: ["TL", "BR"])
    #: Overall footprint of the symbol *including* its quiet zone.
    size: float = 13 * MM
    #: Distance from the paper edge to that footprint.
    inset: float = 5 * MM
    #: Quiet zone in modules. The spec asks for 4, but that assumes something
    #: might be printed right up against the symbol. Here the code sits alone
    #: in a margin several millimetres wide, so the paper supplies the real
    #: quiet zone and a nominal 2 buys a usefully bigger module instead.
    quiet_zone: int = 2
    error_correction: str = "m"
    color: Color = (0.0, 0.0, 0.0)
    #: ``{token}`` keeps the symbol smallest. A URL template makes a generic
    #: phone camera open something useful.
    payload: str = "{token}"
    #: Warn below this printed module size. 0.4mm is about 4.7 pixels per
    #: module at 300dpi, which scans reliably; much under 0.33mm and both
    #: flatbeds and phone cameras start failing.
    min_module_size: float = 0.4 * MM
    #: Tiny human-readable page number printed under each code.
    caption: bool = False
    caption_size: float = 5.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QrSpec":
        data = dict(data or {})
        _reject_unknown(data, cls, "qr")
        kwargs: Dict[str, Any] = {}
        if "enabled" in data:
            kwargs["enabled"] = bool(data["enabled"])
        if "corners" in data:
            kwargs["corners"] = _parse_corners(data["corners"], "qr.corners")
        for key in ("size", "inset", "min_module_size"):
            if key in data:
                kwargs[key] = to_points(data[key])
        if "quiet_zone" in data:
            kwargs["quiet_zone"] = int(data["quiet_zone"])
        if "error_correction" in data:
            kwargs["error_correction"] = _one_of(
                data["error_correction"], ECC_LEVELS, "qr.error_correction"
            )
        if "color" in data:
            kwargs["color"] = parse_color(data["color"])
        if "payload" in data:
            kwargs["payload"] = str(data["payload"])
        if "caption" in data:
            kwargs["caption"] = bool(data["caption"])
        if "caption_size" in data:
            kwargs["caption_size"] = float(data["caption_size"])
        spec = cls(**kwargs)
        _positive(spec.size, "qr.size")
        if spec.quiet_zone < 0:
            raise ConfigError(f"qr.quiet_zone cannot be negative, got {spec.quiet_zone}")
        return spec


@dataclass
class Fiducials:
    """Registration marks in the corners that have no QR code.

    A QR symbol already gives a scanner a precise anchor point, but only where
    one is printed. Marks in the remaining corners let a crop/deskew step find
    all four page corners even on a page where a code is smudged or covered by
    the writer's hand.
    """

    style: str = "bracket"
    #: Corners to mark. ``"auto"`` means "the ones without a QR code".
    corners: Any = "auto"
    size: float = 5 * MM
    thickness: float = 0.9
    inset: float = 5 * MM
    color: Color = (0.0, 0.0, 0.0)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Fiducials":
        data = dict(data or {})
        _reject_unknown(data, cls, "fiducials")
        kwargs: Dict[str, Any] = {}
        if "style" in data:
            kwargs["style"] = _one_of(data["style"], FIDUCIAL_STYLES, "fiducials.style")
        if "corners" in data:
            value = data["corners"]
            kwargs["corners"] = (
                "auto"
                if isinstance(value, str) and value.strip().lower() == "auto"
                else _parse_corners(value, "fiducials.corners")
            )
        for key in ("size", "inset"):
            if key in data:
                kwargs[key] = to_points(data[key])
        if "thickness" in data:
            kwargs["thickness"] = float(data["thickness"])
        if "color" in data:
            kwargs["color"] = parse_color(data["color"])
        return cls(**kwargs)

    def resolve_corners(self, qr_corners: Sequence[str]) -> List[str]:
        if self.style == "none":
            return []
        if self.corners == "auto":
            return [corner for corner in CORNERS if corner not in qr_corners]
        return list(self.corners)


@dataclass
class Furniture:
    """Everything printed inside the text block that is not ruling."""

    date_line: bool = True
    date_label: str = "date"
    page_number: bool = True
    #: ``{page}`` and ``{notebook}`` are available.
    page_number_format: str = "{page}"
    title: bool = False
    title_text: str = ""
    header_rule: bool = True
    font: str = "Helvetica"
    size: float = 7.5
    color: Color = (0.55, 0.6, 0.66)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Furniture":
        data = dict(data or {})
        _reject_unknown(data, cls, "furniture")
        kwargs: Dict[str, Any] = {}
        for key in ("date_line", "page_number", "title", "header_rule"):
            if key in data:
                kwargs[key] = bool(data[key])
        for key in ("date_label", "page_number_format", "title_text"):
            if key in data:
                kwargs[key] = str(data[key])
        if "font" in data:
            kwargs["font"] = _validate_font(data["font"])
        if "size" in data:
            kwargs["size"] = float(data["size"])
        if "color" in data:
            kwargs["color"] = parse_color(data["color"])
        return cls(**kwargs)


def available_fonts() -> List[str]:
    """The fonts ReportLab can use without embedding anything."""
    from reportlab.pdfbase import pdfmetrics

    return sorted(pdfmetrics.standardFonts)


def _validate_font(value: Any) -> str:
    """Catch a bad font name here, rather than as a KeyError mid-render."""
    name = str(value)
    fonts = available_fonts()
    if name in fonts:
        return name
    lowered = {font.lower(): font for font in fonts}
    if name.lower() in lowered:
        return lowered[name.lower()]
    close = difflib.get_close_matches(name, fonts, n=1)
    hint = f"; did you mean {close[0]!r}?" if close else ""
    raise ConfigError(
        f"unknown font {value!r}{hint}. Only the standard PDF fonts are "
        f"available: {', '.join(fonts)}"
    )


def _parse_corners(value: Any, name: str) -> List[str]:
    if isinstance(value, str):
        items = [part for part in re.split(r"[,\s]+", value.strip()) if part]
    elif isinstance(value, (list, tuple)):
        items = [str(item) for item in value]
    else:
        raise ConfigError(f"{name} must be a list or comma-separated string, got {value!r}")

    out: List[str] = []
    for item in items:
        corner = item.strip().upper()
        aliases = {
            "TOP-LEFT": "TL", "TOP_LEFT": "TL",
            "TOP-RIGHT": "TR", "TOP_RIGHT": "TR",
            "BOTTOM-LEFT": "BL", "BOTTOM_LEFT": "BL",
            "BOTTOM-RIGHT": "BR", "BOTTOM_RIGHT": "BR",
        }
        corner = aliases.get(corner, corner)
        if corner == "ALL":
            out.extend(CORNERS)
            continue
        if corner == "NONE":
            continue
        if corner not in CORNERS:
            raise ConfigError(f"{name}: {item!r} is not one of {', '.join(CORNERS)}")
        if corner not in out:
            out.append(corner)
    return out


# --------------------------------------------------------------------------
# the whole journal
# --------------------------------------------------------------------------


@dataclass
class JournalConfig:
    notebook_id: str = field(default_factory=new_notebook_id)
    title: str = ""
    pages: int = 64
    #: Printed double-sided: odd pages are fronts, even pages are backs, and
    #: the inner margin alternates so the gutter stays at the spine.
    duplex: bool = True
    page_width: float = PAGE_SIZES["a5"][0]
    page_height: float = PAGE_SIZES["a5"][1]
    landscape: bool = False
    margins: Margins = field(default_factory=Margins)
    ruling: Ruling = field(default_factory=Ruling)
    qr: QrSpec = field(default_factory=QrSpec)
    fiducials: Fiducials = field(default_factory=Fiducials)
    furniture: Furniture = field(default_factory=Furniture)
    imposition: str = "none"
    #: Draw the crop/fold guides that imposition implies.
    crop_marks: bool = True

    # -- derived -------------------------------------------------------
    @property
    def size(self) -> Tuple[float, float]:
        if self.landscape:
            return (self.page_height, self.page_width)
        return (self.page_width, self.page_height)

    def text_block(self, side: str) -> Tuple[float, float, float, float]:
        """``(x, y, width, height)`` of the writing area for a given side.

        On a duplex journal a front page binds on the left and a back page
        binds on the right, so ``inner`` swaps sides.
        """
        width, height = self.size
        binding_left = (side == "F") or not self.duplex
        left = self.margins.inner if binding_left else self.margins.outer
        right = self.margins.outer if binding_left else self.margins.inner
        block_width = width - left - right
        block_height = height - self.margins.top - self.margins.bottom
        return (left, self.margins.bottom, block_width, block_height)

    def corner_boxes(
        self, size: float, inset: float
    ) -> Dict[str, Tuple[float, float, float, float]]:
        """``(x, y, w, h)`` of a ``size``-square box tucked into each corner."""
        width, height = self.size
        far_x = width - inset - size
        far_y = height - inset - size
        return {
            "TL": (inset, far_y, size, size),
            "TR": (far_x, far_y, size, size),
            "BL": (inset, inset, size, size),
            "BR": (far_x, inset, size, size),
        }

    def qr_boxes(self) -> Dict[str, Tuple[float, float, float, float]]:
        """Corner boxes for the QR codes this journal will actually print."""
        if not (self.qr.enabled and self.qr.corners):
            return {}
        boxes = self.corner_boxes(self.qr.size, self.qr.inset)
        return {corner: boxes[corner] for corner in self.qr.corners}

    # -- construction --------------------------------------------------
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JournalConfig":
        data = dict(data or {})
        kwargs: Dict[str, Any] = {}

        page_size = data.pop("page_size", None)
        if page_size is not None:
            kwargs["page_width"], kwargs["page_height"] = parse_page_size(page_size)
        for key in ("page_width", "page_height"):
            if key in data:
                kwargs[key] = to_points(data.pop(key))

        sections = {
            "margins": Margins,
            "ruling": Ruling,
            "qr": QrSpec,
            "fiducials": Fiducials,
            "furniture": Furniture,
        }
        for key, section_cls in sections.items():
            if key in data:
                value = data.pop(key)
                if not isinstance(value, dict):
                    raise ConfigError(f"{key} must be a mapping, got {type(value).__name__}")
                kwargs[key] = section_cls.from_dict(value)

        if "notebook_id" in data:
            kwargs["notebook_id"] = normalise_notebook_id(str(data.pop("notebook_id")))
        if "title" in data:
            kwargs["title"] = str(data.pop("title"))
        if "pages" in data:
            kwargs["pages"] = int(data.pop("pages"))
        for key in ("duplex", "landscape", "crop_marks"):
            if key in data:
                kwargs[key] = bool(data.pop(key))
        if "imposition" in data:
            kwargs["imposition"] = _one_of(data.pop("imposition"), IMPOSITIONS, "imposition")

        _reject_unknown(data, cls, "journal")
        config = cls(**kwargs)
        config.validate()
        return config

    def validate(self) -> None:
        if self.pages < 1:
            raise ConfigError(f"pages must be at least 1, got {self.pages}")
        _positive(self.page_width, "page_width")
        _positive(self.page_height, "page_height")

        width, height = self.size
        for side in ("F", "B"):
            _, _, block_width, block_height = self.text_block(side)
            if block_width <= 0 or block_height <= 0:
                raise ConfigError(
                    f"margins leave no room to write: page is {width / MM:.0f}x"
                    f"{height / MM:.0f}mm but the text block works out at "
                    f"{block_width / MM:.0f}x{block_height / MM:.0f}mm"
                )
        if self.qr.enabled and self.qr.corners:
            if self.qr.size + self.qr.inset > min(width, height) / 2:
                raise ConfigError(
                    "qr.size + qr.inset does not fit in the corner of this page"
                )
        if self.imposition == "booklet" and self.landscape:
            raise ConfigError(
                "booklet imposition expects portrait pages; two of them are "
                "placed side by side on a landscape sheet"
            )

    def warnings(self) -> List[str]:
        """Non-fatal problems worth telling the user about before they print."""
        from .qrcodes import module_size  # local import: avoids a cycle

        notes: List[str] = []
        if self.qr.enabled and self.qr.corners:
            printed = module_size(self)
            if printed < self.qr.min_module_size:
                notes.append(
                    f"QR modules print at {printed / MM:.2f}mm, below the "
                    f"{self.qr.min_module_size / MM:.2f}mm threshold. Increase qr.size, "
                    f"shorten qr.payload, or lower qr.error_correction."
                )
            notes.extend(self._overlap_warnings())
        if self.imposition == "booklet" and self.pages % 4:
            notes.append(
                f"{self.pages} pages is not a multiple of 4; "
                f"{4 - self.pages % 4} blank page(s) will be added to complete the "
                "last folded sheet."
            )
        if self.duplex and self.pages % 2:
            notes.append(
                f"{self.pages} pages is odd, so the last leaf will have a blank back."
            )
        return notes

    def _overlap_warnings(self) -> List[str]:
        """Report QR codes that would be written over.

        A code only collides with the writing area when it overlaps in *both*
        axes, so the useful advice is per corner: to clear a top corner you can
        widen either the top margin or the side margin next to it, whichever
        costs you less writing room.
        """
        notes: List[str] = []
        sides = ("F", "B") if self.duplex else ("F",)
        clearance = self.qr.size + self.qr.inset

        for corner, box in self.qr_boxes().items():
            hit_sides = [side for side in sides if _intersects(box, self.text_block(side))]
            if not hit_sides:
                continue
            vertical = "top" if corner.startswith("T") else "bottom"
            which = (
                "front and back pages"
                if len(hit_sides) == len(sides) > 1
                else ("front pages" if hit_sides[0] == "F" else "back pages")
            )
            margin_now = getattr(self.margins, vertical)
            notes.append(
                f"The {corner} code overlaps the writing area on {which}. Give it "
                f"{clearance / MM:.0f}mm of clearance: raise the {vertical} margin "
                f"from {margin_now / MM:.0f}mm, widen the side margin beside it, or "
                f"shrink qr.size / qr.inset."
            )
        return notes

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data.pop("page_width")
        data.pop("page_height")
        data["page_size"] = f"{self.page_width / MM:.4g}mm x {self.page_height / MM:.4g}mm"
        for name in PAGE_SIZES:
            named = PAGE_SIZES[name]
            if abs(named[0] - self.page_width) < 0.5 and abs(named[1] - self.page_height) < 0.5:
                data["page_size"] = name
                break
        return data


def _intersects(
    a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]
) -> bool:
    """Do two ``(x, y, w, h)`` rectangles overlap by more than a rounding error?"""
    epsilon = 1e-6
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return (
        ax + aw > bx + epsilon
        and bx + bw > ax + epsilon
        and ay + ah > by + epsilon
        and by + bh > ay + epsilon
    )


def parse_page_size(value: Any) -> Tuple[float, float]:
    """``"a5"``, ``"148x210mm"``, ``"5.5in x 8.5in"``, or ``[width, height]``."""
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ConfigError(f"page_size list must be [width, height], got {value!r}")
        return (to_points(value[0]), to_points(value[1]))
    if not isinstance(value, str):
        raise ConfigError(f"cannot parse page_size {value!r}")

    text = value.strip().lower()
    if text in PAGE_SIZES:
        return PAGE_SIZES[text]

    parts = re.split(r"\s*[x×*]\s*", text)
    if len(parts) == 2:
        width_text, height_text = parts
        # "148x210mm" -- the unit is written once, on the second number.
        unit_match = re.search(r'(mm|cm|in|pt|")$', height_text)
        if unit_match and not re.search(r'(mm|cm|in|pt|")$', width_text):
            width_text += unit_match.group(1)
        try:
            return (to_points(width_text), to_points(height_text))
        except ValueError as exc:
            raise ConfigError(f"cannot parse page_size {value!r}: {exc}") from None

    close = difflib.get_close_matches(text, PAGE_SIZES, n=1)
    hint = f"; did you mean {close[0]!r}?" if close else ""
    raise ConfigError(
        f"unknown page size {value!r}{hint} "
        f"(known: {', '.join(sorted(PAGE_SIZES))}, or '148x210mm')"
    )


def load_config(path: Optional[Path], overrides: Optional[Dict[str, Any]] = None) -> JournalConfig:
    """Load a YAML or JSON config, apply ``overrides``, and validate."""
    data: Dict[str, Any] = {}
    if path is not None:
        text = Path(path).read_text(encoding="utf-8")
        if str(path).endswith(".json"):
            data = json.loads(text)
        else:
            import yaml

            data = yaml.safe_load(text) or {}
        if not isinstance(data, dict):
            raise ConfigError(f"{path}: expected a mapping at the top level")
    if overrides:
        data = deep_merge(data, overrides)
    return JournalConfig.from_dict(data)


def deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """Merge ``patch`` into ``base``, recursing into nested mappings."""
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out
