"""The sidecar manifest.

The PDF is for the printer; the manifest is for whatever software later reads
the scans. It lists every page, the exact string encoded in each of its corner
codes, and enough geometry (in millimetres, relative to the page) for a
scanning pipeline to know where those codes should be found once it has
deskewed the image.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .ids import FORMAT_VERSION, PageRef, render_payload
from .units import format_mm

MANIFEST_VERSION = 1


@dataclass(frozen=True)
class Page:
    """One page of the journal as laid out."""

    number: int
    side: str
    ref: PageRef

    @property
    def token(self) -> str:
        return self.ref.token


def corner_geometry(config) -> Dict[str, Dict[str, float]]:
    """Where each QR code sits, in millimetres from the bottom-left of the page.

    Reported per side because the page geometry itself is symmetric -- only the
    margins mirror -- so corner positions hold for fronts and backs alike.
    """
    return {
        corner: {
            "x_mm": format_mm(x),
            "y_mm": format_mm(y),
            "size_mm": format_mm(size),
        }
        for corner, (x, y, size, _) in config.qr_boxes().items()
    }


def build(config, pages: Sequence[Page]) -> Dict[str, Any]:
    """Assemble the manifest document."""
    width, height = config.size
    from .qrcodes import module_size, symbol_modules, worst_case_payload

    qr_section: Dict[str, Any] = {"enabled": config.qr.enabled and bool(config.qr.corners)}
    if qr_section["enabled"]:
        payload = worst_case_payload(config)
        qr_section.update(
            {
                "corners": list(config.qr.corners),
                "payload_template": config.qr.payload,
                "error_correction": config.qr.error_correction.upper(),
                "quiet_zone_modules": config.qr.quiet_zone,
                "modules": symbol_modules(
                    payload, config.qr.error_correction, config.qr.quiet_zone
                ),
                "module_size_mm": round(format_mm(module_size(config)), 3),
                "geometry": corner_geometry(config),
            }
        )

    page_entries: List[Dict[str, Any]] = []
    for page in pages:
        entry: Dict[str, Any] = {
            "page": page.number,
            "side": page.side,
            "token": page.token,
            "readable": page.ref.readable,
        }
        if qr_section["enabled"]:
            entry["codes"] = {
                corner: render_payload(config.qr.payload, page.ref.for_corner(corner), config.qr.token_format)
                for corner in config.qr.corners
            }
        page_entries.append(entry)

    return {
        "manifest_version": MANIFEST_VERSION,
        "generator": "paper-log",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "notebook": {
            "id": config.notebook_id,
            "title": config.title,
            "page_count": len(page_entries),
            "duplex": config.duplex,
        },
        "page": {
            "width_mm": format_mm(width),
            "height_mm": format_mm(height),
            "ruling": config.ruling.style,
            "line_spacing_mm": format_mm(config.ruling.spacing),
        },
        "token_format": {
            "version": FORMAT_VERSION,
            "grammar": f"{FORMAT_VERSION}:<notebook>:<page>:<F|B>:<TL|TR|BL|BR>:<crc16>",
            "checksum": "CRC-16/CCITT-FALSE over the token up to but excluding the final colon, hex, upper case",
            "alphabet": "Crockford base32 (no I, L, O, U)",
        },
        "imposition": config.imposition,
        "qr": qr_section,
        "pages": page_entries,
    }


def write(manifest: Dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path
