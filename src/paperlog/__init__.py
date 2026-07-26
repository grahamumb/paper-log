"""paper-log: printable journals whose pages identify themselves.

Typical use::

    from paperlog import JournalConfig, build

    config = JournalConfig.from_dict({"page_size": "a5", "pages": 64})
    result = build(config, "journal.pdf")
    print(result.manifest_path, len(result.pages))
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import (
    ConfigError,
    Fiducials,
    Furniture,
    JournalConfig,
    Margins,
    QrSpec,
    Ruling,
    load_config,
)
from .ids import PageRef, TokenError, decode, new_notebook_id
from .manifest import Page

__all__ = [
    "BuildResult",
    "ConfigError",
    "Fiducials",
    "Furniture",
    "JournalConfig",
    "Margins",
    "Page",
    "PageRef",
    "QrSpec",
    "Ruling",
    "TokenError",
    "build",
    "decode",
    "load_config",
    "new_notebook_id",
    "__version__",
]

__version__ = "0.1.0"


@dataclass
class BuildResult:
    pdf_path: Path
    manifest_path: Optional[Path]
    manifest: Dict[str, Any]
    pages: List[Page]
    warnings: List[str]


def build(
    config: JournalConfig,
    output: Path,
    *,
    manifest_path: Optional[Path] = None,
    write_manifest: bool = True,
) -> BuildResult:
    """Render ``config`` to ``output`` and write its manifest alongside.

    ``manifest_path`` defaults to the PDF path with a ``.manifest.json``
    suffix. Pass ``write_manifest=False`` to build the manifest in memory
    without touching the disk.
    """
    from . import manifest as manifest_module
    from .render import render

    output = Path(output)
    warnings = config.warnings()
    pages = render(config, output)
    document = manifest_module.build(config, pages)

    written: Optional[Path] = None
    if write_manifest:
        written = Path(manifest_path) if manifest_path else output.with_suffix(".manifest.json")
        manifest_module.write(document, written)

    return BuildResult(
        pdf_path=output,
        manifest_path=written,
        manifest=document,
        pages=pages,
        warnings=warnings,
    )
