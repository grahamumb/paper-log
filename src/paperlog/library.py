"""A small on-disk register of notebooks you have made.

The manifest is what turns a photograph back into a page: without it there is
no way to know where the codes sit on the paper. That makes "where did I put
the manifest" the weak point of the whole workflow, and it gets weaker with
time -- the notebook you are photographing today may have been printed a year
ago, from a directory you have since tidied away.

So building a notebook also files its manifest here, and ``paperlog scan``
looks here by default. The intended usage is that neither command needs to be
told anything:

    paperlog build --preset a5-dot --out journal.pdf
    ... print, write, photograph ...
    paperlog scan photos/

Nothing here is precious: the library is a directory of copies. Deleting it
loses nothing as long as the manifests written next to the PDFs survive, and
``paperlog notebooks --add`` puts them back.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

#: Overridable so tests -- and anyone who keeps their notes elsewhere -- do not
#: have to touch a real home directory.
HOME_VARIABLE = "PAPERLOG_HOME"


def home() -> Path:
    override = os.environ.get(HOME_VARIABLE)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".paperlog"


def notebooks_dir() -> Path:
    return home() / "notebooks"


@dataclass(frozen=True)
class Entry:
    """One notebook the library knows about."""

    notebook_id: str
    title: str
    pages: int
    created: str
    path: Path

    @property
    def label(self) -> str:
        return f"{self.notebook_id}  {self.title}" if self.title else self.notebook_id


def remember(manifest_path: Path) -> Optional[Path]:
    """File a copy of a manifest under its notebook id.

    Returns where it landed, or ``None`` if the file does not look like a
    manifest. Never raises for an unwritable home directory: failing to file a
    copy is not a reason to fail a build that already produced a PDF.
    """
    manifest_path = Path(manifest_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        notebook_id = str((manifest.get("notebook") or {})["id"]).upper()
    except (OSError, ValueError, KeyError, TypeError):
        return None

    try:
        target_dir = notebooks_dir()
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{notebook_id}.manifest.json"
        shutil.copyfile(manifest_path, target)
        return target
    except OSError:
        return None


def entries() -> List[Entry]:
    """Every notebook in the library, newest first."""
    folder = notebooks_dir()
    if not folder.is_dir():
        return []

    found: List[Entry] = []
    for path in folder.glob("*.manifest.json"):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        notebook = manifest.get("notebook") or {}
        if not notebook.get("id"):
            continue
        found.append(
            Entry(
                notebook_id=str(notebook["id"]).upper(),
                title=str(notebook.get("title") or ""),
                pages=int(notebook.get("page_count") or 0),
                created=str(manifest.get("created_utc") or ""),
                path=path,
            )
        )
    return sorted(found, key=lambda entry: entry.created, reverse=True)


def load() -> Dict[str, Dict]:
    """Every known manifest, keyed by notebook id."""
    out: Dict[str, Dict] = {}
    for entry in entries():
        try:
            out[entry.notebook_id] = json.loads(entry.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
    return out


def forget(notebook_id: str) -> bool:
    """Drop a notebook from the library. The PDF and its manifest are untouched."""
    target = notebooks_dir() / f"{notebook_id.upper()}.manifest.json"
    if target.exists():
        target.unlink()
        return True
    return False
