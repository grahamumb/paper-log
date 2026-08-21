"""Tool calls written on paper: their identity, and the ledger of what has run.

A call is a request the writer made by drawing a box around it. Between the
paper and the answer there are two jobs this module does.

**Identity.** You will photograph the same page more than once -- `scan` is
built to prefer a better shot of a page it already has -- and every reshoot
presents the same drawn box again. Without a stable identity, each one fires the
request again: duplicate reports, duplicate drafts, duplicate bills. The id is a
hash of where the call came from and what it asks, so the same box on the same
page is the same call however many times it is photographed. The prompt is
normalised before hashing, so a transcription that differs by a comma is still
the same call, while a genuine rewrite is a new one.

**Status.** The ledger is an append-only log of what has been dispatched and
what came back. It answers "what is still outstanding", survives a crash
mid-run, and makes a retry cheap and a re-run free.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

LEDGER_FILENAME = "calls.jsonl"

PENDING = "pending"
DONE = "done"
FAILED = "failed"
STATUSES = (PENDING, DONE, FAILED)


class CallError(RuntimeError):
    """Raised when a call cannot be identified, recorded or read back."""


def normalise_prompt(text: str) -> str:
    """Reduce a prompt to what it asks, so trivia does not change its identity.

    Handwriting transcribed twice will differ: a comma, a line break, a capital.
    Those are not a different request. Rewording it is.
    """
    lowered = text.strip().lower()
    lowered = re.sub(r"[^\w\s]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


@dataclass(frozen=True)
class ToolCall:
    """One boxed request, lifted off one page."""

    tool: str
    prompt: str
    notebook: str
    page: int
    side: str = "F"
    #: Which box on the page, in reading order. Distinguishes two calls to the
    #: same tool with the same words on one page.
    ordinal: int = 0
    #: Where on the page it was, kept so a reviewer can find it again.
    region: Optional[List[int]] = None

    @property
    def id(self) -> str:
        """Stable across reshoots, distinct across genuine edits."""
        material = "␟".join(
            [
                self.notebook.upper(),
                str(self.page),
                self.side.upper(),
                str(self.ordinal),
                self.tool.strip().lower(),
                normalise_prompt(self.prompt),
            ]
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]

    @property
    def origin(self) -> str:
        return f"{self.notebook}-p{self.page:04d}{self.side}#{self.ordinal}"


@dataclass
class Record:
    """A ledger row: one call, and what became of it."""

    id: str
    tool: str
    prompt: str
    notebook: str
    page: int
    side: str
    ordinal: int
    status: str = PENDING
    output_path: Optional[str] = None
    error: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: Optional[float] = None
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    region: Optional[List[int]] = None

    @classmethod
    def of(cls, call: ToolCall, **kwargs) -> "Record":
        return cls(
            id=call.id,
            tool=call.tool,
            prompt=call.prompt,
            notebook=call.notebook,
            page=call.page,
            side=call.side,
            ordinal=call.ordinal,
            region=call.region,
            **kwargs,
        )

    @property
    def call(self) -> ToolCall:
        return ToolCall(
            tool=self.tool,
            prompt=self.prompt,
            notebook=self.notebook,
            page=self.page,
            side=self.side,
            ordinal=self.ordinal,
            region=self.region,
        )


class Ledger:
    """An append-only JSONL log of calls, latest row per id wins.

    Append-only rather than rewritten in place: a crash halfway through leaves a
    readable file rather than a truncated one, and the history of a call --
    seen, dispatched, failed, retried, done -- is worth keeping when something
    goes wrong overnight and you are reading it in the morning.
    """

    def __init__(self, path: Path):
        self.path = Path(path)

    @classmethod
    def default(cls) -> "Ledger":
        from .library import home

        return cls(home() / LEDGER_FILENAME)

    def _rows(self) -> Iterable[Dict]:
        if not self.path.exists():
            return []
        rows = []
        for number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # One corrupt line should not cost you the ledger. A partial
                # write at the end is the usual cause, and the rest is fine.
                continue
            if isinstance(row, dict) and row.get("id"):
                rows.append(row)
        return rows

    def records(self) -> List[Record]:
        """Every call, latest state first written wins per id, in order seen."""
        latest: Dict[str, Dict] = {}
        order: List[str] = []
        for row in self._rows():
            if row["id"] not in latest:
                order.append(row["id"])
            latest[row["id"]] = row
        out = []
        for call_id in order:
            row = dict(latest[call_id])
            known = {f for f in Record.__dataclass_fields__}
            out.append(Record(**{k: v for k, v in row.items() if k in known}))
        return out

    def by_id(self) -> Dict[str, Record]:
        return {record.id: record for record in self.records()}

    def append(self, record: Record) -> Record:
        record.updated = time.time()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(asdict(record), ensure_ascii=False, sort_keys=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return record

    def note_seen(self, calls: Iterable[ToolCall]) -> List[Record]:
        """Record calls found on paper, skipping any already known.

        This is the idempotency gate: a call seen for the second time -- because
        the page was photographed again -- is not added and not re-run.
        """
        known = self.by_id()
        added = []
        for call in calls:
            if call.id in known:
                continue
            added.append(self.append(Record.of(call)))
        return added

    def pending(self, tool: Optional[str] = None) -> List[Record]:
        return [
            record
            for record in self.records()
            if record.status == PENDING and (tool is None or record.tool == tool)
        ]
