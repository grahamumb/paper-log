"""Lift the boxed requests out of a transcript.

The writer draws a box around a request. Recognising that box is the
transcriber's job -- a vision model sees a drawn rectangle easily, including a
faint one, a broken one, and one drawn over printed ruling -- so by the time the
page reaches this module it is already text, and the box has become a fenced
block:

    ```paperlog-tool interactive-break
    double pendulum viz, sliders for the angles and masses
    ```

Everything here is then a regular expression. No model reads the transcript to
decide what the request is, which matters more than it sounds.

**Why extraction is deterministic.** Someone has to read the page, and that
reader necessarily sees the whole of it. What must not happen is a model both
seeing the surrounding prose *and* deciding what the request says, because then
the paragraph starts shaping the request and authorial control slides quietly
from the writer to the machine. Splitting the two -- a transcriber that only
copies, an extractor that only matches -- means no model that shapes a request
has ever seen its context. The tool then receives the fenced text and nothing
else (see :mod:`paperlog.tools`).

The residual risk, stated rather than buried: the transcriber could let context
colour how it reads a word inside the box. That is far weaker than tailoring a
request, and the transcription prompt is explicit about copying verbatim, but it
is not nothing. It is the price of reading the box in the same pass as the page.

A typed marker works too, for when a box is inconvenient::

    TOOL research-request
    books on the black plague, primary sources
    END
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

from .calls import ToolCall
from .ids import PageRef

#: The info string the transcriber puts on a boxed region.
FENCE_INFO = "paperlog-tool"

UNKNOWN_TOOL = "?"

#: A fenced block the transcriber emitted for a boxed region. Tolerant about
#: the fence length and about a missing tool name, because a writer who boxed
#: something without naming a tool has still clearly asked for something and
#: should be told so rather than ignored.
_FENCED = re.compile(
    r"^(?P<fence>`{3,}|~{3,})[ \t]*" + re.escape(FENCE_INFO) + r"[ \t]*(?P<tool>[^\n`~]*)\n"
    r"(?P<body>.*?)"
    r"^(?P=fence)[ \t]*$",
    re.MULTILINE | re.DOTALL,
)

#: The typed fallback: a TOOL line, then the request, then END or a blank line.
#: Deliberately loose about the closing marker -- forgetting it is the obvious
#: mistake, and "to the end of the paragraph" is what a reader would assume.
_TYPED = re.compile(
    r"^[ \t]*TOOL[:\s-]+(?P<tool>[A-Za-z][\w-]*)[ \t]*\n"
    r"(?P<body>.*?)"
    r"(?=^[ \t]*END[ \t]*$|^[ \t]*$|\Z)",
    re.MULTILINE | re.DOTALL,
)


@dataclass(frozen=True)
class Extraction:
    """One request, lifted out of one transcript."""

    call: ToolCall
    #: Where in the transcript it was, so a reviewer can find it again.
    span: Tuple[int, int] = (0, 0)
    #: How it was written: a drawn box, or a typed marker.
    style: str = "box"

    @property
    def understood(self) -> bool:
        return self.call.tool != UNKNOWN_TOOL and bool(self.call.prompt.strip())


def _clean_tool(raw: str) -> str:
    name = raw.strip().strip("`'\"").lower()
    # A writer heading the box "TOOL research-request" may have that word
    # transcribed into the info string too.
    name = re.sub(r"^tool[:\s-]+", "", name)
    return name or UNKNOWN_TOOL


def find_calls(text: str, ref: PageRef) -> List[Extraction]:
    """Find every request in one page's transcript, in reading order."""
    seen: List[Tuple[int, int, str, str, str]] = []

    for match in _FENCED.finditer(text):
        seen.append(
            (
                match.start(),
                match.end(),
                _clean_tool(match.group("tool")),
                match.group("body").strip(),
                "box",
            )
        )

    for match in _TYPED.finditer(text):
        start, end = match.start(), match.end()
        # A typed marker inside a fenced block is the same request seen twice.
        if any(start >= s and end <= e for s, e, _, _, _ in seen):
            continue
        seen.append(
            (start, end, _clean_tool(match.group("tool")), match.group("body").strip(), "typed")
        )

    seen.sort(key=lambda item: item[0])
    out: List[Extraction] = []
    for ordinal, (start, end, tool, body, style) in enumerate(seen):
        out.append(
            Extraction(
                call=ToolCall(
                    tool=tool,
                    prompt=body,
                    notebook=ref.notebook,
                    page=ref.page,
                    side=ref.side,
                    ordinal=ordinal,
                ),
                span=(start, end),
                style=style,
            )
        )
    return out


def strip_calls(text: str) -> str:
    """The prose with the requests removed.

    What you want when assembling a draft: the writing, with the machinery of
    asking for things taken back out.
    """
    without = _FENCED.sub("", text)
    without = _TYPED.sub("", without)
    # The typed pattern stops *before* its END so that a forgotten one still
    # closes the request; that leaves the marker behind when one was written.
    without = re.sub(r"^[ \t]*END[ \t]*$\n?", "", without, flags=re.MULTILINE)
    return re.sub(r"\n{3,}", "\n\n", without).strip()
