"""What the boxed requests actually do, and where the answers land.

A tool is a declaration, not code: a name, a contract written in prose, a model
to run it on, what shape the output takes, and where the file goes. Adding one
is writing a prompt, which is the part worth iterating on. Nothing here is
special-cased per tool.

**The isolation rule, restated where it is enforced.** A tool receives the
writer's prompt and nothing else. Not the page it came from, not the paragraph
around it, not the other calls in the same run. :mod:`paperlog.extract` cuts the
box out of the image so the surrounding prose never reaches the reader; this
module carries that through to the dispatch, so it never reaches the doer
either. The prompt is placed in the request inside a fence, as the writer's
words rather than as instructions to follow blindly -- the tool's own contract
is what says how to treat them.

**Because the writer is not there.** These run overnight, hours after the pen
was put down, and there is no way to ask a follow-up question. Every contract
below therefore says what to do with ambiguity: choose, say what you chose, and
carry on. Stopping to ask is the one thing that cannot work.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, fields
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

from .calls import Record, ToolCall
from .vision import DEFAULT_MODEL, Reply, VisionError, VisionSpec, backend_for


class ToolError(RuntimeError):
    """Raised when a tool is unknown, misconfigured, or fails to run."""


ASYNC_CONTRACT = """\
The person who wrote this request is not available. They wrote it by hand hours
ago and will read your answer later, on their own. You cannot ask them anything,
and a reply that asks a question instead of doing the work is a reply that
wasted the run.

So where the request is ambiguous, decide. Pick the reading a thoughtful
colleague would pick, say in one line at the top what you took it to mean and
what you assumed, and then do the work. If a different reading would have
produced something substantially different, note that in a line at the end.
Never refuse for want of detail."""

VERBATIM_NOTE = """\
The request below is transcribed from handwriting. Words the transcriber could
not read with confidence are marked [?likethis], or [?] where nothing could be
made out. Work around those gaps -- infer what you reasonably can from the rest,
and do not treat a marker as part of the request."""

INTERACTIVE_BREAK_PROMPT = f"""\
You build small self-contained interactive visualisations to be embedded in a
blog post.

{ASYNC_CONTRACT}

{VERBATIM_NOTE}

Output a single HTML file and nothing else. No prose before or after it, no
markdown fence. It must:

- be complete and standalone: one file, with all CSS in a <style> and all
  JavaScript in a <script>. No build step, no imports, no external scripts,
  stylesheets, fonts or images. No network requests of any kind -- it has to
  work offline and forever.
- be plain HTML and vanilla JavaScript. No React, no framework, no bundler.
- render at any width from a phone to a wide desktop, and sit inside a blog
  post's content column without escaping it.
- carry its own controls, labelled, with sensible starting values -- and any
  control the request asks for.
- run correctly on first paint without a click, and stay stable if a control is
  dragged to an extreme.
- use requestAnimationFrame for animation, and stop cleanly when the tab is
  hidden.
- be legible on a white background and a dark one; prefer a theme-neutral
  palette over assuming either.
- contain no analytics, no tracking, and no code that reaches outside itself."""

RESEARCH_REQUEST_PROMPT = f"""\
You are a research assistant producing a written report for someone who will
read it later, carefully, on their own.

{ASYNC_CONTRACT}

{VERBATIM_NOTE}

Output Markdown and nothing else. Structure it as:

- a title line
- one short paragraph saying what you understood the request to be, and what you
  assumed
- the report itself, organised under headings that suit the question rather than
  a fixed template
- a section listing sources and further reading, marking clearly which are
  primary sources and which are secondary
- a short closing section on what you could not establish, or where the record
  is thin or contested

Be concrete. Name works, people, dates and places rather than gesturing at them.
Where scholarship disagrees, say so and say who. Where you are unsure of a fact,
mark it as uncertain rather than stating it flatly -- a report that is confidently
wrong is worse than one that admits a gap. Do not pad."""


@dataclass
class Tool:
    """A tool the writer can call from the page."""

    name: str
    #: The contract. Everything the tool knows about how to do its job.
    system_prompt: str
    #: What comes back, which decides the file extension and how it is checked.
    output: str = "markdown"
    model: str = DEFAULT_MODEL
    effort: str = "high"
    max_tokens: int = 32000
    #: A one-line description, for `paperlog tools`.
    summary: str = ""
    #: Refuse to run if the estimated spend exceeds this. None means no ceiling.
    max_cost_usd: Optional[float] = None

    EXTENSIONS = {"markdown": ".md", "html": ".html"}

    def __post_init__(self) -> None:
        if self.output not in self.EXTENSIONS:
            raise ToolError(
                f"tool {self.name}: output must be one of "
                f"{', '.join(sorted(self.EXTENSIONS))}, got {self.output!r}"
            )

    @property
    def extension(self) -> str:
        return self.EXTENSIONS[self.output]

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> "Tool":
        known = {item.name for item in fields(cls)} | {"system_prompt_file"}
        unknown = set(data) - known - {"name"}
        if unknown:
            raise ToolError(
                f"tool {name}: unknown setting(s) {', '.join(sorted(unknown))}"
            )
        payload = {k: v for k, v in data.items() if k != "name"}
        path = payload.pop("system_prompt_file", None)
        if path is not None:
            if "system_prompt" in payload:
                raise ToolError(
                    f"tool {name}: set system_prompt or system_prompt_file, not both"
                )
            try:
                payload["system_prompt"] = (
                    Path(path).expanduser().read_text(encoding="utf-8")
                )
            except OSError as exc:
                raise ToolError(f"tool {name}: cannot read {path}: {exc}") from None
        if "system_prompt" not in payload:
            raise ToolError(f"tool {name}: needs a system_prompt or system_prompt_file")
        return cls(name=name, **payload)


BUILTIN_TOOLS: Dict[str, Tool] = {
    "interactive-break": Tool(
        name="interactive-break",
        summary="a self-contained interactive HTML widget to embed in a post",
        system_prompt=INTERACTIVE_BREAK_PROMPT,
        output="html",
        effort="xhigh",
    ),
    "research-request": Tool(
        name="research-request",
        summary="a written research report, with sources",
        system_prompt=RESEARCH_REQUEST_PROMPT,
        output="markdown",
        effort="high",
    ),
}


def load_tools(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Tool]:
    """The built-in tools, with any configured ones layered over them."""
    tools = dict(BUILTIN_TOOLS)
    for name, data in (overrides or {}).items():
        if not isinstance(data, dict):
            raise ToolError(f"tool {name}: expected a mapping")
        if name in tools and "system_prompt" not in data and "system_prompt_file" not in data:
            # Tweaking a built-in (a different model, a tighter budget) should
            # not mean restating its whole contract.
            merged = {
                "system_prompt": tools[name].system_prompt,
                "output": tools[name].output,
                "model": tools[name].model,
                "effort": tools[name].effort,
                "max_tokens": tools[name].max_tokens,
                "summary": tools[name].summary,
                **data,
            }
            tools[name] = Tool.from_dict(name, merged)
        else:
            tools[name] = Tool.from_dict(name, data)
    return tools


# --------------------------------------------------------------------------
# running one
# --------------------------------------------------------------------------


def _fence(prompt: str) -> str:
    """Hand the request over as the writer's words, clearly bounded.

    The tool's contract is the instruction; this is the material it acts on.
    Fencing keeps the boundary legible even when the request is itself phrased
    as a command, which it almost always is.
    """
    return (
        "Here is the request, transcribed from the writer's notebook. "
        "Everything between the markers is theirs.\n\n"
        "--- BEGIN REQUEST ---\n"
        f"{prompt.strip()}\n"
        "--- END REQUEST ---"
    )


def _slug(text: str, limit: int = 48) -> str:
    words = re.sub(r"[^\w\s-]+", " ", text.lower()).split()
    slug = "-".join(words)[:limit].strip("-")
    return slug or "untitled"


def destination(
    tool: Tool, call: ToolCall, root: Path, *, when: Optional[date] = None
) -> Path:
    """Where the answer goes.

    Named for the day, the tool and the request, with the call id on the end.
    The id is what makes it unambiguous which box on which page produced this
    file -- and stops two similar requests from overwriting each other.
    """
    stamp = (when or date.today()).isoformat()
    name = f"{stamp}-{_slug(call.prompt)}-{call.id[:8]}{tool.extension}"
    return Path(root).expanduser() / tool.name / name


@dataclass
class Result:
    """What a dispatched call produced."""

    record: Record
    text: str = ""
    path: Optional[Path] = None
    input_tokens: int = 0
    output_tokens: int = 0


def run_call(
    call: ToolCall,
    tool: Tool,
    *,
    backend=None,
    spec: Optional[VisionSpec] = None,
) -> Reply:
    """Send one request to its tool. The prompt goes; nothing else does."""
    spec = spec or VisionSpec(
        model=tool.model, effort=tool.effort, max_tokens=tool.max_tokens
    )
    backend = backend or backend_for(spec)
    try:
        reply = backend.ask(
            system=tool.system_prompt, instruction=_fence(call.prompt)
        )
    except VisionError as exc:
        raise ToolError(str(exc)) from None

    if reply.refused:
        raise ToolError(
            f"the model declined this {tool.name} request. The prompt is your "
            "own writing, so this is likely a false positive -- rerun it, or "
            "point the tool at a different model."
        )
    if not reply.text.strip():
        raise ToolError(
            "the model returned nothing"
            + (" (it ran out of output budget)" if reply.truncated else "")
        )
    return reply


def check_output(tool: Tool, text: str) -> List[str]:
    """Cheap sanity checks on what came back, as warnings not failures.

    A tool contract can be broken in ways that are obvious to look for and
    annoying to discover at review time -- an HTML widget that reaches out to a
    CDN will simply be blank on a plane. Warn; do not throw the work away.
    """
    notes: List[str] = []
    if tool.output == "html":
        lowered = text.lower()
        if "<script" not in lowered and "<canvas" not in lowered and "<svg" not in lowered:
            notes.append("no script, canvas or svg -- is this actually interactive?")
        for pattern, what in (
            (r"<script[^>]+\bsrc\s*=", "an external script"),
            (r"<link[^>]+\bhref\s*=\s*[\"']https?:", "an external stylesheet"),
            (r"\bfetch\s*\(", "a fetch() call"),
            (r"<img[^>]+\bsrc\s*=\s*[\"']https?:", "a remote image"),
        ):
            if re.search(pattern, lowered):
                notes.append(f"contains {what}; it was asked to be self-contained")
        if "```" in text:
            notes.append("contains a markdown fence; it was asked for bare HTML")
    return notes


def strip_fence(text: str) -> str:
    """Remove a markdown fence a model wrapped the file in anyway."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) < 2:
        return stripped
    body = lines[1:]
    if body and body[-1].strip().startswith("```"):
        body = body[:-1]
    return "\n".join(body).strip()
