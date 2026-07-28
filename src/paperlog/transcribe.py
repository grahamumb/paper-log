"""Turn flattened pages into text.

This is the last step of the workflow: `scan` gives you a stack of PNGs named
after the page they came from, and this reads the handwriting on them.

**Why a vision model rather than an OCR engine.** Dedicated handwriting
recognition (Tesseract, TrOCR, the cloud OCR APIs) is trained on line images
with a known baseline and returns a flat string. It has no idea that the top
line of a paper-log page is a date, that a line beginning with a dash is a list
item, or that a word smudged beyond reading should be flagged rather than
guessed. A frontier vision model reads the page the way a person does, keeps the
structure, and can say when it is unsure -- which on handwriting matters more
than raw character accuracy, because the failure you cannot see is the one that
hurts.

**What it costs.** A page sent at the vision tier's ceiling runs to a few
thousand input tokens plus a short prompt and a short reply, which at Claude
Opus 5's $5/$25 per MTok puts a page in the region of two or three cents --
order of a few dollars for a whole notebook. Treat that as a sighting shot
rather than a quote: ``paperlog transcribe`` prints the tokens it actually
used and an estimate from them, so the first page you run is worth more than
any figure written here. The Batch API halves it if you are not waiting on the
result.

Needs the optional extra: ``pip install 'paper-log[transcribe]'``.
"""

from __future__ import annotations

import base64
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .ids import PageRef

#: Sent with every page. Written to describe the page rather than to plead: the
#: model is being asked to read, and the only real instructions are what to do
#: at the edges -- unreadable words, blank pages, the printed furniture.
SYSTEM_PROMPT = """\
You are transcribing a photographed page from a paper notebook.

The page has been flattened and de-skewed already, so it should be square-on.
It carries printed furniture that is NOT part of what was written: a QR code in
each corner, faint grey ruled lines, a page number, and a rule at the top with
the word "date" printed before a blank space. Do not transcribe any of that.
The handwriting is everything you should return.

Return the handwriting as Markdown, preserving the structure the writer used:
line breaks, indentation, list markers, headings, emphasis. Do not add
structure that is not there, and do not summarise, correct, tidy, or complete
anything -- transcribe what is on the paper, misspellings included.

If a word is genuinely unclear, give your best reading wrapped in brackets with
a question mark: [?word]. If you cannot read it at all, write [?]. Use these
sparingly and only where you are actually unsure; a transcription that silently
guesses is worse than one that admits a gap.

If the writer filled in the date line at the top, return it as the first line
in the form `date: <what they wrote>`, exactly as written, then a blank line,
then the body.

If the page has no handwriting on it at all, return exactly: (blank page)

Return only the transcription. No preamble, no commentary, no code fences."""

#: The long edge Claude's high-resolution tier accepts.
MAX_EDGE = 2576

#: And the total area, which is the limit that actually bites here: a 300dpi A5
#: page is 1748x2480, whose long edge is comfortably under 2576 but whose 4.34
#: megapixels are over. Resizing on this side as well means the page arrives at
#: a size we picked rather than one the API picked for us, and it is the
#: difference between a known downscale and a surprise one.
MAX_PIXELS = 3_750_000

#: Thinking is on by default on Opus 5 and shares this budget with the reply,
#: so it has to cover both. A dense page is well under 2000 tokens of text.
MAX_TOKENS = 8000

DEFAULT_MODEL = "claude-opus-5"

#: Transcription is reading, not reasoning. Low effort keeps the thinking spend
#: proportionate -- and is much preferred to turning thinking off, which on
#: Opus 5 can leak internal tags into the response.
DEFAULT_EFFORT = "low"

BLANK_MARKER = "(blank page)"

#: Matches the filenames `scan` writes, e.g. PAPER1-p0001F.png.
PAGE_FILENAME = re.compile(r"^(?P<notebook>[0-9A-Z]+)-p(?P<page>\d+)(?P<side>[FB])$")


class TranscribeError(RuntimeError):
    """Raised when a page cannot be transcribed."""


@dataclass
class Transcript:
    """One page, read."""

    source: Path
    text: str
    ref: Optional[PageRef] = None
    #: Words the model flagged as uncertain, in page order.
    unsure: List[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def blank(self) -> bool:
        return self.text.strip() == BLANK_MARKER

    @property
    def name(self) -> str:
        return self.source.stem

    @property
    def sort_key(self) -> Tuple:
        """Reading order: by notebook, then page, then front before back.

        Sorting on the side letter directly would put B before F, which is
        backwards. It rarely bites -- the side is derived from the page number,
        so two pages sharing a number is not a thing a built notebook produces
        -- but a stack of loose pages from elsewhere is exactly the case where
        the order is not already obvious.

        Pages whose filename carries no identity sort last, together, by name:
        they are the ones the reader will have to place by hand.
        """
        if self.ref is None:
            return (1, "", 0, 0, self.name)
        front_first = 0 if self.ref.side == "F" else 1
        return (0, self.ref.notebook, self.ref.page, front_first, self.name)


@dataclass
class Run:
    """Everything that came out of one `paperlog transcribe`."""

    transcripts: List[Transcript] = field(default_factory=list)
    failures: List[Tuple[Path, str]] = field(default_factory=list)

    @property
    def input_tokens(self) -> int:
        return sum(page.input_tokens for page in self.transcripts)

    @property
    def output_tokens(self) -> int:
        return sum(page.output_tokens for page in self.transcripts)

    def cost(self, model: str = DEFAULT_MODEL) -> Optional[float]:
        """Estimated US dollars, or None for a model we have no price for."""
        price = PRICES.get(model)
        if price is None:
            return None
        dollars_in, dollars_out = price
        return (
            self.input_tokens / 1_000_000 * dollars_in
            + self.output_tokens / 1_000_000 * dollars_out
        )


#: Dollars per million tokens, (input, output). Only used for the estimate
#: printed at the end, so an unknown model simply prints no estimate.
PRICES: Dict[str, Tuple[float, float]] = {
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


def _load_client(api_key: Optional[str] = None):
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - exercised by hand
        raise TranscribeError(
            "transcription needs the anthropic package: "
            "pip install 'paper-log[transcribe]'"
        ) from exc

    if api_key is None and not os.environ.get("ANTHROPIC_API_KEY"):
        raise TranscribeError(
            "no API key. Set ANTHROPIC_API_KEY in your environment -- there is "
            "deliberately no --api-key flag, because a key on the command line "
            "ends up in your shell history and in `ps`. Keys are at "
            "https://console.anthropic.com/settings/keys"
        )
    return anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()


def encode_page(
    path: Path, *, max_edge: int = MAX_EDGE, max_pixels: int = MAX_PIXELS
) -> Tuple[str, str]:
    """Read a page image and return (media_type, base64 data) ready to send.

    PNG rather than JPEG: the strokes are thin and high-contrast, which is
    exactly where JPEG puts its ringing, and a flattened page compresses well
    losslessly because most of it is flat white.
    """
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - exercised by hand
        raise TranscribeError(
            "transcription needs Pillow: pip install 'paper-log[transcribe]'"
        ) from exc
    import io

    try:
        image = Image.open(path)
        image.load()
    except Exception as exc:
        raise TranscribeError(f"cannot read {path.name}: {exc}") from exc

    if image.mode not in ("L", "RGB"):
        image = image.convert("L")

    # Whichever limit binds harder wins; both are satisfied by one resize.
    scale = min(
        max_edge / max(image.size),
        (max_pixels / (image.width * image.height)) ** 0.5,
        1.0,
    )
    if scale < 1.0:
        # Round down, not to nearest: rounding up on both axes can put the
        # result back over the area limit by a few hundred pixels, which is
        # exactly the silent server-side resize this is here to avoid.
        image = image.resize(
            (max(int(image.width * scale), 1), max(int(image.height * scale), 1)),
            Image.LANCZOS,
        )

    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return "image/png", base64.standard_b64encode(buffer.getvalue()).decode("ascii")


def _ref_from_name(path: Path) -> Optional[PageRef]:
    """Recover the page identity `scan` put in the filename."""
    match = PAGE_FILENAME.match(path.stem)
    if match is None:
        return None
    return PageRef(
        match.group("notebook"), int(match.group("page")), match.group("side"), "TL"
    )


def find_unsure(text: str) -> List[str]:
    """Pull out the [?word] markers, so a caller can review just those."""
    return [word.strip() for word in re.findall(r"\[\?([^\]]*)\]", text)]


def _text_of(message) -> str:
    parts = [
        block.text
        for block in message.content
        if getattr(block, "type", None) == "text" and getattr(block, "text", "")
    ]
    return "\n".join(parts).strip()


def transcribe_page(
    path: Path,
    client=None,
    *,
    model: str = DEFAULT_MODEL,
    effort: str = DEFAULT_EFFORT,
    context: str = "",
) -> Transcript:
    """Read one page image."""
    if client is None:
        client = _load_client()

    media_type, data = encode_page(path)
    instruction = "Transcribe the handwriting on this page."
    if context:
        instruction += (
            "\n\nContext that may help with proper nouns and jargon -- use it to "
            f"resolve ambiguity, never to add words that are not on the page:\n{context}"
        )

    message = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        output_config={"effort": effort},
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": data,
                        },
                    },
                    {"type": "text", "text": instruction},
                ],
            }
        ],
    )

    # Check the stop reason before touching content: a refusal comes back as a
    # perfectly ordinary 200 with nothing in it, and indexing blindly would
    # turn that into an IndexError three frames from anything meaningful.
    if getattr(message, "stop_reason", None) == "refusal":
        raise TranscribeError(
            "the model declined to transcribe this page. If it is an ordinary "
            "journal page this is a false positive -- try again, or transcribe "
            "it with --model claude-sonnet-5."
        )

    text = _text_of(message)
    if not text:
        raise TranscribeError(
            "the model returned nothing for this page"
            + (
                " (it ran out of output budget)"
                if getattr(message, "stop_reason", None) == "max_tokens"
                else ""
            )
        )

    usage = getattr(message, "usage", None)
    return Transcript(
        source=path,
        text=text,
        ref=_ref_from_name(path),
        unsure=find_unsure(text),
        input_tokens=getattr(usage, "input_tokens", 0) or 0,
        output_tokens=getattr(usage, "output_tokens", 0) or 0,
    )


def transcribe(
    paths: Sequence[Path],
    client=None,
    *,
    model: str = DEFAULT_MODEL,
    effort: str = DEFAULT_EFFORT,
    context: str = "",
    on_page=None,
) -> Run:
    """Read every page, in reading order.

    One page per request rather than a whole notebook per request. It costs the
    same -- pricing is per token, not per call -- and it means a page that fails
    loses one page rather than the batch, keeps each transcript anchored to the
    image it came from, and stops one page's handwriting from colouring the
    reading of the next.
    """
    if client is None:
        client = _load_client()

    run = Run()
    for path in paths:
        try:
            page = transcribe_page(
                path, client, model=model, effort=effort, context=context
            )
        except TranscribeError as exc:
            run.failures.append((path, str(exc)))
            if on_page is not None:
                on_page(path, None, str(exc))
            continue
        except Exception as exc:  # the SDK's own errors: network, auth, rate limit
            run.failures.append((path, f"{type(exc).__name__}: {exc}"))
            if on_page is not None:
                on_page(path, None, str(exc))
            continue
        run.transcripts.append(page)
        if on_page is not None:
            on_page(path, page, None)

    run.transcripts.sort(key=lambda page: page.sort_key)
    return run


def as_markdown(run: Run, *, skip_blank: bool = True) -> str:
    """One document for the whole notebook, in reading order."""
    chunks: List[str] = []
    for page in run.transcripts:
        if skip_blank and page.blank:
            continue
        heading = page.name
        if page.ref is not None:
            side = "front" if page.ref.side == "F" else "back"
            heading = f"{page.ref.notebook} page {page.ref.page} ({side})"
        chunks.append(f"## {heading}\n\n{page.text}")
    return "\n\n".join(chunks) + ("\n" if chunks else "")


def write_pages(run: Run, out_dir: Path) -> List[Path]:
    """Write one .md beside each page, named to match the image."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for page in run.transcripts:
        target = out_dir / f"{page.name}.md"
        target.write_text(page.text + "\n", encoding="utf-8")
        written.append(target)
    return written
