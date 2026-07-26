"""Page identity: the tokens that go inside the QR codes.

A token names exactly one *corner* of exactly one *page* of exactly one
*notebook*. There are two spellings of the same four facts.

**Compact** (the default, and what gets printed)::

    PJ4M7QX0084K1
    |\----------/
    |     60 bits: notebook (30) | page (13) | corner (2) | side (1) | crc (14)
    marker

13 characters, which is what matters: it fits QR version 1 at error correction
level Q. That is a 21x21 symbol -- against 25x25 for the readable spelling --
so every module is ~16% larger in the same printed footprint, *and* a quarter
of the symbol can be destroyed and still decode. Both of those are the
difference between a code that survives a phone photo and one that does not.

**Readable**, for URL payloads and for debugging by eye::

    PL1:K7M2QX:42:F:TR:A19C
    |   |      |  | |  |
    |   |      |  | |  CRC-16/CCITT-FALSE of everything before it, hex
    |   |      |  | corner: TL / TR / BL / BR
    |   |      |  side: F (front/recto) or B (back/verso)
    |   |      page number within the notebook, 1-based
    |   notebook id, Crockford base32
    format version

:func:`decode` accepts either, and tells them apart on sight.

Design notes, all of which matter once you point a camera at the paper:

* Every character is in QR "alphanumeric" mode (digits, A-Z, and ``$%*+-./:``),
  so the symbol stays in the smallest version that will hold it. Byte mode
  would cost a whole version for the same content.
* The corner is *in* the token. A scanner that catches even one corner knows
  which corner it caught, and therefore the page's orientation -- upside-down
  and 90-degree-rotated photographs resolve without guessing. It is also what
  lets four codes anchor a perspective correction (see :mod:`paperlog.capture`).
* The CRC means a misread is detected rather than silently filed as some other
  page. QR has its own error correction, but the CRC also covers the case where
  a token is retyped, OCR'd, or truncated by a URL handler.
* Crockford base32 drops I, L, O and U, so notebook ids survive being read
  aloud or copied by hand off the cover.
"""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass
from typing import Iterable

FORMAT_VERSION = "PL1"

#: Crockford base32 -- no I, L, O, U, so it is unambiguous when handwritten.
ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

#: 6 characters is 30 bits: a billion notebooks, and short enough to write on
#: a cover. It is also what the compact token has room for.
NOTEBOOK_ID_LENGTH = 6

# -- compact token layout --------------------------------------------------
COMPACT_MARKER = "P"
NOTEBOOK_BITS = 30
PAGE_BITS = 13
CORNER_BITS = 2
SIDE_BITS = 1
CRC_BITS = 14
BODY_BITS = NOTEBOOK_BITS + PAGE_BITS + CORNER_BITS + SIDE_BITS  # 46
COMPACT_BITS = BODY_BITS + CRC_BITS  # 60, exactly 12 base32 characters
COMPACT_CHARS = COMPACT_BITS // 5
COMPACT_LENGTH = len(COMPACT_MARKER) + COMPACT_CHARS  # 13

MAX_NOTEBOOK_VALUE = (1 << NOTEBOOK_BITS) - 1
MAX_PAGE = (1 << PAGE_BITS) - 1

SIDES = ("F", "B")
CORNERS = ("TL", "TR", "BL", "BR")

CORNER_NAMES = {
    "TL": "top-left",
    "TR": "top-right",
    "BL": "bottom-left",
    "BR": "bottom-right",
}

_NOTEBOOK_RE = re.compile(f"^[{ALPHABET}]{{1,32}}$")
_COMPACT_RE = re.compile(f"^{COMPACT_MARKER}([{ALPHABET}]{{{COMPACT_CHARS}}})$")
_TOKEN_RE = re.compile(
    rf"^{FORMAT_VERSION}:([{ALPHABET}]{{1,32}}):(\d{{1,6}}):([FB]):(TL|TR|BL|BR):([0-9A-F]{{4}})$"
)


class TokenError(ValueError):
    """Raised when a token is malformed or fails its checksum."""


def crc16(data: bytes) -> int:
    """CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF, no reflection).

    Implemented directly rather than pulled from a dependency so the scanning
    side can be reimplemented in any language from this file alone.
    """
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def base32_encode(value: int, length: int) -> str:
    """Crockford base32, fixed width, most significant character first."""
    if value < 0:
        raise ValueError("cannot encode a negative value")
    out = []
    for shift in range(length - 1, -1, -1):
        out.append(ALPHABET[(value >> (5 * shift)) & 0x1F])
    return "".join(out)


def base32_decode(text: str) -> int:
    value = 0
    for char in text:
        index = ALPHABET.find(char)
        if index < 0:
            raise TokenError(f"{char!r} is not a Crockford base32 character")
        value = (value << 5) | index
    return value


def new_notebook_id(length: int = NOTEBOOK_ID_LENGTH) -> str:
    """Generate a random notebook id.

    6 characters of Crockford base32 is 30 bits -- a billion notebooks, which
    is plenty for one person, and short enough to write on a cover. It is also
    the width the compact token reserves.
    """
    if length < 1:
        raise ValueError("notebook id length must be at least 1")
    return "".join(secrets.choice(ALPHABET) for _ in range(length))


def normalise_notebook_id(value: str) -> str:
    """Upper-case a notebook id and map look-alike characters onto Crockford.

    ``I`` and ``L`` become ``1``, ``O`` becomes ``0``; this is the standard
    Crockford decoding leniency and makes hand-typed ids work.
    """
    cleaned = value.strip().upper().replace("-", "")
    cleaned = cleaned.translate(str.maketrans({"I": "1", "L": "1", "O": "0", "U": "V"}))
    if not _NOTEBOOK_RE.match(cleaned):
        raise TokenError(
            f"invalid notebook id {value!r}: expected 1-32 characters from {ALPHABET}"
        )
    return cleaned


@dataclass(frozen=True)
class PageRef:
    """One corner of one page."""

    notebook: str
    page: int
    side: str
    corner: str

    def __post_init__(self) -> None:
        if self.side not in SIDES:
            raise TokenError(f"side must be one of {SIDES}, got {self.side!r}")
        if self.corner not in CORNERS:
            raise TokenError(f"corner must be one of {CORNERS}, got {self.corner!r}")
        if not 0 < self.page < 1_000_000:
            raise TokenError(f"page must be between 1 and 999999, got {self.page}")
        object.__setattr__(self, "notebook", normalise_notebook_id(self.notebook))

    @property
    def body(self) -> str:
        """The token without its checksum."""
        return f"{FORMAT_VERSION}:{self.notebook}:{self.page}:{self.side}:{self.corner}"

    @property
    def readable(self) -> str:
        """The long, human-legible spelling, with its checksum."""
        return f"{self.body}:{crc16(self.body.encode('ascii')):04X}"

    @property
    def token(self) -> str:
        """The compact spelling -- what actually gets printed.

        13 characters, so the symbol stays at QR version 1 even at error
        correction level Q.
        """
        notebook_value = base32_decode(self.notebook)
        if notebook_value > MAX_NOTEBOOK_VALUE:
            raise TokenError(
                f"notebook id {self.notebook!r} needs more than {NOTEBOOK_BITS} bits, "
                f"so it will not fit a compact token. Use at most "
                f"{NOTEBOOK_BITS // 5} characters, or set token_format: readable."
            )
        if self.page > MAX_PAGE:
            raise TokenError(
                f"page {self.page} exceeds the {MAX_PAGE} a compact token can hold; "
                "use token_format: readable for a notebook this long"
            )
        body = (
            (notebook_value << (PAGE_BITS + CORNER_BITS + SIDE_BITS))
            | (self.page << (CORNER_BITS + SIDE_BITS))
            | (CORNERS.index(self.corner) << SIDE_BITS)
            | SIDES.index(self.side)
        )
        checksum = crc16(body.to_bytes(6, "big")) & ((1 << CRC_BITS) - 1)
        return COMPACT_MARKER + base32_encode((body << CRC_BITS) | checksum, COMPACT_CHARS)

    @property
    def corner_name(self) -> str:
        return CORNER_NAMES[self.corner]

    def for_corner(self, corner: str) -> "PageRef":
        return PageRef(self.notebook, self.page, self.side, corner)


def decode(token: str) -> PageRef:
    """Parse and verify a token, returning the page it names.

    Raises :class:`TokenError` on anything that is not an intact token. Accepts
    surrounding whitespace and lower case, and will pull the token out of a URL
    payload (the last path segment or the fragment), so the same function works
    whether the QR held a bare token or a link.
    """
    if not isinstance(token, str):
        raise TokenError(f"expected a string, got {type(token).__name__}")

    text = token.strip().upper()

    # A URL payload wraps the token in path segments, a fragment, or a query
    # string, so try the whole string first and then each piece of it. Both
    # spellings have to survive this: the compact one is bare characters with
    # no marker to search for, so the only way to find it is to look at the
    # segments.
    candidates = [text.rstrip("/")]
    if FORMAT_VERSION + ":" in text:
        tail = text[text.index(FORMAT_VERSION + ":") :]
        candidates.append(re.split(r"[?#&\s]", tail, maxsplit=1)[0].rstrip("/"))
    candidates.extend(
        piece for piece in re.split(r"[/?#&\s]+", text) if piece
    )

    for candidate in candidates:
        compact = _COMPACT_RE.match(candidate)
        if compact:
            return _decode_compact(candidate, compact.group(1))

    match = None
    for candidate in candidates:
        match = _TOKEN_RE.match(candidate)
        if match:
            break
    if not match:
        raise TokenError(
            f"not a paper-log token: {token!r} (expected a {COMPACT_LENGTH}-character "
            f"compact token, or {FORMAT_VERSION}:<notebook>:<page>:<F|B>:<corner>:<crc>)"
        )
    candidate = match.group(0)
    notebook, page, side, corner, checksum = match.groups()
    ref = PageRef(notebook=notebook, page=int(page), side=side, corner=corner)
    expected = f"{crc16(ref.body.encode('ascii')):04X}"
    if expected != checksum:
        raise TokenError(
            f"checksum mismatch on {candidate!r}: expected {expected}, read {checksum}. "
            "The code was misread or the token was edited."
        )
    return ref


def _decode_compact(candidate: str, payload: str) -> PageRef:
    value = base32_decode(payload)
    checksum = value & ((1 << CRC_BITS) - 1)
    body = value >> CRC_BITS
    expected = crc16(body.to_bytes(6, "big")) & ((1 << CRC_BITS) - 1)
    if checksum != expected:
        raise TokenError(
            f"checksum mismatch on {candidate!r}: expected {expected:04X}, "
            f"read {checksum:04X}. The code was misread or the token was edited."
        )

    side = SIDES[body & ((1 << SIDE_BITS) - 1)]
    corner = CORNERS[(body >> SIDE_BITS) & ((1 << CORNER_BITS) - 1)]
    page = (body >> (SIDE_BITS + CORNER_BITS)) & ((1 << PAGE_BITS) - 1)
    notebook_value = body >> (SIDE_BITS + CORNER_BITS + PAGE_BITS)
    if page == 0:
        raise TokenError(f"{candidate!r} decodes to page 0, which cannot exist")
    return PageRef(
        notebook=base32_encode(notebook_value, NOTEBOOK_ID_LENGTH),
        page=page,
        side=side,
        corner=corner,
    )


def encode(ref: PageRef, token_format: str = "compact") -> str:
    """The token for ``ref`` in the requested spelling."""
    if token_format == "compact":
        return ref.token
    if token_format == "readable":
        return ref.readable
    raise TokenError(f"unknown token_format {token_format!r}; use compact or readable")


def render_payload(template: str, ref: PageRef, token_format: str = "compact") -> str:
    """Build the string that actually goes into a QR symbol.

    ``template`` may be ``"{token}"`` (the default, and the smallest symbol) or
    something like ``"https://notes.example/p/{token}"`` for scanners that
    prefer to open a link. Available fields: ``token``, ``readable``,
    ``notebook``, ``page``, ``side``, ``corner``.
    """
    try:
        return template.format(
            token=encode(ref, token_format),
            readable=ref.readable,
            notebook=ref.notebook,
            page=ref.page,
            side=ref.side,
            corner=ref.corner,
        )
    except (KeyError, IndexError) as exc:
        raise TokenError(
            f"unknown placeholder {exc} in qr payload template {template!r}; "
            "available: {token} {readable} {notebook} {page} {side} {corner}"
        ) from None


def iter_pages(notebook: str, count: int, duplex: bool = True) -> Iterable[PageRef]:
    """Yield one :class:`PageRef` per page (top-left corner) for a notebook.

    With ``duplex`` the odd pages are fronts and the even pages are backs, which
    is what you get printing double-sided; otherwise every page is a front.
    """
    for number in range(1, count + 1):
        side = "F" if (not duplex or number % 2 == 1) else "B"
        yield PageRef(notebook=notebook, page=number, side=side, corner="TL")
