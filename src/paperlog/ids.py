"""Page identity: the tokens that go inside the QR codes.

A token is a short, self-describing, self-validating string that names exactly
one *corner* of exactly one *page* of exactly one *notebook*::

    PL1:K7M2QX4A:42:F:TR:A19C
    |   |        |  | |  |
    |   |        |  | |  CRC-16/CCITT-FALSE of everything before it, hex
    |   |        |  | corner: TL / TR / BL / BR
    |   |        |  side: F (front/recto) or B (back/verso)
    |   |        page number within the notebook, 1-based
    |   notebook id, 8 Crockford base32 chars
    format version

Design notes, all of which matter once you point a camera at the paper:

* Every character is in QR "alphanumeric" mode (digits, A-Z, and ``$%*+-./:``),
  so the symbol stays small and low-density -- readable from a phone held over
  a page, and forgiving of a mediocre flatbed scan.
* The corner is *in* the token. A scanner that catches even one corner knows
  which corner it caught, and therefore the page's orientation -- upside-down
  and 90-degree-rotated scans deskew without guessing.
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

NOTEBOOK_ID_LENGTH = 8

SIDES = ("F", "B")
CORNERS = ("TL", "TR", "BL", "BR")

CORNER_NAMES = {
    "TL": "top-left",
    "TR": "top-right",
    "BL": "bottom-left",
    "BR": "bottom-right",
}

_NOTEBOOK_RE = re.compile(f"^[{ALPHABET}]{{1,32}}$")
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


def new_notebook_id(length: int = NOTEBOOK_ID_LENGTH) -> str:
    """Generate a random notebook id.

    8 characters of Crockford base32 is 40 bits: enough that two notebooks
    printed by the same person will not collide, short enough to write on a
    cover.
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
    def token(self) -> str:
        """The full, checksummed token."""
        return f"{self.body}:{crc16(self.body.encode('ascii')):04X}"

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

    candidate = token.strip().upper()
    if FORMAT_VERSION + ":" in candidate and not candidate.startswith(FORMAT_VERSION + ":"):
        # A URL payload: keep everything from the version marker onwards, then
        # trim any trailing query string the scanner app may have appended.
        candidate = candidate[candidate.index(FORMAT_VERSION + ":") :]
        candidate = re.split(r"[?#&\s]", candidate, maxsplit=1)[0]
    candidate = candidate.rstrip("/")

    match = _TOKEN_RE.match(candidate)
    if not match:
        raise TokenError(
            f"not a {FORMAT_VERSION} token: {token!r} "
            f"(expected {FORMAT_VERSION}:<notebook>:<page>:<F|B>:<corner>:<crc>)"
        )
    notebook, page, side, corner, checksum = match.groups()
    ref = PageRef(notebook=notebook, page=int(page), side=side, corner=corner)
    expected = f"{crc16(ref.body.encode('ascii')):04X}"
    if expected != checksum:
        raise TokenError(
            f"checksum mismatch on {candidate!r}: expected {expected}, read {checksum}. "
            "The code was misread or the token was edited."
        )
    return ref


def render_payload(template: str, ref: PageRef) -> str:
    """Build the string that actually goes into a QR symbol.

    ``template`` may be ``"{token}"`` (the default, and the most compact) or
    something like ``"https://notes.example/p/{token}"`` for scanners that
    prefer to open a link. Available fields: ``token``, ``notebook``, ``page``,
    ``side``, ``corner``, ``crc``.
    """
    try:
        return template.format(
            token=ref.token,
            notebook=ref.notebook,
            page=ref.page,
            side=ref.side,
            corner=ref.corner,
            crc=ref.token.rsplit(":", 1)[1],
        )
    except (KeyError, IndexError) as exc:
        raise TokenError(
            f"unknown placeholder {exc} in qr payload template {template!r}; "
            "available: {token} {notebook} {page} {side} {corner} {crc}"
        ) from None


def iter_pages(notebook: str, count: int, duplex: bool = True) -> Iterable[PageRef]:
    """Yield one :class:`PageRef` per page (top-left corner) for a notebook.

    With ``duplex`` the odd pages are fronts and the even pages are backs, which
    is what you get printing double-sided; otherwise every page is a front.
    """
    for number in range(1, count + 1):
        side = "F" if (not duplex or number % 2 == 1) else "B"
        yield PageRef(notebook=notebook, page=number, side=side, corner="TL")
