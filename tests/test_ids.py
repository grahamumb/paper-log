import random

import pytest

from paperlog.qrcodes import build_matrix
from paperlog.ids import (
    ALPHABET,
    COMPACT_LENGTH,
    MAX_PAGE,
    NOTEBOOK_ID_LENGTH,
    PageRef,
    TokenError,
    crc16,
    decode,
    iter_pages,
    new_notebook_id,
    normalise_notebook_id,
    render_payload,
)


def test_crc16_known_vector():
    # The canonical CRC-16/CCITT-FALSE check value for "123456789".
    assert crc16(b"123456789") == 0x29B1


def test_compact_token_shape():
    """13 characters is the whole point: it keeps the symbol at QR version 1."""
    ref = PageRef("K7M2QX", 42, "F", "TR")
    assert len(ref.token) == COMPACT_LENGTH == 13
    assert ref.token.startswith("P")
    assert set(ref.token[1:]) <= set(ALPHABET)


def test_readable_token_shape():
    ref = PageRef("K7M2QX", 42, "F", "TR")
    assert ref.readable.startswith("PL1:K7M2QX:42:F:TR:")
    assert len(ref.readable.rsplit(":", 1)[1]) == 4


def test_compact_token_is_smaller_than_readable():
    """The compact spelling has to buy a whole QR version, or it is pointless."""
    ref = PageRef("K7M2QX", 42, "F", "TR")
    assert len(build_matrix(ref.token, "q")) == 21
    assert len(build_matrix(ref.readable, "q")) == 25


@pytest.mark.parametrize("spelling", ["token", "readable"])
def test_round_trip(spelling):
    for page in (1, 9, 10, 99, 100, 1000, MAX_PAGE):
        for side in ("F", "B"):
            for corner in ("TL", "TR", "BL", "BR"):
                ref = PageRef("K7M2QX", page, side, corner)
                assert decode(getattr(ref, spelling)) == ref


def test_random_round_trips():
    """Bit packing is the kind of thing that works for the case you tried."""
    rng = random.Random(0)
    for _ in range(2000):
        ref = PageRef(
            notebook="".join(rng.choice(ALPHABET) for _ in range(6)),
            page=rng.randint(1, MAX_PAGE),
            side=rng.choice("FB"),
            corner=rng.choice(["TL", "TR", "BL", "BR"]),
        )
        assert decode(ref.token) == ref


def test_compact_token_refuses_what_it_cannot_hold():
    """Better a clear error than a token that silently loses the page number."""
    with pytest.raises(TokenError, match="compact"):
        PageRef("K7M2QX", MAX_PAGE + 1, "F", "TL").token
    with pytest.raises(TokenError, match="compact"):
        PageRef("K7M2QXTOOLONG", 1, "F", "TL").token
    # The readable spelling has no such limit.
    assert decode(PageRef("K7M2QXTOOLONG", 99999, "F", "TL").readable).page == 99999


def test_tokens_are_alphanumeric_qr_safe():
    """QR alphanumeric mode covers 0-9 A-Z and $%*+-./: -- staying inside it
    keeps the symbol a version smaller than byte mode would."""
    allowed = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:")
    ref = PageRef("K7M2QX", 128, "B", "BL")
    assert set(ref.token) <= allowed
    assert set(ref.readable) <= allowed


def test_corrupted_readable_token_is_rejected():
    token = PageRef("K7M2QX", 42, "F", "TR").readable
    body, checksum = token.rsplit(":", 1)
    # A single-digit page misread is exactly the failure the CRC exists for.
    tampered = body.replace(":42:", ":43:") + ":" + checksum
    with pytest.raises(TokenError, match="checksum"):
        decode(tampered)


def test_every_single_character_corruption_is_caught():
    """A compact token is opaque, so the CRC is the only thing standing between
    a misread and a page filed under the wrong number."""
    token = PageRef("K7M2QX", 42, "F", "TR").token
    escaped = 0
    for index in range(1, len(token)):
        for replacement in ALPHABET:
            if replacement == token[index]:
                continue
            broken = token[:index] + replacement + token[index + 1 :]
            try:
                decode(broken)
            except TokenError:
                continue
            escaped += 1
    assert escaped == 0


def test_truncated_and_junk_tokens_are_rejected():
    for bad in ("", "hello", "PL1:K7M2QX:42:F:TR", "PL1:K7M2QX:42:X:TR:0000", "PL2:A:1:F:TL:0000"):
        with pytest.raises(TokenError):
            decode(bad)


def test_decode_is_case_and_whitespace_insensitive():
    ref = PageRef("K7M2QX", 7, "B", "BR")
    assert decode(f"  {ref.token.lower()}\n") == ref


def test_decode_extracts_token_from_url_payload():
    ref = PageRef("K7M2QX", 7, "F", "TL")
    for payload in (
        f"https://notes.example/p/{ref.token}",
        f"https://notes.example/p/{ref.token}/",
        f"https://notes.example/p/{ref.token}?src=camera",
        f"https://notes.example/#{ref.token}",
    ):
        assert decode(payload) == ref


def test_notebook_id_normalisation_maps_lookalikes():
    # Crockford leniency: what someone reads off a cover still resolves.
    assert normalise_notebook_id("k7m2qx") == "K7M2QX"
    assert normalise_notebook_id("IL0O") == "1100"
    with pytest.raises(TokenError):
        normalise_notebook_id("has spaces")


def test_new_notebook_id_uses_the_alphabet():
    for _ in range(20):
        value = new_notebook_id()
        assert len(value) == NOTEBOOK_ID_LENGTH == 6
        assert set(value) <= set(ALPHABET)


def test_invalid_components_are_rejected():
    with pytest.raises(TokenError):
        PageRef("K7M2QX", 0, "F", "TL")
    with pytest.raises(TokenError):
        PageRef("K7M2QX", 1, "X", "TL")
    with pytest.raises(TokenError):
        PageRef("K7M2QX", 1, "F", "MIDDLE")


def test_iter_pages_alternates_sides_when_duplex():
    pages = list(iter_pages("K7M2QX", 4, duplex=True))
    assert [page.side for page in pages] == ["F", "B", "F", "B"]
    assert [page.side for page in iter_pages("K7M2QX", 3, duplex=False)] == ["F"] * 3


def test_render_payload_templates():
    ref = PageRef("K7M2QX", 12, "F", "TL")
    assert render_payload("{token}", ref) == ref.token
    assert render_payload("https://x.test/{notebook}/{page}", ref) == "https://x.test/K7M2QX/12"
    with pytest.raises(TokenError, match="unknown placeholder"):
        render_payload("{nope}", ref)
