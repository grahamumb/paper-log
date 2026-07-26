import pytest

from paperlog.ids import (
    ALPHABET,
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


def test_token_shape():
    ref = PageRef("K7M2QX4A", 42, "F", "TR")
    assert ref.token.startswith("PL1:K7M2QX4A:42:F:TR:")
    assert len(ref.token.rsplit(":", 1)[1]) == 4


def test_round_trip():
    for page in (1, 9, 10, 99, 100, 1000, 999999):
        for side in ("F", "B"):
            for corner in ("TL", "TR", "BL", "BR"):
                ref = PageRef("K7M2QX4A", page, side, corner)
                assert decode(ref.token) == ref


def test_tokens_are_alphanumeric_qr_safe():
    """QR alphanumeric mode covers 0-9 A-Z and $%*+-./: -- staying inside it
    keeps the symbol a version smaller than byte mode would."""
    allowed = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:")
    token = PageRef("K7M2QX4A", 128, "B", "BL").token
    assert set(token) <= allowed


def test_corrupted_token_is_rejected():
    token = PageRef("K7M2QX4A", 42, "F", "TR").token
    body, checksum = token.rsplit(":", 1)
    # A single-digit page misread is exactly the failure the CRC exists for.
    tampered = body.replace(":42:", ":43:") + ":" + checksum
    with pytest.raises(TokenError, match="checksum"):
        decode(tampered)


def test_truncated_and_junk_tokens_are_rejected():
    for bad in ("", "hello", "PL1:K7M2QX4A:42:F:TR", "PL1:K7M2QX4A:42:X:TR:0000", "PL2:A:1:F:TL:0000"):
        with pytest.raises(TokenError):
            decode(bad)


def test_decode_is_case_and_whitespace_insensitive():
    ref = PageRef("K7M2QX4A", 7, "B", "BR")
    assert decode(f"  {ref.token.lower()}\n") == ref


def test_decode_extracts_token_from_url_payload():
    ref = PageRef("K7M2QX4A", 7, "F", "TL")
    for payload in (
        f"https://notes.example/p/{ref.token}",
        f"https://notes.example/p/{ref.token}/",
        f"https://notes.example/p/{ref.token}?src=camera",
        f"https://notes.example/#{ref.token}",
    ):
        assert decode(payload) == ref


def test_notebook_id_normalisation_maps_lookalikes():
    # Crockford leniency: what someone reads off a cover still resolves.
    assert normalise_notebook_id("k7m2qx4a") == "K7M2QX4A"
    assert normalise_notebook_id("IL0O") == "1100"
    with pytest.raises(TokenError):
        normalise_notebook_id("has spaces")


def test_new_notebook_id_uses_the_alphabet():
    for _ in range(20):
        value = new_notebook_id()
        assert len(value) == 8
        assert set(value) <= set(ALPHABET)


def test_invalid_components_are_rejected():
    with pytest.raises(TokenError):
        PageRef("K7M2QX4A", 0, "F", "TL")
    with pytest.raises(TokenError):
        PageRef("K7M2QX4A", 1, "X", "TL")
    with pytest.raises(TokenError):
        PageRef("K7M2QX4A", 1, "F", "MIDDLE")


def test_iter_pages_alternates_sides_when_duplex():
    pages = list(iter_pages("K7M2QX4A", 4, duplex=True))
    assert [page.side for page in pages] == ["F", "B", "F", "B"]
    assert [page.side for page in iter_pages("K7M2QX4A", 3, duplex=False)] == ["F"] * 3


def test_render_payload_templates():
    ref = PageRef("K7M2QX4A", 12, "F", "TL")
    assert render_payload("{token}", ref) == ref.token
    assert render_payload("https://x.test/{notebook}/{page}", ref) == "https://x.test/K7M2QX4A/12"
    with pytest.raises(TokenError, match="unknown placeholder"):
        render_payload("{nope}", ref)
