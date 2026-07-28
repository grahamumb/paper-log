"""Handwriting to text.

Every test here runs against a stand-in for the Anthropic client. That is a
real limitation and worth naming: these tests prove the request we build, the
response handling, and the file output, but they cannot prove the model reads
handwriting well. Only a page you wrote yourself can do that.

What they *can* catch is the whole class of bugs that would otherwise only show
up as a 400 from the API after you have already paid for the pages before it.
"""

import base64
import io

import pytest

from paperlog.cli import main
from paperlog.transcribe import (
    BLANK_MARKER,
    MAX_EDGE,
    MAX_PIXELS,
    SYSTEM_PROMPT,
    TranscribeError,
    as_markdown,
    encode_page,
    find_unsure,
    transcribe,
    transcribe_page,
)

PIL = pytest.importorskip("PIL", reason="needs the [transcribe] extras")
from PIL import Image  # noqa: E402


# -- a stand-in for anthropic.Anthropic -------------------------------------


class FakeBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class FakeUsage:
    def __init__(self, input_tokens=1000, output_tokens=100):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class FakeMessage:
    def __init__(self, text="", stop_reason="end_turn", blocks=None):
        self.content = blocks if blocks is not None else [FakeBlock(text)]
        self.stop_reason = stop_reason
        self.usage = FakeUsage()


class FakeClient:
    """Records what it was asked for and replays canned answers."""

    def __init__(self, replies=None):
        self.calls = []
        self._replies = list(replies) if replies else []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        reply = self._replies.pop(0) if self._replies else FakeMessage("hello")
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture
def page(tmp_path):
    """An image shaped like something `scan` would have written."""
    path = tmp_path / "K7M2QX-p0003F.png"
    Image.new("L", (1748, 2480), 255).save(path)
    return path


# -- the request we build ---------------------------------------------------


def test_the_page_is_sent_as_an_image_with_an_instruction(page):
    client = FakeClient()
    transcribe_page(page, client)

    (call,) = client.calls
    assert call["model"] == "claude-opus-5"
    assert call["system"] == SYSTEM_PROMPT
    content = call["messages"][0]["content"]
    assert [block["type"] for block in content] == ["image", "text"]
    assert content[0]["source"]["type"] == "base64"
    assert content[0]["source"]["media_type"] == "image/png"
    # The payload is real base64 of a real PNG, not a path or a placeholder.
    decoded = base64.standard_b64decode(content[0]["source"]["data"])
    sent = Image.open(io.BytesIO(decoded))
    assert sent.width * sent.height <= MAX_PIXELS
    assert sent.size[0] / sent.size[1] == pytest.approx(1748 / 2480, abs=0.01)


def test_thinking_is_left_on_but_kept_cheap(page):
    """Effort rather than disabled thinking, deliberately.

    Transcription needs almost no reasoning, so the tempting setting is
    ``thinking: {"type": "disabled"}``. On Opus 5 that is the setting that can
    leak internal tags into the response -- which here would land in the middle
    of someone's journal. Low effort costs a little more and cannot do that.
    """
    client = FakeClient()
    transcribe_page(page, client)

    (call,) = client.calls
    assert call["output_config"] == {"effort": "low"}
    assert "thinking" not in call
    assert call["max_tokens"] >= 4000  # shared between thinking and the reply


def test_context_is_passed_but_fenced(page):
    client = FakeClient()
    transcribe_page(page, client, context="Kowalczyk, Bräuer, tetrahydrofuran")

    text = client.calls[0]["messages"][0]["content"][1]["text"]
    assert "tetrahydrofuran" in text
    assert "never to add words that are not on the page" in text


def test_no_context_means_no_empty_context_section(page):
    client = FakeClient()
    transcribe_page(page, client)
    text = client.calls[0]["messages"][0]["content"][1]["text"]
    assert "Context" not in text


# -- image encoding ---------------------------------------------------------


def _decode(data):
    return Image.open(io.BytesIO(base64.standard_b64decode(data)))


def test_an_oversized_page_is_shrunk_to_fit_the_vision_tier(tmp_path):
    """A 600dpi scan is past what the vision tier accepts.

    Resizing here rather than letting the API do it means the page arrives at a
    size we chose, with an aspect ratio that is ours rather than a guess.
    """
    path = tmp_path / "big.png"
    Image.new("L", (4000, 5600), 255).save(path)

    resized = _decode(encode_page(path)[1])
    assert max(resized.size) <= MAX_EDGE
    assert resized.width * resized.height <= MAX_PIXELS
    assert resized.size[0] / resized.size[1] == pytest.approx(4000 / 5600, abs=0.01)


def test_the_area_limit_binds_on_an_ordinary_a5_page(page):
    """The limit that actually applies to paper-log's own output.

    A 300dpi A5 page is 1748x2480: its long edge is well inside 2576, so a
    long-edge-only check would pass it through at 4.34 megapixels and let the
    API resize it to something unchosen.
    """
    assert 1748 * 2480 > MAX_PIXELS  # the premise, in case the tier changes
    resized = _decode(encode_page(page)[1])
    assert max(resized.size) < MAX_EDGE
    assert resized.width * resized.height <= MAX_PIXELS
    assert resized.size[0] / resized.size[1] == pytest.approx(1748 / 2480, abs=0.01)


def test_a_page_within_both_limits_is_untouched(tmp_path):
    path = tmp_path / "small.png"
    Image.new("L", (1200, 1700), 255).save(path)
    assert _decode(encode_page(path)[1]).size == (1200, 1700)


def test_an_unreadable_image_is_reported_not_raised_raw(tmp_path):
    junk = tmp_path / "notreally.png"
    junk.write_text("this is not a png")
    with pytest.raises(TranscribeError, match="cannot read"):
        encode_page(junk)


# -- reading the response ---------------------------------------------------


def test_the_page_identity_comes_from_the_filename(page):
    """`scan` already worked out which page this is; don't ask the model."""
    result = transcribe_page(page, FakeClient([FakeMessage("Some notes.")]))
    assert result.ref is not None
    assert (result.ref.notebook, result.ref.page, result.ref.side) == ("K7M2QX", 3, "F")


def test_an_unrecognised_filename_still_transcribes(tmp_path):
    path = tmp_path / "holiday-snap.png"
    Image.new("L", (800, 1000), 255).save(path)
    result = transcribe_page(path, FakeClient([FakeMessage("words")]))
    assert result.ref is None
    assert result.text == "words"


def test_uncertain_words_are_pulled_out_for_review():
    assert find_unsure("met [?Aoife] about the [?] budget") == ["Aoife", ""]
    assert find_unsure("nothing unclear here") == []


def test_a_blank_page_is_recognised(page):
    result = transcribe_page(page, FakeClient([FakeMessage(BLANK_MARKER)]))
    assert result.blank


def test_a_refusal_is_explained_rather_than_crashing(page):
    """A refusal is a 200 with empty content.

    Reading content[0] unconditionally would turn a policy decision into an
    IndexError with no hint about what went wrong.
    """
    refusal = FakeMessage(stop_reason="refusal", blocks=[])
    with pytest.raises(TranscribeError, match="declined"):
        transcribe_page(page, FakeClient([refusal]))


def test_running_out_of_output_budget_says_so(page):
    truncated = FakeMessage(stop_reason="max_tokens", blocks=[])
    with pytest.raises(TranscribeError, match="output budget"):
        transcribe_page(page, FakeClient([truncated]))


def test_usage_is_carried_through_for_the_cost_estimate(page):
    result = transcribe_page(page, FakeClient([FakeMessage("hi")]))
    assert result.input_tokens == 1000
    assert result.output_tokens == 100


# -- a run of pages ---------------------------------------------------------


@pytest.fixture
def pages(tmp_path):
    made = []
    for name in ("K7M2QX-p0002B", "K7M2QX-p0001F", "K7M2QX-p0002F"):
        path = tmp_path / f"{name}.png"
        Image.new("L", (600, 850), 255).save(path)
        made.append(path)
    return made


def test_pages_come_back_in_reading_order(pages):
    client = FakeClient([FakeMessage(f"page {i}") for i in range(3)])
    run = transcribe(pages, client)
    assert [t.name for t in run.transcripts] == [
        "K7M2QX-p0001F",
        "K7M2QX-p0002F",
        "K7M2QX-p0002B",
    ]


def test_one_bad_page_does_not_lose_the_rest(pages):
    client = FakeClient(
        [FakeMessage("first"), RuntimeError("connection reset"), FakeMessage("third")]
    )
    run = transcribe(pages, client)
    assert len(run.transcripts) == 2
    assert len(run.failures) == 1
    assert "connection reset" in run.failures[0][1]


def test_the_cost_estimate_adds_up(pages):
    client = FakeClient([FakeMessage("x") for _ in range(3)])
    run = transcribe(pages, client)
    assert run.input_tokens == 3000 and run.output_tokens == 300
    # 3000/1e6 * $5 + 300/1e6 * $25
    assert run.cost("claude-opus-5") == pytest.approx(0.0225)
    assert run.cost("some-other-model") is None


def test_markdown_skips_blank_pages_but_keeps_headings(pages):
    # Replies are consumed in the order `pages` lists them -- p0002B, p0001F,
    # p0002F -- so the blank lands on the page that would have sorted last.
    client = FakeClient(
        [FakeMessage(BLANK_MARKER), FakeMessage("first page"), FakeMessage("second page")]
    )
    document = as_markdown(transcribe(pages, client))
    assert BLANK_MARKER not in document
    assert document.index("first page") < document.index("second page")
    assert "## K7M2QX page 1 (front)" in document
    assert "## K7M2QX page 2 (front)" in document


# -- through the CLI --------------------------------------------------------


def run_cli(capsys, monkeypatch, replies, *argv):
    client = FakeClient(replies)
    monkeypatch.setattr("paperlog.transcribe._load_client", lambda *a, **k: client)
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err, client


def test_transcribe_writes_one_document_and_one_file_per_page(
    pages, tmp_path, capsys, monkeypatch
):
    code, stdout, _, _ = run_cli(
        capsys,
        monkeypatch,
        [FakeMessage("alpha"), FakeMessage("beta"), FakeMessage("gamma")],
        "transcribe", str(pages[0].parent),
        "-o", str(tmp_path / "notebook.md"),
        "-d", str(tmp_path / "text"),
    )
    assert code == 0
    assert (tmp_path / "notebook.md").exists()
    assert (tmp_path / "text" / "K7M2QX-p0001F.md").read_text().strip() in {
        "alpha", "beta", "gamma"
    }
    assert "read      3 page(s), 0 failed" in stdout
    assert "$0.02" in stdout


def test_unclear_words_are_surfaced_at_the_end(pages, tmp_path, capsys, monkeypatch):
    code, stdout, _, _ = run_cli(
        capsys,
        monkeypatch,
        [FakeMessage("met [?Aoife] at [?]"), FakeMessage("clean"), FakeMessage("clean")],
        "transcribe", str(pages[0].parent), "-o", str(tmp_path / "out.md"),
    )
    assert code == 0
    assert "unclear   2 word(s)" in stdout


def test_transcribing_without_the_setup_says_which_step_is_missing(
    pages, tmp_path, capsys, monkeypatch
):
    """Two ways to be unconfigured, and each names its own fix.

    Which one fires depends on whether the ``anthropic`` package is installed
    in the environment running the tests, so this asserts on both rather than
    pinning the one that happens to apply here.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    code = main(["transcribe", str(pages[0]), "-o", str(tmp_path / "out.md")])
    captured = capsys.readouterr()
    assert code == 1
    assert "paper-log[transcribe]" in captured.err or "ANTHROPIC_API_KEY" in captured.err
    assert not (tmp_path / "out.md").exists()


def test_a_missing_api_key_says_what_to_set(monkeypatch):
    pytest.importorskip("anthropic", reason="needs the [transcribe] extras")
    from paperlog.transcribe import _load_client

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(TranscribeError, match="ANTHROPIC_API_KEY"):
        _load_client()


def test_no_images_is_an_error_not_an_empty_document(tmp_path, capsys):
    empty = tmp_path / "nothing"
    empty.mkdir()
    code = main(["transcribe", str(empty)])
    assert code == 1
    assert "no page images" in capsys.readouterr().err
