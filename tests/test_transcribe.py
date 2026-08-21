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
    DEFAULT_SYSTEM_PROMPT,
    TranscribeConfig,
    TranscribeError,
    as_markdown,
    find_unsure,
    load_transcribe_config,
    transcribe,
    transcribe_page,
)
from paperlog.vision import (
    CONSERVATIVE_LIMITS,
    MODEL_LIMITS,
    VisionError,
    VisionSpec,
    encode_image,
    limits_for,
)

HIGH_RES = MODEL_LIMITS["claude-opus-5"]
MAX_EDGE, MAX_PIXELS = HIGH_RES

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


def cfg(**kwargs) -> TranscribeConfig:
    """A config with the vision spec's own defaults, plus any overrides."""
    vision = kwargs.pop("vision", None)
    return TranscribeConfig(vision=vision or VisionSpec(), **kwargs)


def backend(client, spec=None):
    """An Anthropic backend wired to a FakeClient instead of a real one."""
    from paperlog.vision import AnthropicBackend

    return AnthropicBackend(spec or VisionSpec(), client=client)


@pytest.fixture
def page(tmp_path):
    """An image shaped like something `scan` would have written."""
    path = tmp_path / "K7M2QX-p0003F.png"
    Image.new("L", (1748, 2480), 255).save(path)
    return path


# -- the request we build ---------------------------------------------------


def test_the_page_is_sent_as_an_image_with_an_instruction(page):
    client = FakeClient()
    transcribe_page(page, backend=backend(client))

    (call,) = client.calls
    assert call["model"] == "claude-opus-5"
    assert call["system"] == DEFAULT_SYSTEM_PROMPT
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
    transcribe_page(page, backend=backend(client))

    (call,) = client.calls
    assert call["output_config"] == {"effort": "low"}
    assert "thinking" not in call
    assert call["max_tokens"] >= 4000  # shared between thinking and the reply


def test_context_is_passed_but_fenced(page):
    client = FakeClient()
    transcribe_page(
        page,
        cfg(context="Kowalczyk, Bräuer, tetrahydrofuran"),
        backend=backend(client),
    )

    text = client.calls[0]["messages"][0]["content"][1]["text"]
    assert "tetrahydrofuran" in text
    assert "never to add words that are not on the page" in text


def test_no_context_means_no_empty_context_section(page):
    client = FakeClient()
    transcribe_page(page, backend=backend(client))
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

    resized = _decode(encode_image(path, max_edge=MAX_EDGE, max_pixels=MAX_PIXELS)[1])
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
    resized = _decode(encode_image(page, max_edge=MAX_EDGE, max_pixels=MAX_PIXELS)[1])
    assert max(resized.size) < MAX_EDGE
    assert resized.width * resized.height <= MAX_PIXELS
    assert resized.size[0] / resized.size[1] == pytest.approx(1748 / 2480, abs=0.01)


def test_a_page_within_both_limits_is_untouched(tmp_path):
    path = tmp_path / "small.png"
    Image.new("L", (1200, 1700), 255).save(path)
    assert _decode(encode_image(path, max_edge=MAX_EDGE, max_pixels=MAX_PIXELS)[1]).size == (1200, 1700)


def test_an_unreadable_image_is_reported_not_raised_raw(tmp_path):
    junk = tmp_path / "notreally.png"
    junk.write_text("this is not a png")
    with pytest.raises(VisionError, match="cannot read"):
        encode_image(junk, max_edge=MAX_EDGE, max_pixels=MAX_PIXELS)


# -- reading the response ---------------------------------------------------


def test_the_page_identity_comes_from_the_filename(page):
    """`scan` already worked out which page this is; don't ask the model."""
    result = transcribe_page(page, backend=backend(FakeClient([FakeMessage("Some notes.")])))
    assert result.ref is not None
    assert (result.ref.notebook, result.ref.page, result.ref.side) == ("K7M2QX", 3, "F")


def test_an_unrecognised_filename_still_transcribes(tmp_path):
    path = tmp_path / "holiday-snap.png"
    Image.new("L", (800, 1000), 255).save(path)
    result = transcribe_page(path, backend=backend(FakeClient([FakeMessage("words")])))
    assert result.ref is None
    assert result.text == "words"


def test_uncertain_words_are_pulled_out_for_review():
    assert find_unsure("met [?Aoife] about the [?] budget") == ["Aoife", ""]
    assert find_unsure("nothing unclear here") == []


def test_a_blank_page_is_recognised(page):
    result = transcribe_page(page, backend=backend(FakeClient([FakeMessage(BLANK_MARKER)])))
    assert result.blank


def test_a_refusal_is_explained_rather_than_crashing(page):
    """A refusal is a 200 with empty content.

    Reading content[0] unconditionally would turn a policy decision into an
    IndexError with no hint about what went wrong.
    """
    refusal = FakeMessage(stop_reason="refusal", blocks=[])
    with pytest.raises(TranscribeError, match="declined"):
        transcribe_page(page, backend=backend(FakeClient([refusal])))


def test_running_out_of_output_budget_says_so(page):
    truncated = FakeMessage(stop_reason="max_tokens", blocks=[])
    with pytest.raises(TranscribeError, match="output budget"):
        transcribe_page(page, backend=backend(FakeClient([truncated])))


def test_usage_is_carried_through_for_the_cost_estimate(page):
    result = transcribe_page(page, backend=backend(FakeClient([FakeMessage("hi")])))
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
    run = transcribe(pages, backend=backend(client))
    assert [t.name for t in run.transcripts] == [
        "K7M2QX-p0001F",
        "K7M2QX-p0002F",
        "K7M2QX-p0002B",
    ]


def test_one_bad_page_does_not_lose_the_rest(pages):
    client = FakeClient(
        [FakeMessage("first"), RuntimeError("connection reset"), FakeMessage("third")]
    )
    run = transcribe(pages, backend=backend(client))
    assert len(run.transcripts) == 2
    assert len(run.failures) == 1
    assert "connection reset" in run.failures[0][1]


def test_the_cost_estimate_adds_up(pages):
    client = FakeClient([FakeMessage("x") for _ in range(3)])
    run = transcribe(pages, backend=backend(client))
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
    document = as_markdown(transcribe(pages, backend=backend(client)))
    assert BLANK_MARKER not in document
    assert document.index("first page") < document.index("second page")
    assert "## K7M2QX page 1 (front)" in document
    assert "## K7M2QX page 2 (front)" in document


# -- through the CLI --------------------------------------------------------


def run_cli(capsys, monkeypatch, replies, *argv):
    """Drive the CLI with a fake client behind the real backend.

    Patched at `transcribe.backend_for`, which is the seam the command actually
    calls -- so the config plumbing, the spec and the request body are all still
    exercised for real; only the socket is fake.
    """
    client = FakeClient(replies)
    monkeypatch.setattr(
        "paperlog.transcribe.backend_for",
        lambda spec, **_: backend(client, spec),
    )
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
    from paperlog.vision import AnthropicBackend

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    with pytest.raises(VisionError, match="ANTHROPIC_API_KEY"):
        AnthropicBackend(VisionSpec()).preflight()


def test_a_custom_key_variable_is_named_in_the_error(monkeypatch):
    """The message has to name the variable *you* configured, not the default."""
    pytest.importorskip("anthropic", reason="needs the [transcribe] extras")
    from paperlog.vision import AnthropicBackend

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("WORK_KEY", raising=False)
    spec = VisionSpec(api_key_env="WORK_KEY")
    with pytest.raises(VisionError, match="WORK_KEY"):
        AnthropicBackend(spec).preflight()


def test_no_images_is_an_error_not_an_empty_document(tmp_path, capsys):
    empty = tmp_path / "nothing"
    empty.mkdir()
    code = main(["transcribe", str(empty)])
    assert code == 1
    assert "no page images" in capsys.readouterr().err


# -- configurability --------------------------------------------------------


def test_image_limits_come_from_the_model_not_a_constant():
    """The bug this table exists to fix.

    Haiku 4.5 predates the high-resolution vision tier: it takes 1568px against
    the 2576px of the current models, which is under half the pixel area. Sizing
    every model for the high tier means Haiku silently resizes on the far end --
    the one thing local resizing is here to prevent -- and on handwriting the
    resolution is most of the accuracy.
    """
    assert limits_for("claude-opus-5") == (2576, 3_750_000)
    assert limits_for("claude-haiku-4-5") < limits_for("claude-opus-5")
    high_edge, high_px = limits_for("claude-opus-5")
    low_edge, low_px = limits_for("claude-haiku-4-5")
    assert low_px / high_px < 0.5  # the gap is large enough to matter


def test_a_model_sends_a_smaller_image_when_its_limits_are_smaller(page):
    big = FakeClient()
    small = FakeClient()
    transcribe_page(page, cfg(), backend=backend(big, VisionSpec(model="claude-opus-5")))
    transcribe_page(
        page, cfg(), backend=backend(small, VisionSpec(model="claude-haiku-4-5"))
    )

    def sent(client):
        data = client.calls[0]["messages"][0]["content"][0]["source"]["data"]
        return _decode(data).size

    wide, narrow = sent(big), sent(small)
    assert max(narrow) < max(wide)
    assert narrow[0] * narrow[1] <= limits_for("claude-haiku-4-5")[1]


def test_an_unknown_model_falls_back_conservatively_and_says_so():
    spec = VisionSpec(model="some-new-model-v9")
    assert spec.limits == CONSERVATIVE_LIMITS
    assert spec.limits_are_a_guess


def test_overriding_the_limits_silences_the_guess():
    spec = VisionSpec(model="some-new-model-v9", max_edge=4000, max_pixels=9_000_000)
    assert spec.limits == (4000, 9_000_000)
    assert not spec.limits_are_a_guess


def test_deployment_prefixes_resolve_to_the_same_tier():
    """Bedrock prefixes and Vertex snapshot suffixes name a deployment, not a
    different model, so they must not fall through to the conservative default."""
    for name in (
        "anthropic.claude-opus-5",
        "us.anthropic.claude-opus-5",
        "claude-opus-5@20260101",
    ):
        assert limits_for(name) == limits_for("claude-opus-5"), name


def test_config_round_trips_through_yaml(tmp_path):
    path = tmp_path / "t.yaml"
    path.write_text(
        "vision:\n"
        "  model: claude-sonnet-5\n"
        "  effort: medium\n"
        "  base_url: http://127.0.0.1:9000\n"
        "  api_key_env: WORK_KEY\n"
        "skip_blank: false\n"
    )
    config = load_transcribe_config(path)
    assert config.vision.model == "claude-sonnet-5"
    assert config.vision.effort == "medium"
    assert config.vision.base_url == "http://127.0.0.1:9000"
    assert config.vision.api_key_env == "WORK_KEY"
    assert config.skip_blank is False


def test_flags_beat_the_config_file(tmp_path):
    path = tmp_path / "t.yaml"
    path.write_text("vision:\n  model: claude-sonnet-5\n  effort: medium\n")
    config = load_transcribe_config(path, {"vision": {"model": "claude-haiku-4-5"}})
    assert config.vision.model == "claude-haiku-4-5"
    assert config.vision.effort == "medium"  # untouched keys survive the merge


def test_the_config_is_found_under_paperlog_home(tmp_path, monkeypatch):
    from paperlog.library import HOME_VARIABLE

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv(HOME_VARIABLE, str(home))
    (home / "transcribe.yaml").write_text("vision:\n  model: claude-haiku-4-5\n")
    assert load_transcribe_config().vision.model == "claude-haiku-4-5"


def test_no_config_anywhere_is_not_an_error():
    assert load_transcribe_config().vision.model == "claude-opus-5"


def test_a_nested_transcribe_section_is_accepted(tmp_path):
    """So the same file can grow other sections later."""
    path = tmp_path / "t.yaml"
    path.write_text("transcribe:\n  vision:\n    model: claude-sonnet-5\n")
    assert load_transcribe_config(path).vision.model == "claude-sonnet-5"


def test_a_typo_in_the_config_is_rejected_with_a_hint(tmp_path):
    path = tmp_path / "t.yaml"
    path.write_text("vision:\n  modle: claude-opus-5\n")
    with pytest.raises(TranscribeError, match="modle"):
        load_transcribe_config(path)


def test_a_bad_effort_is_rejected_at_load_time(tmp_path):
    path = tmp_path / "t.yaml"
    path.write_text("vision:\n  effort: turbo\n")
    with pytest.raises(TranscribeError, match="effort must be one of"):
        load_transcribe_config(path)


def test_the_prompt_can_be_replaced_from_a_file(tmp_path, page):
    prompt = tmp_path / "prompt.md"
    prompt.write_text("Read the page. Output JSON.")
    config = load_transcribe_config(None, {"prompt_file": str(prompt)})
    assert config.system_prompt == "Read the page. Output JSON."

    client = FakeClient()
    transcribe_page(page, config, backend=backend(client))
    assert client.calls[0]["system"] == "Read the page. Output JSON."


def test_a_missing_prompt_file_is_reported_clearly(tmp_path):
    with pytest.raises(TranscribeError, match="cannot read prompt_file"):
        load_transcribe_config(None, {"prompt_file": str(tmp_path / "nope.md")})


def test_the_config_file_never_holds_the_key_itself(tmp_path):
    """Only the *name* of the variable is configurable, by design.

    A settings file you can commit is worth more than one that can hold a
    secret, so there is no api_key field to put one in.
    """
    path = tmp_path / "t.yaml"
    path.write_text("vision:\n  api_key: sk-ant-oops\n")
    with pytest.raises(TranscribeError, match="api_key"):
        load_transcribe_config(path)


def test_write_config_produces_a_file_that_loads(tmp_path, capsys):
    target = tmp_path / "written.yaml"
    assert main(["transcribe", "--write-config", "-c", str(target)]) == 0
    config = load_transcribe_config(target)
    assert config.vision.model == "claude-opus-5"

    # And it refuses to clobber, unless told to.
    assert main(["transcribe", "--write-config", "-c", str(target)]) == 1
    assert "--force" in capsys.readouterr().err
    assert main(["transcribe", "--write-config", "-c", str(target), "--force"]) == 0


def test_cli_flags_reach_the_request(pages, tmp_path, capsys, monkeypatch):
    code, stdout, _, client = run_cli(
        capsys,
        monkeypatch,
        [FakeMessage("a"), FakeMessage("b"), FakeMessage("c")],
        "transcribe", str(pages[0].parent),
        "--model", "claude-sonnet-5",
        "--effort", "high",
        "--max-tokens", "5000",
        "-o", str(tmp_path / "out.md"),
    )
    assert code == 0
    call = client.calls[0]
    assert call["model"] == "claude-sonnet-5"
    assert call["output_config"] == {"effort": "high"}
    assert call["max_tokens"] == 5000
    assert "model     claude-sonnet-5 (effort high)" in stdout


def test_an_unknown_model_warns_on_stderr(pages, tmp_path, capsys, monkeypatch):
    code, _, stderr, _ = run_cli(
        capsys, monkeypatch, [FakeMessage("a")] * 3,
        "transcribe", str(pages[0].parent),
        "--model", "mystery-model-1", "-o", str(tmp_path / "out.md"),
    )
    assert code == 0
    assert "no image limits known for mystery-model-1" in stderr


def test_nothing_read_means_nothing_written(pages, tmp_path, capsys, monkeypatch):
    """An empty document is worse than no document: it looks like a result."""
    out = tmp_path / "out.md"
    code, _, stderr, _ = run_cli(
        capsys, monkeypatch, [RuntimeError("boom")] * 3,
        "transcribe", str(pages[0].parent), "-o", str(out),
    )
    assert code == 1
    assert not out.exists()
    assert "left alone" in stderr


def test_cost_uses_the_model_that_actually_ran(pages, tmp_path, capsys, monkeypatch):
    code, stdout, _, _ = run_cli(
        capsys, monkeypatch, [FakeMessage("a")] * 3,
        "transcribe", str(pages[0].parent),
        "--model", "claude-haiku-4-5", "-o", str(tmp_path / "out.md"),
    )
    assert code == 0
    # 3000 in / 300 out at Haiku's $1/$5 = $0.0045, not Opus's $0.0225.
    assert "$0.00" in stdout


# -- a second provider ------------------------------------------------------


class FakePart:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeGeminiModels:
    """Stands in for client.models, recording the request it was handed."""

    def __init__(self, text="page text", finish="STOP"):
        self.calls = []
        self._text = text
        self._finish = finish

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        reason = type("Reason", (), {"name": self._finish})()
        candidate = type("Candidate", (), {"finish_reason": reason})()
        usage = type("Usage", (), {"prompt_token_count": 1200, "candidates_token_count": 90})()
        return type(
            "Response",
            (),
            {
                "text": self._text,
                "candidates": [candidate],
                "usage_metadata": usage,
                "model_version": model,
            },
        )()


class FakeGeminiClient:
    def __init__(self, text="page text", finish="STOP"):
        self.models = FakeGeminiModels(text, finish)


def gemini(**kwargs):
    from paperlog.vision import GeminiBackend

    spec = VisionSpec(provider="gemini", model=kwargs.pop("model", "gemini-3-pro"), **kwargs)
    client = FakeGeminiClient()
    return GeminiBackend(spec, client=client), client


def test_a_second_provider_slots_in_behind_the_same_seam(page):
    """The point of the seam: transcription does not know who answered."""
    backend, client = gemini()
    result = transcribe_page(page, cfg(), backend=backend)
    assert result.text == "page text"
    assert result.input_tokens == 1200 and result.output_tokens == 90
    assert len(client.models.calls) == 1


def test_the_gemini_request_carries_the_system_prompt_and_the_image(page):
    backend, client = gemini()
    transcribe_page(page, cfg(), backend=backend)

    (call,) = client.models.calls
    assert call["model"] == "gemini-3-pro"
    assert call["config"].system_instruction == DEFAULT_SYSTEM_PROMPT
    # An image part and a text part, in that order.
    assert len(call["contents"]) == 2
    assert call["contents"][0].inline_data is not None
    assert call["contents"][1].text.startswith("Transcribe the handwriting")


def test_a_gemini_safety_stop_reads_as_a_refusal(page):
    """Gemini reports it as a finish reason on the candidate rather than a stop
    reason on the message; the rest of paper-log should not have to know."""
    from paperlog.vision import GeminiBackend

    spec = VisionSpec(provider="gemini", model="gemini-3-pro")
    backend = GeminiBackend(spec, client=FakeGeminiClient(text="", finish="SAFETY"))
    with pytest.raises(TranscribeError, match="declined"):
        transcribe_page(page, cfg(), backend=backend)


def test_running_out_of_budget_reads_the_same_on_either_provider(page):
    from paperlog.vision import GeminiBackend

    spec = VisionSpec(provider="gemini", model="gemini-3-pro")
    backend = GeminiBackend(spec, client=FakeGeminiClient(text="", finish="MAX_TOKENS"))
    with pytest.raises(TranscribeError, match="output budget"):
        transcribe_page(page, cfg(), backend=backend)


def test_the_key_variable_follows_the_provider():
    """Switching provider should not send you hunting for a key under the
    other vendor's name -- but an explicit choice is always kept."""
    assert VisionSpec().api_key_env == "ANTHROPIC_API_KEY"
    assert VisionSpec(provider="gemini").api_key_env == "GEMINI_API_KEY"
    assert VisionSpec(provider="gemini", api_key_env="WORK_KEY").api_key_env == "WORK_KEY"


def test_an_unknown_provider_is_rejected_with_the_list():
    with pytest.raises(VisionError, match="anthropic, gemini"):
        VisionSpec(provider="openai")


def test_the_provider_is_selectable_from_config(tmp_path):
    path = tmp_path / "t.yaml"
    path.write_text("vision:\n  provider: gemini\n  model: gemini-3-pro\n")
    config = load_transcribe_config(path)
    assert config.vision.provider == "gemini"
    assert config.vision.api_key_env == "GEMINI_API_KEY"
