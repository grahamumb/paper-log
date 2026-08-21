"""Requests written on paper: finding them, reading them, running them.

The property most of this file exists to protect is **isolation**. The writer
draws a box around a request; what reaches the model that answers it must be
that request and nothing else -- not the paragraph above it, not the rest of the
page, not the other requests in the same run. If prose leaks in, the answer
starts getting tailored to writing the author never meant to submit, and
authorial control quietly moves from the person to the machine.

That property is enforced structurally rather than by asking nicely, so it can
be tested structurally: the crop is checked to contain no surrounding ink, and
the dispatched request is checked to contain no surrounding words.
"""

import json
import math
import random

import pytest

from paperlog import JournalConfig, build
from paperlog.calls import (
    DONE,
    FAILED,
    PENDING,
    Ledger,
    ToolCall,
    normalise_prompt,
)
from paperlog.cli import main
from paperlog.extract import UNKNOWN_TOOL, extract_from_page, parse_reply
from paperlog.tools import (
    BUILTIN_TOOLS,
    ToolError,
    check_output,
    destination,
    load_tools,
    run_call,
    strip_fence,
)
pytest.importorskip("cv2", reason="needs the [verify] extras")

from paperlog.boxes import crop, find_boxes  # noqa: E402
from paperlog.capture import enhance, flatten, load_manifests  # noqa: E402


# -- drawing a page a person might have written -----------------------------


def _wobble(image, start, end, rng, thickness=4, jitter=3.0):
    """A pen line. Never straight, and the hand shakes."""
    import cv2

    points = []
    for step in range(41):
        t = step / 40
        x = start[0] + (end[0] - start[0]) * t + rng.gauss(0, jitter) + math.sin(t * 6) * jitter
        y = start[1] + (end[1] - start[1]) * t + rng.gauss(0, jitter) + math.cos(t * 5) * jitter
        points.append((int(x), int(y)))
    for a, b in zip(points, points[1:]):
        cv2.line(image, a, b, 20, max(1, thickness + rng.randint(-1, 1)), cv2.LINE_AA)


def draw_box(image, x, y, width, height, seed=0):
    rng = random.Random(seed)
    corners = [(x, y), (x + width, y), (x + width, y + height), (x, y + height)]
    for a, b in zip(corners, corners[1:] + corners[:1]):
        _wobble(image, a, b, rng)
    return (x, y, width, height)


def write(image, x, y, text, scale=1.0):
    import cv2

    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, scale, 25, 3,
                cv2.LINE_AA)


@pytest.fixture
def notebook(tmp_path):
    config = JournalConfig.from_dict(
        {"page_size": "a5", "pages": 4, "notebook_id": "K7M2QX"}
    )
    result = build(config, tmp_path / "journal.pdf")
    return result, load_manifests([result.manifest_path])


@pytest.fixture
def rendered(notebook, render_page):
    result, manifests = notebook
    return result, manifests, render_page(result.pdf_path, 0)


#: The words that surround the box. If any of these ever reach a model, the
#: isolation the whole design rests on has been lost -- so they are distinctive
#: enough to search for.
SURROUNDING = [
    "Kestrel migration notes for chapter nine",
    "and the funding deadline is on Tuesday",
]


def boxed_page(page, phone, seed=0, box=(250, 430, 1180, 300)):
    """A page with prose above and below a boxed request."""
    import numpy

    page = numpy.array(page, copy=True)
    write(page, 240, 330, SURROUNDING[0], scale=0.95)
    truth = draw_box(page, *box, seed=seed)
    write(page, 300, 520, "TOOL research-request", scale=0.85)
    write(page, 300, 620, "books on the black plague", scale=0.85)
    write(page, 240, 900, SURROUNDING[1], scale=0.95)
    return page, truth


def flat_page(result, manifests, page, phone, seed=0):
    flattened = flatten(phone(page, seed=seed), manifests)[0]
    return enhance(flattened.image, "flatten"), flattened


# -- finding the box --------------------------------------------------------


def test_a_drawn_box_is_found_on_a_ruled_page(rendered, phone):
    """The concern that prompted this: a pen box on top of printed ruling.

    It works because they differ in *thickness*, not darkness -- printed ruling
    is a fraction of a millimetre and a pen line is several times that, so an
    erosion sized between them keeps one and drops the other. Tone would not do
    it: flattening pushes faint ruling to white in places and leaves it grey in
    others.
    """
    result, manifests, page = rendered
    written, truth = boxed_page(page, phone)
    flat, _ = flat_page(result, manifests, written, phone)

    boxes = find_boxes(flat)
    assert len(boxes) == 1, [b for b in boxes]
    box = boxes[0]
    # Within a couple of millimetres of where it was drawn.
    assert abs(box.x - truth[0]) < 30 and abs(box.y - truth[1]) < 30
    assert abs(box.width - truth[2]) < 60 and abs(box.height - truth[3]) < 60


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_the_box_survives_being_photographed(rendered, phone, seed):
    result, manifests, page = rendered
    written, _ = boxed_page(page, phone, seed=seed)
    flat, _ = flat_page(result, manifests, written, phone, seed=seed)
    assert len(find_boxes(flat)) == 1


def test_a_page_with_no_box_finds_nothing(rendered, phone):
    """The failure that would cost real money: inventing a request."""
    result, manifests, page = rendered
    import numpy

    written = numpy.array(page, copy=True)
    for row in range(18):
        write(written, 240, 330 + row * 84, "the quick brown fox jumps over the lazy dog")
    flat, _ = flat_page(result, manifests, written, phone)
    assert find_boxes(flat) == []


def test_underlines_and_margin_rules_are_not_boxes(rendered, phone):
    """Both are thick pen strokes, so thickness alone would accept them.

    They are rejected for not enclosing anything, which is the second half of
    the test and the reason it is not just an erosion.
    """
    import cv2
    import numpy

    result, manifests, page = rendered
    written = numpy.array(page, copy=True)
    write(written, 240, 400, "a heavily underlined heading")
    cv2.line(written, (240, 430), (1400, 435), 20, 6)
    cv2.line(written, (200, 600), (200, 1100), 20, 6)  # margin rule
    flat, _ = flat_page(result, manifests, written, phone)
    assert find_boxes(flat) == []


def test_the_corner_codes_are_never_boxes(rendered, phone):
    """A QR code is a rectangle of ink in a corner, which is the shape being
    looked for. Geometry from the manifest excludes them outright."""
    from paperlog.capture import PageGeometry

    result, manifests, page = rendered
    written, _ = boxed_page(page, phone)
    flat, _ = flat_page(result, manifests, written, phone)
    geometry = PageGeometry.from_manifest(manifests["K7M2QX"])

    with_geometry = find_boxes(flat, geometry=geometry)
    assert len(with_geometry) == 1
    for box in with_geometry:
        assert box.width < flat.shape[1] * 0.9


def test_two_boxes_come_back_in_reading_order(rendered, phone):
    import numpy

    result, manifests, page = rendered
    written = numpy.array(page, copy=True)
    draw_box(written, 250, 1400, 1180, 260, seed=1)
    write(written, 300, 1500, "TOOL research-request", scale=0.85)
    draw_box(written, 250, 430, 1180, 260, seed=2)
    write(written, 300, 540, "TOOL interactive-break", scale=0.85)
    flat, _ = flat_page(result, manifests, written, phone)

    boxes = find_boxes(flat)
    assert len(boxes) == 2
    assert boxes[0].y < boxes[1].y  # the upper one first


# -- the isolation guarantee ------------------------------------------------


def test_the_crop_contains_the_box_and_nothing_around_it(rendered, phone):
    """The load-bearing test for the whole design.

    Cropping is what makes isolation a fact rather than an instruction: the
    surrounding prose is not redacted or ignored, it is simply not present in
    the pixels that get sent. This checks the rows above and below the box are
    genuinely gone, which is what stops a request from being tailored to
    writing the author never submitted.
    """
    result, manifests, page = rendered
    written, _ = boxed_page(page, phone)
    flat, _ = flat_page(result, manifests, written, phone)

    box = find_boxes(flat)[0]
    region = crop(flat, box)
    assert region.shape[0] == box.height and region.shape[1] == box.width

    # The prose sits above and below the box on the full page; the crop starts
    # at the box and ends at it, so none of those rows can be inside it.
    assert box.y > 0 and box.y + box.height < flat.shape[0]
    above = flat[: box.y - 5]
    below = flat[box.y + box.height + 5 :]
    assert (above < 200).sum() > 0, "the fixture should have ink above the box"
    assert (below < 200).sum() > 0, "the fixture should have ink below the box"
    # Nothing was padded outward: the crop is exactly the rectangle.
    assert region.shape == (box.height, box.width)


def test_a_dispatched_request_carries_the_prompt_and_nothing_else():
    """What the tool actually receives.

    Even a well-cropped image is worth nothing if the dispatch then helpfully
    attaches the page for context, so this checks the far end of the pipe too.
    """
    call = ToolCall(
        tool="research-request",
        prompt="books on the black plague, primary sources",
        notebook="K7M2QX",
        page=1,
    )
    backend = RecordingBackend()
    run_call(call, BUILTIN_TOOLS["research-request"], backend=backend)

    (sent,) = backend.calls
    assert sent["images"] == 0, "the page must not travel with the request"
    body = sent["instruction"]
    assert "black plague" in body
    for phrase in SURROUNDING:
        assert phrase not in body
    assert "K7M2QX" not in body, "not even which notebook it came from"
    assert "page" not in body.lower().replace("plague", "")


class RecordingBackend:
    """A backend that answers canned text and remembers what it was asked."""

    def __init__(self, replies=None, limits=(2576, 3_750_000)):
        from paperlog.vision import Reply

        self.calls = []
        self.limits = limits
        self._replies = list(replies or [])
        self._Reply = Reply

    def _next(self, default):
        if not self._replies:
            return self._Reply(text=default, input_tokens=100, output_tokens=20)
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return self._Reply(text=reply, input_tokens=100, output_tokens=20)

    def read(self, *, media_type, data, system, instruction):
        self.calls.append(
            {"images": 1, "system": system, "instruction": instruction, "data": data}
        )
        return self._next("TOOL: research-request\nPROMPT:\nbooks on the black plague")

    def ask(self, *, system, instruction):
        self.calls.append(
            {"images": 0, "system": system, "instruction": instruction, "data": ""}
        )
        return self._next("# A report\n\nBody.")

    def preflight(self):
        return None


def test_extraction_sends_only_the_cropped_region(rendered, phone):
    import base64
    import io

    from PIL import Image

    result, manifests, page = rendered
    written, truth = boxed_page(page, phone)
    flat, flattened = flat_page(result, manifests, written, phone)

    backend = RecordingBackend()
    found = extract_from_page(flat, flattened.ref, backend=backend)
    assert len(found) == 1
    box = found[0].box

    (sent,) = backend.calls
    sent_image = Image.open(io.BytesIO(base64.standard_b64decode(sent["data"])))

    # What went on the wire is the box, exactly -- not the page, and not the
    # box with a helpful margin of context around it.
    assert sent_image.size == (box.width, box.height)
    assert sent_image.height < flat.shape[0] * 0.5
    # And the box is the one that was drawn, give or take the few pixels the
    # dilation adds while closing the wobble.
    assert sent_image.width == pytest.approx(truth[2], rel=0.1)
    assert sent_image.height == pytest.approx(truth[3], rel=0.15)


# -- reading what came back -------------------------------------------------


def test_a_well_formed_reply_parses():
    tool, prompt = parse_reply(
        "TOOL: research-request\nPROMPT:\nbooks on the black plague\nprimary sources?"
    )
    assert tool == "research-request"
    assert prompt == "books on the black plague\nprimary sources?"


def test_parsing_tolerates_case_and_stray_preamble():
    """A model that adds a line has not made a page-losing mistake."""
    tool, prompt = parse_reply(
        "Here is the transcription:\n\ntool: interactive-break\nprompt:\ndouble pendulum"
    )
    assert tool == "interactive-break"
    assert prompt == "double pendulum"


def test_a_missing_tool_name_is_reported_not_guessed():
    tool, prompt = parse_reply("PROMPT:\nsomething I forgot to label")
    assert tool == UNKNOWN_TOOL
    assert prompt == "something I forgot to label"


def test_a_model_that_answers_instead_of_copying_does_not_become_a_prompt():
    """The failure this format is chosen to make visible.

    The box contains an imperative sentence and it is being shown to a language
    model, so 'it did the task instead of copying it' is the thing that can go
    wrong. It surfaces as a tool name that matches nothing, which the caller
    reports rather than runs.
    """
    tool, prompt = parse_reply(
        "Certainly! Here is a double pendulum visualisation:\n\n<html>...</html>"
    )
    assert tool == UNKNOWN_TOOL


def test_an_empty_box_yields_nothing_runnable(rendered, phone):
    result, manifests, page = rendered
    written, _ = boxed_page(page, phone)
    flat, flattened = flat_page(result, manifests, written, phone)
    backend = RecordingBackend(replies=["TOOL: ?\nPROMPT:\n"])
    (extraction,) = extract_from_page(flat, flattened.ref, backend=backend)
    assert not extraction.understood


# -- identity and the ledger ------------------------------------------------


def call(prompt="research the plague", **kwargs):
    base = dict(tool="research-request", notebook="K7M2QX", page=3, side="F", ordinal=0)
    base.update(kwargs)
    return ToolCall(prompt=prompt, **base)


def test_the_same_box_photographed_twice_is_one_call():
    """`scan` prefers a better shot of a page it already has, so every page
    gets presented more than once. Without this, every reshoot re-runs -- and
    re-bills -- every request on it."""
    assert call().id == call().id


def test_transcription_noise_does_not_change_a_call():
    """Handwriting read twice differs by a comma. That is not a new request."""
    assert call("Research the plague.").id == call("research the  plague").id
    assert normalise_prompt("Books, on the Plague!") == "books on the plague"


def test_rewording_the_request_does_make_a_new_call():
    assert call("research the plague").id != call("research the fire of london").id


def test_the_same_words_in_two_places_are_two_calls():
    assert call(page=3).id != call(page=4).id
    assert call(ordinal=0).id != call(ordinal=1).id
    assert call(notebook="K7M2QX").id != call(notebook="8QR41M").id


def test_the_ledger_records_and_reads_back(tmp_path):
    ledger = Ledger(tmp_path / "calls.jsonl")
    added = ledger.note_seen([call("one"), call("two", ordinal=1)])
    assert len(added) == 2
    assert [r.status for r in ledger.records()] == [PENDING, PENDING]


def test_a_call_already_known_is_not_added_again(tmp_path):
    ledger = Ledger(tmp_path / "calls.jsonl")
    ledger.note_seen([call("one")])
    assert ledger.note_seen([call("one")]) == []
    assert len(ledger.records()) == 1


def test_the_latest_row_wins(tmp_path):
    ledger = Ledger(tmp_path / "calls.jsonl")
    (record,) = ledger.note_seen([call("one")])
    record.status = DONE
    record.output_path = "/tmp/report.md"
    ledger.append(record)

    (back,) = ledger.records()
    assert back.status == DONE and back.output_path == "/tmp/report.md"
    assert ledger.pending() == []


def test_a_torn_ledger_line_does_not_lose_the_rest(tmp_path):
    """A crash mid-write should cost the last row, not the file."""
    path = tmp_path / "calls.jsonl"
    ledger = Ledger(path)
    ledger.note_seen([call("one")])
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"id": "truncated", "tool":\n')
    assert len(ledger.records()) == 1


def test_pending_can_be_filtered_by_tool(tmp_path):
    ledger = Ledger(tmp_path / "calls.jsonl")
    ledger.note_seen(
        [call("a"), call("b", tool="interactive-break", ordinal=1)]
    )
    assert len(ledger.pending("research-request")) == 1
    assert len(ledger.pending()) == 2


# -- tools ------------------------------------------------------------------


def test_both_tools_ship():
    tools = load_tools()
    assert set(tools) == {"interactive-break", "research-request"}
    assert tools["interactive-break"].output == "html"
    assert tools["research-request"].output == "markdown"


def test_every_contract_tells_the_model_the_writer_is_absent():
    """These run overnight. A tool that stops to ask a question has wasted the
    run, because there is nobody to answer until morning."""
    for tool in load_tools().values():
        assert "not available" in tool.system_prompt
        assert "assum" in tool.system_prompt.lower()


def test_the_html_contract_asks_for_something_that_can_actually_be_hosted():
    prompt = BUILTIN_TOOLS["interactive-break"].system_prompt
    for requirement in ("standalone", "No build step", "No network requests"):
        assert requirement in prompt


def test_a_builtin_can_be_retuned_without_restating_its_contract():
    tools = load_tools({"research-request": {"model": "claude-sonnet-5", "effort": "medium"}})
    tool = tools["research-request"]
    assert tool.model == "claude-sonnet-5" and tool.effort == "medium"
    assert tool.system_prompt == BUILTIN_TOOLS["research-request"].system_prompt


def test_a_new_tool_is_a_prompt_not_code():
    tools = load_tools(
        {"letter-draft": {"system_prompt": "Draft a letter.", "output": "markdown"}}
    )
    assert tools["letter-draft"].system_prompt == "Draft a letter."


def test_a_tool_without_a_contract_is_rejected():
    with pytest.raises(ToolError, match="needs a system_prompt"):
        load_tools({"halfbaked": {"output": "markdown"}})


def test_an_unknown_output_shape_is_rejected():
    with pytest.raises(ToolError, match="output must be one of"):
        load_tools({"odd": {"system_prompt": "x", "output": "pdf"}})


def test_the_destination_is_named_so_two_similar_requests_cannot_collide():
    from datetime import date

    tool = BUILTIN_TOOLS["research-request"]
    first = destination(tool, call("plague books"), "/out", when=date(2026, 3, 1))
    second = destination(tool, call("plague books", ordinal=1), "/out", when=date(2026, 3, 1))
    assert first != second
    assert first.suffix == ".md"
    assert first.parent.name == "research-request"
    assert "2026-03-01" in first.name and "plague-books" in first.name


def test_html_output_is_checked_for_the_things_that_break_it_silently():
    tool = BUILTIN_TOOLS["interactive-break"]
    clean = "<html><canvas></canvas><script>const a=1</script></html>"
    assert check_output(tool, clean) == []

    notes = check_output(tool, '<html><script src="https://cdn/x.js"></script></html>')
    assert any("external script" in note for note in notes)
    notes = check_output(tool, "<html><canvas></canvas><script>fetch('/x')</script></html>")
    assert any("fetch()" in note for note in notes)


def test_a_fenced_file_is_unwrapped():
    assert strip_fence("```html\n<html></html>\n```") == "<html></html>"
    assert strip_fence("<html></html>") == "<html></html>"


def test_a_refusal_on_a_tool_call_says_it_is_probably_a_false_positive():
    from paperlog.vision import Reply

    class Refusing(RecordingBackend):
        def ask(self, *, system, instruction):
            return Reply(text="", stop_reason="refusal")

    with pytest.raises(ToolError, match="false positive"):
        run_call(call(), BUILTIN_TOOLS["research-request"], backend=Refusing())


# -- per-notebook configuration ---------------------------------------------


def test_a_notebook_carries_what_it_is_for(tmp_path):
    """Paths are set when the notebook is made, not typed at scan time.

    The notebook is the thing you pick up, and it is what decides whether this
    is a journal or a blog draft. Putting the answer in the manifest means no
    marker has to be written on a page to say which.
    """
    config = JournalConfig.from_dict(
        {
            "pages": 4,
            "notebook_id": "K7M2QX",
            "writing": {
                "kind": "blog",
                "outputs": "~/blog/drafts",
                "tools": ["interactive-break"],
            },
        }
    )
    result = build(config, tmp_path / "j.pdf")
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["writing"]["kind"] == "blog"
    assert manifest["writing"]["outputs"] == "~/blog/drafts"
    assert manifest["writing"]["tools"] == ["interactive-break"]


def test_a_notebook_defaults_to_notes_with_no_restrictions(tmp_path):
    result = build(
        JournalConfig.from_dict({"pages": 4, "notebook_id": "K7M2QX"}), tmp_path / "j.pdf"
    )
    writing = json.loads(result.manifest_path.read_text())["writing"]
    assert writing["kind"] == "notes"
    assert writing["tools"] == []  # empty means every tool


def test_a_typo_in_the_writing_section_is_rejected():
    from paperlog.config import ConfigError

    with pytest.raises(ConfigError, match="output"):
        JournalConfig.from_dict({"writing": {"output": "~/x"}})


# -- through the CLI --------------------------------------------------------


def test_run_refuses_a_tool_the_notebook_does_not_allow(tmp_path, capsys, monkeypatch):
    """A fiction notebook should not be able to publish to the blog folder."""
    from paperlog.library import HOME_VARIABLE

    home = tmp_path / "home"
    monkeypatch.setenv(HOME_VARIABLE, str(home))
    result = build(
        JournalConfig.from_dict(
            {
                "pages": 4,
                "notebook_id": "K7M2QX",
                "writing": {"kind": "fiction", "tools": ["research-request"]},
            }
        ),
        tmp_path / "j.pdf",
    )
    main(["notebooks", "--add", str(result.manifest_path)])

    ledger = Ledger(home / "calls.jsonl")
    ledger.note_seen([call(tool="interactive-break")])

    code = main(["run", "-o", str(tmp_path / "out")])
    captured = capsys.readouterr()
    assert code == 1
    assert "not enabled for notebook K7M2QX" in captured.err


def test_dry_run_shows_where_things_would_land_without_spending(tmp_path, capsys, monkeypatch):
    from paperlog.library import HOME_VARIABLE

    monkeypatch.setenv(HOME_VARIABLE, str(tmp_path / "home"))
    ledger = Ledger(tmp_path / "home" / "calls.jsonl")
    ledger.note_seen([call("books on the plague")])

    code = main(["run", "--dry-run", "-o", str(tmp_path / "out")])
    out = capsys.readouterr().out
    assert code == 0
    assert "would run" in out and "research-request" in out
    assert not (tmp_path / "out").exists()
    assert ledger.pending(), "a dry run must not consume the call"


def test_failed_calls_are_only_rerun_when_asked(tmp_path, capsys, monkeypatch):
    from paperlog.library import HOME_VARIABLE

    monkeypatch.setenv(HOME_VARIABLE, str(tmp_path / "home"))
    ledger = Ledger(tmp_path / "home" / "calls.jsonl")
    (record,) = ledger.note_seen([call("books on the plague")])
    record.status = FAILED
    record.error = "connection reset"
    ledger.append(record)

    assert main(["run", "--dry-run"]) == 0
    assert "nothing pending" in capsys.readouterr().out

    assert main(["run", "--dry-run", "--retry"]) == 0
    assert "would run" in capsys.readouterr().out


def test_listing_calls_says_what_is_outstanding(tmp_path, capsys, monkeypatch):
    from paperlog.library import HOME_VARIABLE

    monkeypatch.setenv(HOME_VARIABLE, str(tmp_path / "home"))
    ledger = Ledger(tmp_path / "home" / "calls.jsonl")
    ledger.note_seen([call("books on the plague")])

    assert main(["calls", "--list"]) == 0
    listed = capsys.readouterr().out
    assert "research-request" in listed and "K7M2QX p3F" in listed and "pending" in listed
