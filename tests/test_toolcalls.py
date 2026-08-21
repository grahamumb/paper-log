"""Requests written on paper: finding them, reading them, running them.

The property most of this file exists to protect is **isolation**. The writer
boxes a request; what reaches the model that answers it must be that request and
nothing else -- not the paragraph above it, not the rest of the page, not the
other requests in the same run. If prose leaks in, the answer gets tailored to
writing the author never meant to submit, and authorial control moves quietly
from the person to the machine.

The guarantee is a division of labour rather than a promise. The transcriber
sees the whole page but only copies it. The extractor decides what the request
is but only ever sees text, by pattern match, with no model involved. So no
model both sees the context and shapes the request -- and the tests below check
the far end of that pipe, where the dispatched request is inspected for any
trace of the words that surrounded it.
"""

import json

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
from paperlog.extract import UNKNOWN_TOOL, find_calls, strip_calls
from paperlog.tools import (
    BUILTIN_TOOLS,
    ToolError,
    check_output,
    destination,
    load_tools,
    run_call,
    strip_fence,
)
from paperlog.ids import PageRef

REF = PageRef("K7M2QX", 1, "F", "TL")

#: The words that surround a request in the transcript. If any of these ever
#: reach the model that answers it, the isolation the design rests on is gone --
#: so they are distinctive enough to search for.
SURROUNDING = [
    "Kestrel migration notes for chapter nine",
    "and the funding deadline is on Tuesday",
]


def transcript(*, fenced=True, tool="research-request", body="books on the black plague"):
    """A page transcript shaped the way the transcriber is asked to write one."""
    if fenced:
        request = f"```paperlog-tool {tool}\n{body}\n```"
    else:
        request = f"TOOL {tool}\n{body}\nEND"
    return f"{SURROUNDING[0]}\n\n{request}\n\n{SURROUNDING[1]}\n"


# -- pulling requests out of a transcript -----------------------------------


def test_a_boxed_request_becomes_a_call():
    (found,) = find_calls(transcript(), REF)
    assert found.call.tool == "research-request"
    assert found.call.prompt == "books on the black plague"
    assert found.style == "box"
    assert found.understood


def test_a_typed_marker_works_too():
    """For when a box is inconvenient, or the transcriber missed one."""
    (found,) = find_calls(transcript(fenced=False), REF)
    assert found.call.tool == "research-request"
    assert found.call.prompt == "books on the black plague"
    assert found.style == "typed"


def test_a_forgotten_END_still_closes_the_request():
    """The obvious mistake to make. A blank line is what a reader would assume."""
    text = "prose\n\nTOOL research-request\nbooks on the plague\n\nmore prose\n"
    (found,) = find_calls(text, REF)
    assert found.call.prompt == "books on the plague"
    assert "more prose" not in found.call.prompt


def test_the_surrounding_prose_is_not_part_of_the_request():
    (found,) = find_calls(transcript(), REF)
    for phrase in SURROUNDING:
        assert phrase not in found.call.prompt


def test_a_page_with_no_request_yields_none():
    assert find_calls("Just some ordinary writing.\n\nAnd more of it.\n", REF) == []


def test_a_multi_line_request_keeps_its_shape():
    text = "```paperlog-tool interactive-break\ndouble pendulum viz\nsliders for angles\n```"
    (found,) = find_calls(text, REF)
    assert found.call.prompt == "double pendulum viz\nsliders for angles"


def test_two_requests_come_back_in_reading_order():
    text = (
        "```paperlog-tool research-request\nfirst\n```\n\n"
        "some prose\n\n"
        "```paperlog-tool interactive-break\nsecond\n```\n"
    )
    first, second = find_calls(text, REF)
    assert (first.call.tool, first.call.ordinal) == ("research-request", 0)
    assert (second.call.tool, second.call.ordinal) == ("interactive-break", 1)


def test_a_box_with_no_tool_name_is_reported_not_guessed():
    """The writer clearly asked for something; they just did not say what."""
    (found,) = find_calls("```paperlog-tool\nsomething I forgot to label\n```", REF)
    assert found.call.tool == UNKNOWN_TOOL
    assert not found.understood
    assert found.call.prompt == "something I forgot to label"


def test_a_tool_word_transcribed_into_the_fence_is_tolerated():
    """The writer heads the box 'TOOL research-request'; the transcriber may
    carry that word into the info string as well as the body."""
    (found,) = find_calls("```paperlog-tool TOOL research-request\nbody\n```", REF)
    assert found.call.tool == "research-request"


def test_a_tilde_fence_is_accepted():
    (found,) = find_calls("~~~paperlog-tool research-request\nbody\n~~~", REF)
    assert found.call.tool == "research-request"


def test_a_typed_marker_inside_a_box_is_one_request_not_two():
    text = "```paperlog-tool research-request\nTOOL research-request\nbody\n```"
    assert len(find_calls(text, REF)) == 1


def test_an_ordinary_code_block_is_not_a_request():
    """Someone writing about code should not accidentally spend money."""
    assert find_calls("```python\nprint('hello')\n```", REF) == []


def test_the_prose_can_be_recovered_without_the_requests():
    """What a draft needs: the writing, without the machinery of asking."""
    prose = strip_calls(transcript())
    assert SURROUNDING[0] in prose and SURROUNDING[1] in prose
    assert "paperlog-tool" not in prose and "black plague" not in prose

    typed = strip_calls(transcript(fenced=False))
    assert "TOOL" not in typed and "END" not in typed
    assert SURROUNDING[1] in typed




# -- the isolation guarantee ------------------------------------------------


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
        self.calls.append({"images": 1, "system": system, "instruction": instruction})
        return self._next("transcribed page")

    def ask(self, *, system, instruction):
        self.calls.append({"images": 0, "system": system, "instruction": instruction})
        return self._next("# A report\n\nBody.")

    def preflight(self):
        return None


def test_a_dispatched_request_carries_the_prompt_and_nothing_else():
    """The load-bearing test for the whole design.

    Everything upstream is arrangement; this is the check that what actually
    goes on the wire to the model answering the request is the request. The
    page it came from does not travel with it, the prose around it does not,
    and neither does which notebook it was written in.
    """
    (found,) = find_calls(transcript(), REF)
    backend = RecordingBackend()
    run_call(found.call, BUILTIN_TOOLS["research-request"], backend=backend)

    (sent,) = backend.calls
    assert sent["images"] == 0, "the page must not travel with the request"
    body = sent["instruction"]
    assert "black plague" in body
    for phrase in SURROUNDING:
        assert phrase not in body
    assert "K7M2QX" not in body, "not even which notebook it came from"


def test_no_model_both_reads_the_page_and_chooses_the_request():
    """The division of labour that makes isolation structural.

    Extraction takes text and returns calls with no model in the loop at all,
    so there is no step where a model can see the surrounding prose *and*
    decide what was asked. Calling it with no backend, no key and no network is
    the proof.
    """
    calls = find_calls(transcript(), REF)
    assert len(calls) == 1 and calls[0].call.prompt == "books on the black plague"


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
