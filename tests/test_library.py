"""The notebook library.

The manifest says where the codes sit on the paper, so without it a photograph
cannot be turned back into a page. "Where did I put the manifest" is therefore
the weakest link in the workflow, and it gets weaker with time. These tests
pin down the behaviour that removes the question.
"""

import json

import pytest

from paperlog import JournalConfig, build
from paperlog.cli import main
from paperlog.library import HOME_VARIABLE, entries, forget, home, load, remember


@pytest.fixture
def private_home(tmp_path, monkeypatch):
    """A home of this test's own.

    ``conftest`` already redirects PAPERLOG_HOME for every test; this narrows
    it to a path the test can also inspect directly.
    """
    monkeypatch.setenv(HOME_VARIABLE, str(tmp_path / "home"))
    return tmp_path / "home"


@pytest.fixture
def journal(tmp_path):
    config = JournalConfig.from_dict(
        {"page_size": "a5", "pages": 4, "notebook_id": "K7M2QX", "title": "Field notes"}
    )
    return build(config, tmp_path / "journal.pdf")


def test_home_follows_the_environment(private_home):
    assert home() == private_home


def test_tests_never_touch_the_real_home():
    """The autouse fixture in conftest has to actually be doing its job."""
    from pathlib import Path

    assert home() != Path.home() / ".paperlog"


def test_remembering_a_manifest_files_it_under_its_id(journal):
    filed = remember(journal.manifest_path)
    assert filed is not None
    assert filed.name == "K7M2QX.manifest.json"
    assert json.loads(filed.read_text())["notebook"]["id"] == "K7M2QX"


def test_a_registered_notebook_is_listed(journal):
    remember(journal.manifest_path)
    listed = entries()
    assert [entry.notebook_id for entry in listed] == ["K7M2QX"]
    assert listed[0].pages == 4
    assert listed[0].title == "Field notes"


def test_load_returns_manifests_keyed_by_id(journal):
    remember(journal.manifest_path)
    manifests = load()
    assert set(manifests) == {"K7M2QX"}
    assert manifests["K7M2QX"]["qr"]["geometry"]


def test_registering_twice_updates_rather_than_duplicates(journal, tmp_path):
    remember(journal.manifest_path)
    reprint = build(
        JournalConfig.from_dict(
            {"page_size": "a5", "pages": 12, "notebook_id": "K7M2QX"}
        ),
        tmp_path / "again.pdf",
    )
    remember(reprint.manifest_path)
    listed = entries()
    assert len(listed) == 1
    assert listed[0].pages == 12


def test_forgetting_removes_it_from_the_library_only(journal):
    remember(journal.manifest_path)
    assert forget("k7m2qx") is True
    assert entries() == []
    assert not forget("K7M2QX")
    # The manifest written beside the PDF is untouched.
    assert journal.manifest_path.exists()


def test_rubbish_is_not_registered(tmp_path):
    junk = tmp_path / "notes.json"
    junk.write_text('{"hello": "world"}')
    assert remember(junk) is None
    assert remember(tmp_path / "missing.json") is None
    assert entries() == []


def test_an_empty_library_is_not_an_error():
    assert entries() == []
    assert load() == {}


def test_a_damaged_entry_does_not_break_the_listing(journal, private_home):
    remember(journal.manifest_path)
    (private_home / "notebooks" / "BROKEN.manifest.json").write_text("{not json")
    assert [entry.notebook_id for entry in entries()] == ["K7M2QX"]


# -- through the CLI -------------------------------------------------------


def run(capsys, *argv):
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_building_registers_the_notebook(tmp_path, capsys):
    code, stdout, _ = run(
        capsys, "build", "--pages", "4", "--notebook-id", "K7M2QX",
        "--out", str(tmp_path / "j.pdf"),
    )
    assert code == 0
    assert "filed" in stdout
    assert [entry.notebook_id for entry in entries()] == ["K7M2QX"]


def test_no_library_opts_out(tmp_path, capsys):
    code, stdout, _ = run(
        capsys, "build", "--pages", "4", "--out", str(tmp_path / "j.pdf"), "--no-library"
    )
    assert code == 0
    assert "filed" not in stdout
    assert entries() == []


def test_notebooks_command_lists_and_forgets(tmp_path, capsys):
    run(capsys, "build", "--pages", "4", "--notebook-id", "K7M2QX",
        "--out", str(tmp_path / "j.pdf"))

    code, stdout, _ = run(capsys, "notebooks")
    assert code == 0
    assert "K7M2QX" in stdout and "4 pages" in stdout

    code, stdout, _ = run(capsys, "notebooks", "--forget", "K7M2QX")
    assert code == 0 and "forgot" in stdout

    code, stdout, _ = run(capsys, "notebooks")
    assert "no notebooks registered" in stdout


def test_notebooks_add_registers_a_manifest_from_elsewhere(tmp_path, capsys):
    result = build(
        JournalConfig.from_dict({"pages": 4, "notebook_id": "K7M2QX"}),
        tmp_path / "j.pdf",
    )
    forget("K7M2QX")  # build already filed it; start from nothing
    assert entries() == []

    code, stdout, _ = run(capsys, "notebooks", "--add", str(result.manifest_path))
    assert code == 0 and "registered" in stdout
    assert [entry.notebook_id for entry in entries()] == ["K7M2QX"]


def test_notebooks_add_rejects_a_non_manifest(tmp_path, capsys):
    junk = tmp_path / "junk.json"
    junk.write_text("{}")
    code, _, stderr = run(capsys, "notebooks", "--add", str(junk))
    assert code == 1
    assert "not a readable manifest" in stderr


def test_scan_without_a_manifest_says_what_to_do(tmp_path, capsys):
    pytest.importorskip("cv2", reason="needs the [verify] extras")
    import numpy
    import cv2

    photos = tmp_path / "photos"
    photos.mkdir()
    cv2.imwrite(str(photos / "a.jpg"), numpy.full((600, 400), 200, numpy.uint8))

    code, _, stderr = run(capsys, "scan", str(photos), "-o", str(tmp_path / "out"))
    assert code == 1
    assert "manifest" in stderr and "paperlog notebooks --add" in stderr
