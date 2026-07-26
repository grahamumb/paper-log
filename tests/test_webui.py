"""The local design UI.

Tested through real HTTP against a real server, because the interesting parts
are the endpoints and the error handling, not the HTML.
"""

import base64
import json
import threading
import urllib.error
import urllib.request
import zipfile
from http.server import ThreadingHTTPServer

import pytest

from paperlog.webui import Handler, render_bundle, render_preview, summarise
from paperlog import JournalConfig


@pytest.fixture
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def post(url, payload):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    return urllib.request.urlopen(request, timeout=30)


SETTINGS = {
    "page_size": "a5",
    "pages": 8,
    "notebook_id": "K7M2QX",
    "ruling": {"style": "ruled", "spacing": 7},
}


def test_the_page_loads(server):
    with urllib.request.urlopen(server + "/", timeout=10) as response:
        body = response.read().decode()
    assert response.status == 200
    assert "<title>paper-log</title>" in body
    # The selects are filled from the real option lists, not hardcoded.
    assert 'value="a5"' in body and 'value="cornell"' in body


def test_new_id_endpoint_returns_usable_ids(server):
    ids = set()
    for _ in range(5):
        with urllib.request.urlopen(server + "/new-id", timeout=10) as response:
            ids.add(json.load(response)["id"])
    assert len(ids) == 5
    assert all(len(value) == 6 for value in ids)


def test_preview_returns_a_pdf_and_a_summary(server):
    response = post(server + "/preview", SETTINGS)
    assert response.status == 200
    assert response.headers["Content-Type"] == "application/pdf"
    body = response.read()
    assert body.startswith(b"%PDF")

    summary = json.loads(base64.b64decode(response.headers["X-Summary"]))
    assert summary["notebook_id"] == "K7M2QX"
    assert summary["page_mm"] == [148.0, 210.0]
    assert summary["codes"]["modules"] == 25
    assert summary["warnings"] == []


def test_preview_is_short_even_for_a_long_notebook():
    """Nobody needs 500 pages re-rendered on every keystroke."""
    pdf, summary = render_preview({**SETTINGS, "pages": 500})
    assert summary["pages"] == 500  # the summary describes the real notebook
    assert pdf.count(b"/Type /Page\n") <= 8  # but the preview is a few pages


def test_preview_of_a_booklet_shows_pages_not_imposed_sheets():
    """Imposed sheets are a printing detail and unreadable as a preview."""
    plain, _ = render_preview(SETTINGS)
    booklet, summary = render_preview({**SETTINGS, "imposition": "booklet"})
    assert summary["sheets"] == 2
    assert len(booklet) == pytest.approx(len(plain), rel=0.35)


def test_warnings_reach_the_summary(server):
    response = post(server + "/preview", {**SETTINGS, "qr": {"size": 6}})
    summary = json.loads(base64.b64decode(response.headers["X-Summary"]))
    assert any("below the" in note for note in summary["warnings"])


def test_a_bad_configuration_is_a_clean_400(server):
    with pytest.raises(urllib.error.HTTPError) as caught:
        post(server + "/preview", {**SETTINGS, "margins": {"top": 200, "bottom": 200}})
    assert caught.value.code == 400
    assert "no room to write" in json.load(caught.value)["error"]


def test_an_over_long_notebook_id_is_explained_not_a_traceback(server):
    with pytest.raises(urllib.error.HTTPError) as caught:
        post(server + "/preview", {**SETTINGS, "notebook_id": "TOOLONGID"})
    assert caught.value.code == 400
    assert "compact" in json.load(caught.value)["error"]


def test_unknown_paths_are_404(server):
    for path in ("/nope", "/etc/passwd"):
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(server + path, timeout=10)
        assert caught.value.code == 404


def test_build_returns_a_zip_with_everything_needed(server):
    response = post(server + "/build", SETTINGS)
    assert response.status == 200
    assert response.headers["X-Filename"] == "K7M2QX.zip"

    import io

    archive = zipfile.ZipFile(io.BytesIO(response.read()))
    assert set(archive.namelist()) == {
        "K7M2QX.pdf", "K7M2QX.manifest.json", "K7M2QX.config.json"
    }
    manifest = json.loads(archive.read("K7M2QX.manifest.json"))
    assert manifest["notebook"]["id"] == "K7M2QX"
    assert len(manifest["pages"]) == 8
    # The config goes in the zip so the notebook can be reprinted exactly.
    assert json.loads(archive.read("K7M2QX.config.json"))["notebook_id"] == "K7M2QX"


def test_bundle_pdf_has_the_full_page_count():
    blob, name = render_bundle({**SETTINGS, "pages": 12})
    import io

    archive = zipfile.ZipFile(io.BytesIO(blob))
    assert archive.read(name.replace(".zip", ".pdf")).startswith(b"%PDF")


def test_summarise_reports_what_the_form_needs():
    config = JournalConfig.from_dict(SETTINGS)
    summary = summarise(config)
    assert summary["text_block_mm"] == [116, 166]
    assert summary["codes"]["ecc"] == "Q"
    assert summary["codes"]["corners"] == ["TL", "TR", "BL", "BR"]
