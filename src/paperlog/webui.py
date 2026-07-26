"""A small local web UI for making notebooks.

Choosing a page size, a ruling and a set of margins is a visual decision, and
reading numbers off ``--help`` is a poor way to make it. This serves a form
with a live preview of the actual PDF next to it, so you can see the thing you
are about to print.

Deliberately dependency-free: the server is :mod:`http.server` from the
standard library, and the preview is the real PDF in an ``<iframe>``, rendered
by the browser. No web framework, no rasteriser, no JavaScript bundle.

It binds to the loopback interface only. There is no authentication, and none
is wanted -- this is a tool for one person on one machine.
"""

from __future__ import annotations

import io
import json
import tempfile
import threading
import webbrowser
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Tuple

from . import __version__, build
from .config import (
    PAGE_SIZES,
    RULING_STYLES,
    ConfigError,
    JournalConfig,
)
from .ids import TokenError, new_notebook_id
from .qrcodes import module_size, symbol_modules, worst_case_payload
from .units import MM

#: Pages rendered into the preview. Enough to show a front and a back, and the
#: mirrored gutter between them, without waiting for a 200-page render.
PREVIEW_PAGES = 4


def summarise(config: JournalConfig) -> Dict[str, Any]:
    """The facts worth putting on screen next to the preview."""
    width, height = config.size
    _, _, block_w, block_h = config.text_block("F")
    out: Dict[str, Any] = {
        "notebook_id": config.notebook_id,
        "page_mm": [round(width / MM, 1), round(height / MM, 1)],
        "text_block_mm": [round(block_w / MM), round(block_h / MM)],
        "pages": config.pages,
        "codes": None,
        "warnings": config.warnings(),
    }
    if config.qr.enabled and config.qr.corners:
        payload = worst_case_payload(config)
        modules = symbol_modules(payload, config.qr.error_correction, config.qr.quiet_zone)
        out["codes"] = {
            "corners": config.qr.corners,
            "modules": modules,
            "module_mm": round(module_size(config) / MM, 2),
            "ecc": config.qr.error_correction.upper(),
            "sample": payload,
        }
    if config.imposition == "booklet":
        from .imposition import padded_count

        out["sheets"] = padded_count(config.pages) // 4
    return out


def render_preview(settings: Dict[str, Any]) -> Tuple[bytes, Dict[str, Any]]:
    """Build a few pages of the described notebook, in memory."""
    config = JournalConfig.from_dict(settings)
    summary = summarise(config)

    preview = dict(settings)
    preview["pages"] = min(config.pages, PREVIEW_PAGES)
    # A booklet preview should show pages, not imposed sheets -- the sheets are
    # a printing detail and make the preview unreadable.
    preview["imposition"] = "none"
    preview_config = JournalConfig.from_dict(preview)

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "preview.pdf"
        result = build(preview_config, path, write_manifest=False)
        return result.pdf_path.read_bytes(), summary


def render_bundle(settings: Dict[str, Any]) -> Tuple[bytes, str]:
    """The real thing: a zip holding the journal PDF and its manifest."""
    config = JournalConfig.from_dict(settings)
    with tempfile.TemporaryDirectory() as folder:
        stem = f"{config.notebook_id}"
        result = build(config, Path(folder) / f"{stem}.pdf")
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(result.pdf_path, f"{stem}.pdf")
            if result.manifest_path:
                archive.write(result.manifest_path, f"{stem}.manifest.json")
            archive.writestr(
                f"{stem}.config.json", json.dumps(settings, indent=2) + "\n"
            )
        return buffer.getvalue(), f"{stem}.zip"


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>paper-log</title>
<style>
  :root {
    --bg: #f6f7f9; --panel: #fff; --ink: #1c2430; --muted: #667085;
    --line: #dfe3e8; --accent: #2f6feb; --warn: #8a5a00; --warn-bg: #fff6e0;
    --err: #a11; --err-bg: #fdecec;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #14171c; --panel: #1c2027; --ink: #e6e9ee; --muted: #98a2b3;
      --line: #2c323b; --accent: #6f9bf5; --warn: #f0c26b; --warn-bg: #33280f;
      --err: #f08a8a; --err-bg: #351a1a;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--ink);
    font: 14px/1.5 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  header {
    padding: 14px 20px; border-bottom: 1px solid var(--line);
    display: flex; align-items: baseline; gap: 12px; background: var(--panel);
  }
  header h1 { font-size: 16px; margin: 0; letter-spacing: .01em; }
  header span { color: var(--muted); font-size: 12px; }
  main { display: grid; grid-template-columns: 340px 1fr; gap: 0; height: calc(100vh - 53px); }
  @media (max-width: 860px) { main { grid-template-columns: 1fr; height: auto; } }
  form { padding: 18px 20px 40px; overflow-y: auto; border-right: 1px solid var(--line); }
  fieldset { border: 0; padding: 0; margin: 0 0 22px; }
  legend {
    font-size: 11px; text-transform: uppercase; letter-spacing: .08em;
    color: var(--muted); margin-bottom: 10px; font-weight: 600;
  }
  label { display: block; margin-bottom: 10px; }
  label > span { display: block; font-size: 12px; color: var(--muted); margin-bottom: 3px; }
  input[type=text], input[type=number], select {
    width: 100%; padding: 7px 9px; border: 1px solid var(--line); border-radius: 6px;
    background: var(--panel); color: var(--ink); font: inherit; font-size: 13px;
  }
  input:focus, select:focus { outline: 2px solid var(--accent); outline-offset: -1px; }
  .row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
  .check { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
  .check input { margin: 0; }
  .check span { font-size: 13px; color: var(--ink); }
  .corners { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 6px; }
  .corners label {
    display: flex; align-items: center; gap: 6px; margin: 0; font-size: 12px;
    border: 1px solid var(--line); border-radius: 6px; padding: 6px 8px;
    white-space: nowrap;
  }
  .corners input { flex: none; margin: 0; }
  button {
    font: inherit; border-radius: 6px; border: 1px solid var(--line);
    background: var(--panel); color: var(--ink); padding: 7px 12px; cursor: pointer;
  }
  button:hover { border-color: var(--accent); }
  button.primary { background: var(--accent); border-color: var(--accent); color: #fff; font-weight: 600; }
  button.primary:hover { filter: brightness(1.08); }
  .idrow { display: flex; gap: 8px; }
  .idrow input { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; letter-spacing: .06em; }
  aside { padding: 18px 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 14px; }
  .facts { display: flex; flex-wrap: wrap; gap: 8px; }
  .fact {
    background: var(--panel); border: 1px solid var(--line); border-radius: 6px;
    padding: 6px 10px; font-size: 12px;
  }
  .fact b { font-weight: 600; }
  .fact i { font-style: normal; color: var(--muted); display: block; font-size: 11px; }
  .note { border-radius: 6px; padding: 9px 11px; font-size: 12.5px; }
  .note.warn { background: var(--warn-bg); color: var(--warn); }
  .note.err { background: var(--err-bg); color: var(--err); }
  iframe {
    width: 100%; flex: 1; min-height: 520px; border: 1px solid var(--line);
    border-radius: 8px; background: var(--panel);
  }
  code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11.5px; }
  .actions { display: flex; gap: 10px; align-items: center; }
  .muted { color: var(--muted); font-size: 12px; }
</style>
</head>
<body>
<header>
  <h1>paper-log</h1>
  <span>new notebook &middot; v__VERSION__</span>
</header>
<main>
  <form id="f">
    <fieldset>
      <legend>Notebook</legend>
      <label><span>Title (printed only if you enable it below)</span>
        <input type="text" name="title" value=""></label>
      <label><span>Notebook id &mdash; keep it to reprint the same notebook</span>
        <div class="idrow">
          <input type="text" name="notebook_id" id="nbid" value="__ID__" maxlength="6">
          <button type="button" id="reroll" title="Generate a new id">&#x21bb;</button>
        </div>
      </label>
      <div class="row">
        <label><span>Pages</span><input type="number" name="pages" value="64" min="1" max="8191"></label>
        <label><span>Paper</span><select name="page_size">__SIZES__</select></label>
      </div>
      <div class="check"><input type="checkbox" name="duplex" id="duplex" checked><label for="duplex" style="margin:0"><span style="color:inherit;font-size:13px">Double-sided (mirrors the gutter)</span></label></div>
      <div class="check"><input type="checkbox" name="booklet" id="booklet"><label for="booklet" style="margin:0"><span style="color:inherit;font-size:13px">Booklet imposition (fold &amp; staple)</span></label></div>
    </fieldset>

    <fieldset>
      <legend>Ruling</legend>
      <div class="row">
        <label><span>Style</span><select name="ruling_style">__RULINGS__</select></label>
        <label><span>Spacing (mm)</span><input type="number" name="spacing" value="7" step="0.5" min="1"></label>
      </div>
    </fieldset>

    <fieldset>
      <legend>Margins (mm)</legend>
      <div class="row">
        <label><span>Top</span><input type="number" name="margin_top" value="22" step="1" min="0"></label>
        <label><span>Bottom</span><input type="number" name="margin_bottom" value="22" step="1" min="0"></label>
        <label><span>Inner (binding)</span><input type="number" name="margin_inner" value="20" step="1" min="0"></label>
        <label><span>Outer</span><input type="number" name="margin_outer" value="12" step="1" min="0"></label>
      </div>
    </fieldset>

    <fieldset>
      <legend>Codes</legend>
      <div class="row">
        <label><span>Size (mm)</span><input type="number" name="qr_size" value="16" step="1" min="5"></label>
        <label><span>Inset (mm)</span><input type="number" name="qr_inset" value="5" step="1" min="0"></label>
      </div>
      <span class="muted" style="display:block;margin-bottom:6px">Corners &mdash; four are needed to undo the perspective of a photo</span>
      <div class="corners">
        <label><input type="checkbox" name="corner" value="TL" checked> top-left</label>
        <label><input type="checkbox" name="corner" value="TR" checked> top-right</label>
        <label><input type="checkbox" name="corner" value="BL" checked> bottom-left</label>
        <label><input type="checkbox" name="corner" value="BR" checked> bottom-right</label>
      </div>
      <div class="check" style="margin-top:10px"><input type="checkbox" name="caption" id="caption"><label for="caption" style="margin:0"><span style="color:inherit;font-size:13px">Print the page id beside each code</span></label></div>
    </fieldset>

    <fieldset>
      <legend>On the page</legend>
      <div class="check"><input type="checkbox" name="date_line" id="dl" checked><label for="dl" style="margin:0"><span style="color:inherit;font-size:13px">Date line</span></label></div>
      <div class="check"><input type="checkbox" name="page_number" id="pn" checked><label for="pn" style="margin:0"><span style="color:inherit;font-size:13px">Page number</span></label></div>
      <div class="check"><input type="checkbox" name="show_title" id="st"><label for="st" style="margin:0"><span style="color:inherit;font-size:13px">Title in the header</span></label></div>
    </fieldset>
  </form>

  <aside>
    <div class="actions">
      <button class="primary" type="button" id="download">Download notebook</button>
      <span class="muted" id="status">&nbsp;</span>
    </div>
    <div class="facts" id="facts"></div>
    <div id="notes"></div>
    <iframe id="preview" title="Preview"></iframe>
  </aside>
</main>

<script>
const form = document.getElementById('f');
const frame = document.getElementById('preview');
const facts = document.getElementById('facts');
const notes = document.getElementById('notes');
const status = document.getElementById('status');
let current = null, timer = null, generation = 0;

function settings() {
  const d = new FormData(form);
  const corners = [...form.querySelectorAll('input[name=corner]:checked')].map(c => c.value);
  const on = n => form.querySelector(`[name=${n}]`).checked;
  return {
    title: d.get('title') || '',
    notebook_id: (d.get('notebook_id') || '').toUpperCase(),
    pages: Number(d.get('pages')) || 1,
    page_size: d.get('page_size'),
    duplex: on('duplex'),
    imposition: on('booklet') ? 'booklet' : 'none',
    margins: {
      top: Number(d.get('margin_top')), bottom: Number(d.get('margin_bottom')),
      inner: Number(d.get('margin_inner')), outer: Number(d.get('margin_outer')),
    },
    ruling: { style: d.get('ruling_style'), spacing: Number(d.get('spacing')) },
    qr: {
      enabled: corners.length > 0, corners: corners.length ? corners : ['TL'],
      size: Number(d.get('qr_size')), inset: Number(d.get('qr_inset')),
      caption: on('caption'),
    },
    furniture: {
      date_line: on('date_line'), page_number: on('page_number'),
      title: on('show_title'), title_text: d.get('title') || '',
    },
  };
}

function fact(label, value) {
  return `<div class="fact"><i>${label}</i><b>${value}</b></div>`;
}

async function refresh() {
  const mine = ++generation;
  status.textContent = 'rendering…';
  let response;
  try {
    response = await fetch('/preview', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(settings()),
    });
  } catch (e) { status.textContent = 'server gone'; return; }
  if (mine !== generation) return;           // a newer edit already won

  if (!response.ok) {
    const problem = await response.json();
    notes.innerHTML = `<div class="note err">${problem.error}</div>`;
    facts.innerHTML = '';
    status.textContent = '';
    return;
  }
  const summary = JSON.parse(atob(response.headers.get('X-Summary')));
  const blob = await response.blob();
  if (current) URL.revokeObjectURL(current);
  current = URL.createObjectURL(blob);
  frame.src = current + '#toolbar=0&view=Fit';

  const bits = [
    fact('notebook', summary.notebook_id),
    fact('paper', summary.page_mm[0] + ' × ' + summary.page_mm[1] + ' mm'),
    fact('writing area', summary.text_block_mm[0] + ' × ' + summary.text_block_mm[1] + ' mm'),
    fact('pages', summary.pages),
  ];
  if (summary.sheets) bits.push(fact('folded sheets', summary.sheets));
  if (summary.codes) {
    bits.push(fact('code', summary.codes.modules + '×' + summary.codes.modules
      + ' @ ' + summary.codes.module_mm + 'mm, ECC ' + summary.codes.ecc));
  } else {
    bits.push(fact('code', 'none'));
  }
  facts.innerHTML = bits.join('');
  notes.innerHTML = (summary.warnings || [])
    .map(w => `<div class="note warn">${w}</div>`).join('');
  status.textContent = '';
}

function schedule() { clearTimeout(timer); timer = setTimeout(refresh, 220); }
form.addEventListener('input', schedule);
form.addEventListener('change', schedule);

document.getElementById('reroll').addEventListener('click', async () => {
  const r = await fetch('/new-id');
  document.getElementById('nbid').value = (await r.json()).id;
  refresh();
});

document.getElementById('download').addEventListener('click', async () => {
  status.textContent = 'building…';
  const response = await fetch('/build', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(settings()),
  });
  if (!response.ok) {
    const problem = await response.json();
    notes.innerHTML = `<div class="note err">${problem.error}</div>`;
    status.textContent = '';
    return;
  }
  const name = response.headers.get('X-Filename') || 'notebook.zip';
  const blob = await response.blob();
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = name;
  link.click();
  URL.revokeObjectURL(link.href);
  status.textContent = 'saved ' + name;
});

refresh();
</script>
</body>
</html>
"""


def _page_html() -> str:
    sizes = "".join(
        f'<option value="{name}"{" selected" if name == "a5" else ""}>{name}</option>'
        for name in sorted(PAGE_SIZES)
    )
    rulings = "".join(
        f'<option value="{name}"{" selected" if name == "ruled" else ""}>{name}</option>'
        for name in RULING_STYLES
    )
    return (
        PAGE.replace("__SIZES__", sizes)
        .replace("__RULINGS__", rulings)
        .replace("__ID__", new_notebook_id())
        .replace("__VERSION__", __version__)
    )


class Handler(BaseHTTPRequestHandler):
    server_version = f"paper-log/{__version__}"

    def log_message(self, format, *args):  # noqa: A002 - signature is fixed
        pass  # a local design tool does not need an access log

    def _send(self, code: int, body: bytes, content_type: str, headers=None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: Dict[str, Any]) -> None:
        self._send(code, json.dumps(payload).encode("utf-8"), "application/json")

    def _read_settings(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self) -> None:  # noqa: N802 - http.server's naming
        if self.path.startswith("/new-id"):
            self._json(200, {"id": new_notebook_id()})
        elif self.path in ("/", "/index.html"):
            self._send(200, _page_html().encode("utf-8"), "text/html; charset=utf-8")
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        import base64

        try:
            settings = self._read_settings()
        except ValueError:
            self._json(400, {"error": "malformed request"})
            return

        try:
            if self.path.startswith("/preview"):
                pdf, summary = render_preview(settings)
                self._send(
                    200, pdf, "application/pdf",
                    {"X-Summary": base64.b64encode(
                        json.dumps(summary).encode("utf-8")).decode("ascii")},
                )
            elif self.path.startswith("/build"):
                blob, filename = render_bundle(settings)
                self._send(200, blob, "application/zip", {"X-Filename": filename})
            else:
                self._json(404, {"error": "not found"})
        except (ConfigError, TokenError) as exc:
            self._json(400, {"error": str(exc)})
        except Exception as exc:  # a design tool should not die on a bad number
            self._json(500, {"error": f"{type(exc).__name__}: {exc}"})


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    """Run the UI until interrupted."""
    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{server.server_port}/"
    print(f"paper-log UI on {url}")
    print("press ctrl-c to stop")
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
