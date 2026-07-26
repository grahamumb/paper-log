# paper-log

QR code identified journal pages to auto catalog my writing.

Generate printable journals whose pages identify themselves. Every page carries
QR codes in its corners encoding which notebook it belongs to, which page it
is, whether it is a front or a back, and which corner the code is in. Scan a
stack of handwritten pages and software can file them without you naming a
single file — in the right order, right way up, even if they went through the
feeder shuffled and upside down.

Pages, paper size, ruling, margins and the codes themselves are all
configurable, and the layout is checked before you print: paper-log tells you
if a code would land on top of your writing or come out too small to scan.

```
paperlog build --preset a5-ruled --pages 64 --out journal.pdf
```

## Install

```bash
pip install -e .                 # generating journals
pip install -e '.[verify]'       # + reading the codes back off a PDF
pip install -e '.[dev]'          # + the test suite
```

## Quick start

```bash
paperlog presets                             # sizes, rulings, ready-made recipes
paperlog build --preset a5-dot --out j.pdf   # a 64-page A5 dot-grid journal
paperlog verify j.pdf                        # prove the codes actually decode
```

Or start from a config file and edit it:

```bash
paperlog init -o journal.yaml
paperlog build -c journal.yaml --out journal.pdf
```

Each build writes two files: `journal.pdf` to print, and
`journal.manifest.json` listing every page and the exact string in each of its
codes — the map your scanning pipeline reads.

## The page identifier

Each code holds one token:

```
PL1:K7M2QX4A:42:F:TR:A19C
 │      │     │  │ │   └── CRC-16/CCITT-FALSE of everything before it
 │      │     │  │ └────── corner: TL, TR, BL or BR
 │      │     │  └──────── side: F (front/recto) or B (back/verso)
 │      │     └─────────── page number, 1-based
 │      └───────────────── notebook id, 8 Crockford base32 characters
 └──────────────────────── format version
```

Four decisions worth knowing about, since they are what makes the scanning end
easy:

**The corner is inside the token.** A scanner that reads even one code knows
which corner it read, so it can work out the page's rotation without guessing.
Codes in opposite corners give two anchor points; all four give a full set for
perspective-correcting a photo.

**Every character is QR "alphanumeric".** Sticking to `0-9 A-Z` and `$%*+-./:`
keeps the symbol a whole version smaller than byte mode would — 25×25 modules
instead of 29×29 — which means bigger, more readable modules in the same space.

**There is a checksum.** QR has its own error correction, but the CRC also
covers tokens that get retyped, OCR'd, or truncated by a URL handler. A misread
page is rejected rather than silently filed as a different one.

**Notebook ids use Crockford base32** — no I, L, O or U — so an id stays
unambiguous written on a cover and read back later. `paperlog decode` accepts
the lenient spellings (`0`/`O`, `1`/`I`/`L`).

Decode one by hand at any time:

```console
$ paperlog decode PL1:K7M2QX4A:42:F:TR:DF6B
notebook  K7M2QX4A
page      42 (front)
corner    top-right
token     PL1:K7M2QX4A:42:F:TR:DF6B  ✓ checksum ok
```

Reprinting a notebook is just reusing its id — `--notebook-id K7M2QX4A`
regenerates identical tokens.

## Scanning: sizes and resolution

Reliability comes down to one number: **the printed size of a single QR
module**. The defaults give 0.45mm, and `paperlog build` warns if your settings
drop below 0.4mm.

Measured against this repo's own output, decoded with OpenCV:

| Code size | Module | 200dpi | 250dpi | 300dpi |
|-----------|--------|--------|--------|--------|
| 13mm (default) | 0.45mm | no | marginal | yes |
| 17mm | 0.59mm | yes | yes | yes |

So: **scan at 300dpi**, or raise `qr.size` to ~17mm if you are stuck at 200.
Anything under about 0.33mm per module will not survive real paper regardless
of resolution — ink spreads and the scanner's optics blur.

Longer payloads mean denser symbols. `payload: "{token}"` is the most compact;
a URL template like `https://notes.example/p/{token}` is friendlier to a phone
camera but pushes the symbol to the next QR version, so give it more room.

Every code in a run is printed at the same module pitch — sized for the highest
page number — so a scanner tuned on page 1 still works on page 200.

### Checking before you print

```console
$ paperlog verify journal.pdf
scanned   64 page(s) of journal.pdf at 300 dpi
decoded   128 code(s)
expected  128 code(s) across the whole document
ok        every code decoded and matches the manifest
```

This rasterises the finished PDF and runs a real decoder over it, then checks
what came back against the manifest. Worth doing before committing a long print
run to paper.

One caveat, because it shows up in the output: OpenCV's bundled QR detector is
noticeably less capable than the symbols it reads. It misses one code of four
on a large sheet and then reads it perfectly from a crop; it fails at 600dpi on
what it reads at 300; and on a handful of payloads it refuses a *pristine*
symbol at one scale while decoding the identical symbol at another. None of
that is about print quality.

So a page that comes up short is retried, in increasing order of help:

1. the whole page, multi-symbol — what a naive scanner does;
2. overlapping tiles, single-symbol — what a careful one does;
3. the exact region the manifest points at, resampled to a whole number of
   pixels per module;
4. failing all of those, the printed modules are compared bit for bit against
   the symbol the expected payload encodes — no decoder involved.

Steps 3 and 4 use the manifest as a map, which makes them *stricter* than a
blind scan rather than looser: the symbol still has to be the one that belongs
in that spot, so a wrong, misplaced or missing code fails either way. Every
retry switches off below 3 pixels per module, so codes too small to survive
real paper are never rescued by resampling.

`retried … needed a targeted crop` and `bit-check … confirmed correct module by
module` are therefore reports about the decoder, not warnings about your pages.

## Configuration

Lengths accept `mm` (the default), `cm`, `in` or `pt`: `7mm`, `0.25in`, `18pt`.
Unknown keys are errors, with a suggestion — a typo will not silently print you
the wrong notebook.

```yaml
notebook_id: K7M2QX4A
title: "Field notes"
pages: 64
duplex: true          # odd pages are fronts, even are backs; margins mirror
page_size: a5         # a3 a4 a5 a6 b5 b6 letter half-letter legal
                      # junior-legal pocket travelers travelers-passport
                      # or "148x210mm", "5.5in x 8.5in"
landscape: false
imposition: none      # or "booklet"

margins:
  top: 18mm
  bottom: 20mm
  inner: 20mm         # binding edge; swaps sides each page when duplex
  outer: 12mm         # (also accepts left/right, or "all")

ruling:
  style: ruled        # blank ruled dotted grid cornell
  spacing: 7mm
  line_width: 0.4
  color: "#9eabb9"    # hex, a 0-1 gray, or [r, g, b]
  first_line_offset: 0mm
  margin_rule: false  # vertical rule at the binding edge, legal-pad style
  margin_rule_offset: 12mm
  cue_column: 40mm    # cornell only
  summary_band: 45mm  # cornell only

qr:
  enabled: true
  corners: [TL, BR]   # any of TL TR BL BR, or "all"
  size: 13mm          # footprint including the quiet zone
  inset: 5mm          # from the paper edge
  quiet_zone: 2       # in modules (see below)
  error_correction: m # l m q h
  payload: "{token}"  # or "https://notes.example/p/{token}"
  caption: false      # print the page id in small type beside each code,
                      # auto-shrunk to fit and shortened to the page
                      # number where the header leaves no room
  color: "#000000"

fiducials:
  style: bracket      # bracket square cross none
  corners: auto       # "auto" = the corners without a QR code
  size: 5mm
  inset: 5mm

furniture:
  date_line: true
  date_label: date
  page_number: true
  page_number_format: "{page}"
  title: false
  header_rule: true
  font: Helvetica      # one of the 14 standard PDF fonts; nothing is embedded
  size: 7.5
```

Anything in the file can be overridden on the command line —
`paperlog build -c journal.yaml --pages 96 --ruling dotted --spacing 5mm`.
Run `paperlog build --help` for the full list.

**On `quiet_zone: 2`.** The QR spec asks for 4 modules of clear space, on the
assumption something might be printed against the symbol. Here the code sits
alone in a margin millimetres wide, so the paper supplies the real quiet zone
and a nominal 2 buys a ~15% bigger module in the same footprint. Raise it if
you move codes into a busy area.

**Fiducials** are registration marks in the corners that have no QR code. A QR
symbol is already a precise anchor, but only where one is printed; the marks
let a crop-and-deskew step find all four page corners even when a code is
smudged or covered by the writer's hand.

### Margins and corner clearance

A code overlaps your writing only if it overlaps in *both* axes, so a corner
can be cleared by widening either the top/bottom margin or the side margin next
to it — whichever costs less writing room. paper-log checks each corner
individually, on fronts and backs, and says which knob to turn:

```
warning:  The TR code overlaps the writing area on front pages. Give it 18mm of
          clearance: raise the top margin from 12mm, widen the side margin
          beside it, or shrink qr.size / qr.inset.
```

The default margins clear all four corners at the default code size.

## Booklets

`imposition: booklet` lays the pages out two-up on double-width sheets in
saddle-stitch order, with a dashed fold line and trim ticks:

```bash
paperlog build --preset a5-booklet --pages 32 --out booklet.pdf
```

Print double-sided flipping on the **short edge**, stack the sheets in order,
fold the pile in half, staple through the spine. Two A5 pages make an A4 sheet;
two half-letter make letter. Page counts are padded up to a multiple of 4,
since that is what a folded sheet holds.

## Python API

```python
from paperlog import JournalConfig, build

config = JournalConfig.from_dict({
    "page_size": "a5",
    "pages": 64,
    "ruling": {"style": "dotted", "spacing": "5mm"},
    "qr": {"corners": "all"},
})

for note in config.warnings():
    print("warning:", note)

result = build(config, "journal.pdf")
print(result.pdf_path, result.manifest_path)
for page in result.pages:
    print(page.number, page.side, page.token)
```

## The manifest

```json
{
  "manifest_version": 1,
  "notebook": { "id": "K7M2QX4A", "page_count": 64, "duplex": true },
  "page": { "width_mm": 148.0, "height_mm": 210.0, "ruling": "ruled" },
  "token_format": {
    "grammar": "PL1:<notebook>:<page>:<F|B>:<TL|TR|BL|BR>:<crc16>",
    "checksum": "CRC-16/CCITT-FALSE over the token up to but excluding the final colon, hex, upper case"
  },
  "qr": {
    "corners": ["TL", "BR"],
    "modules": 29,
    "module_size_mm": 0.448,
    "geometry": {
      "TL": { "x_mm": 5.0, "y_mm": 192.0, "size_mm": 13.0 },
      "BR": { "x_mm": 130.0, "y_mm": 5.0, "size_mm": 13.0 }
    }
  },
  "pages": [
    {
      "page": 1,
      "side": "F",
      "token": "PL1:K7M2QX4A:1:F:TL:0370",
      "codes": {
        "TL": "PL1:K7M2QX4A:1:F:TL:0370",
        "BR": "PL1:K7M2QX4A:1:F:BR:595A"
      }
    }
  ]
}
```

Geometry is in millimetres from the bottom-left of the page, which is what a
deskewed scan gives you. `token_format` is spelled out in full so the scanning
side can be written against the manifest alone — and `paperlog/ids.py`
implements the whole scheme, CRC included, in one dependency-free file if you
would rather port it.

## Tests

```bash
python -m pytest
```

The suite includes an end-to-end check that renders journals, rasterises them
and decodes the codes back — including negative cases, so the check can fail:
undersized codes, mismatched manifests, and scans at too low a resolution are
all expected to be rejected. Those tests skip if the `[verify]` extras are not
installed.
