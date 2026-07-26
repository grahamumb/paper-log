"""Command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import __version__, build
from .config import (
    IMPOSITIONS,
    PAGE_SIZES,
    RULING_STYLES,
    ConfigError,
    JournalConfig,
    deep_merge,
    load_config,
)
from .ids import CORNER_NAMES, TokenError, decode, new_notebook_id
from .imposition import padded_count
from .units import MM, UnitError

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".tif", ".tiff", ".webp", ".bmp"}

PRESETS: Dict[str, Dict[str, Any]] = {
    "a5-ruled": {
        "page_size": "a5",
        "pages": 64,
        "ruling": {"style": "ruled", "spacing": "7mm"},
    },
    "a5-dot": {
        "page_size": "a5",
        "pages": 64,
        "ruling": {"style": "dotted", "spacing": "5mm"},
    },
    "a4-grid": {
        "page_size": "a4",
        "pages": 48,
        "ruling": {"style": "grid", "spacing": "5mm"},
    },
    "pocket-dot": {
        "page_size": "pocket",
        "pages": 48,
        "ruling": {"style": "dotted", "spacing": "5mm"},
        "margins": {"top": "16mm", "bottom": "16mm", "inner": "14mm", "outer": "9mm"},
        "qr": {"size": "12mm", "inset": "3mm", "corners": "TL,BR"},
        "fiducials": {"size": "4mm", "inset": "3mm"},
    },
    "letter-cornell": {
        "page_size": "letter",
        "pages": 40,
        "ruling": {"style": "cornell", "spacing": "8mm"},
    },
    "a5-booklet": {
        "page_size": "a5",
        "pages": 32,
        "imposition": "booklet",
        "ruling": {"style": "ruled", "spacing": "7mm"},
    },
}

SAMPLE_CONFIG = """\
# paper-log journal definition. Lengths accept mm (default), cm, in, or pt.
# Generate a fresh id with `paperlog new-id`; keep it stable to reprint the
# same notebook, change it whenever you start a new one.
notebook_id: {notebook_id}
title: ""
pages: 64
duplex: true          # odd pages are fronts, even pages backs; margins mirror
page_size: a5         # a4 a5 a6 b5 letter half-letter pocket travelers, or 148x210mm
imposition: none      # or "booklet" to print folded sheets two-up

margins:
  top: 22mm
  bottom: 22mm        # top/bottom clear the codes, whatever the sides do
  inner: 20mm         # binding edge
  outer: 12mm

ruling:
  style: ruled        # blank ruled dotted grid cornell
  spacing: 7mm
  line_width: 0.4
  color: "#9eabb9"
  margin_rule: false

qr:
  enabled: true
  corners: all        # all four: what a phone photo needs to undo perspective
  size: 16mm          # footprint including the quiet zone
  inset: 5mm          # from the paper edge
  error_correction: q # 25% of the symbol can be lost and still decode
  token_format: compact   # 13 chars, 21x21 modules; or "readable"
  payload: "{{token}}"  # or e.g. "https://notes.example/p/{{token}}"
  caption: false      # print the page id in small type beside each code

fiducials:
  style: bracket      # bracket square cross none
  corners: auto       # the corners without a QR code
  size: 5mm
  inset: 5mm

furniture:
  date_line: true
  date_label: date
  page_number: true
  header_rule: true
"""


def _add_build_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-c", "--config", type=Path, help="YAML or JSON journal definition")
    parser.add_argument("-o", "--out", type=Path, default=Path("journal.pdf"), help="output PDF (default: journal.pdf)")
    parser.add_argument("--preset", choices=sorted(PRESETS), help="start from a named preset")
    parser.add_argument("--manifest", type=Path, help="manifest path (default: alongside the PDF)")
    parser.add_argument("--no-manifest", action="store_true", help="skip writing the manifest")
    parser.add_argument("--dry-run", action="store_true", help="validate and report, write nothing")

    page = parser.add_argument_group("page")
    page.add_argument("--pages", type=int, help="number of journal pages")
    page.add_argument("--page-size", help="named size or WxH, e.g. a5 or 148x210mm")
    page.add_argument("--landscape", action="store_true", default=None)
    page.add_argument("--duplex", dest="duplex", action="store_true", default=None)
    page.add_argument("--no-duplex", dest="duplex", action="store_false")
    page.add_argument("--imposition", choices=IMPOSITIONS)
    page.add_argument("--notebook-id", help="reuse an existing notebook id")
    page.add_argument("--title")

    margins = parser.add_argument_group("margins")
    margins.add_argument("--margin", help="all four margins at once")
    margins.add_argument("--margin-top")
    margins.add_argument("--margin-bottom")
    margins.add_argument("--margin-inner", help="binding edge")
    margins.add_argument("--margin-outer")

    ruling = parser.add_argument_group("ruling")
    ruling.add_argument("--ruling", choices=RULING_STYLES)
    ruling.add_argument("--spacing", help="line or grid pitch, e.g. 7mm")
    ruling.add_argument("--line-color", help="e.g. '#9eabb9' or a 0-1 gray")
    ruling.add_argument("--line-width", type=float)
    ruling.add_argument("--margin-rule", action="store_true", default=None, help="vertical rule at the binding edge")

    codes = parser.add_argument_group("codes")
    codes.add_argument("--no-qr", action="store_true", help="omit the QR codes entirely")
    codes.add_argument("--qr-corners", help="e.g. 'TL,BR' or 'all'")
    codes.add_argument("--qr-size")
    codes.add_argument("--qr-inset")
    codes.add_argument("--qr-ecc", choices=("l", "m", "q", "h"))
    codes.add_argument("--qr-token-format", choices=("compact", "readable"),
                       help="compact is 21x21 modules; readable is the legible PL1:... form")
    codes.add_argument("--qr-payload", help="payload template, default '{token}'")
    codes.add_argument("--qr-caption", action="store_true", default=None, help="print the page id under each code")
    codes.add_argument("--fiducials", dest="fiducial_style", choices=("bracket", "square", "cross", "none"))


def _nest(overrides: Dict[str, Any], path: str, value: Any) -> None:
    """Set ``a.b.c`` in a nested dict, skipping ``None`` (meaning 'unset')."""
    if value is None:
        return
    keys = path.split(".")
    cursor = overrides
    for key in keys[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[keys[-1]] = value


def overrides_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    _nest(overrides, "pages", args.pages)
    _nest(overrides, "page_size", args.page_size)
    _nest(overrides, "landscape", args.landscape)
    _nest(overrides, "duplex", args.duplex)
    _nest(overrides, "imposition", args.imposition)
    _nest(overrides, "notebook_id", args.notebook_id)
    _nest(overrides, "title", args.title)

    _nest(overrides, "margins.all", args.margin)
    _nest(overrides, "margins.top", args.margin_top)
    _nest(overrides, "margins.bottom", args.margin_bottom)
    _nest(overrides, "margins.inner", args.margin_inner)
    _nest(overrides, "margins.outer", args.margin_outer)

    _nest(overrides, "ruling.style", args.ruling)
    _nest(overrides, "ruling.spacing", args.spacing)
    _nest(overrides, "ruling.color", args.line_color)
    _nest(overrides, "ruling.line_width", args.line_width)
    _nest(overrides, "ruling.margin_rule", args.margin_rule)

    if args.no_qr:
        _nest(overrides, "qr.enabled", False)
    _nest(overrides, "qr.corners", args.qr_corners)
    _nest(overrides, "qr.size", args.qr_size)
    _nest(overrides, "qr.inset", args.qr_inset)
    _nest(overrides, "qr.error_correction", args.qr_ecc)
    _nest(overrides, "qr.payload", args.qr_payload)
    _nest(overrides, "qr.token_format", args.qr_token_format)
    _nest(overrides, "qr.caption", args.qr_caption)
    _nest(overrides, "fiducials.style", args.fiducial_style)
    return overrides


def resolve_config(args: argparse.Namespace) -> JournalConfig:
    overrides = overrides_from_args(args)
    if args.preset:
        overrides = deep_merge(PRESETS[args.preset], overrides)
    return load_config(args.config, overrides)


def command_build(args: argparse.Namespace) -> int:
    config = resolve_config(args)
    width, height = config.size

    print(f"notebook  {config.notebook_id}" + (f"  — {config.title}" if config.title else ""))
    print(f"paper     {width / MM:.0f} × {height / MM:.0f} mm, {config.ruling.style}, {config.pages} pages")

    if config.qr.enabled and config.qr.corners:
        from .qrcodes import module_size, symbol_modules, worst_case_payload

        payload = worst_case_payload(config)
        modules = symbol_modules(payload, config.qr.error_correction, config.qr.quiet_zone)
        corners = ", ".join(CORNER_NAMES[corner] for corner in config.qr.corners)
        print(
            f"codes     {corners} — {modules}×{modules} modules at "
            f"{module_size(config) / MM:.2f} mm, ecc {config.qr.error_correction.upper()}"
        )
        print(f"          sample payload: {payload}")
    else:
        print("codes     none")

    if config.imposition == "booklet":
        total = padded_count(config.pages)
        print(f"sheets    {total // 4} folded sheet(s), printed two-up double-sided")

    for note in config.warnings():
        print(f"warning:  {note}", file=sys.stderr)

    if args.dry_run:
        print("dry run   nothing written")
        return 0

    result = build(
        config,
        args.out,
        manifest_path=args.manifest,
        write_manifest=not args.no_manifest,
    )
    print(f"wrote     {result.pdf_path}")
    if result.manifest_path:
        print(f"          {result.manifest_path}")
    return 0


def command_decode(args: argparse.Namespace) -> int:
    ref = decode(args.token)
    if args.json:
        print(json.dumps(
            {
                "notebook": ref.notebook,
                "page": ref.page,
                "side": ref.side,
                "corner": ref.corner,
                "token": ref.token,
            },
            indent=2,
        ))
    else:
        side = "front" if ref.side == "F" else "back"
        print(f"notebook  {ref.notebook}")
        print(f"page      {ref.page} ({side})")
        print(f"corner    {ref.corner_name}")
        print(f"token     {ref.token}  ✓ checksum ok")
    return 0


def command_verify(args: argparse.Namespace) -> int:
    from .scancheck import BackendMissing, verify

    manifest_path: Optional[Path] = args.manifest
    if manifest_path is None:
        guess = args.pdf.with_suffix(".manifest.json")
        manifest_path = guess if guess.exists() else None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path else None

    try:
        report = verify(args.pdf, manifest, dpi=args.dpi, limit=args.pages)
    except BackendMissing as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"scanned   {report.pages_scanned} page(s) of {args.pdf} at {args.dpi} dpi")
    print(f"decoded   {report.decoded} code(s)")
    if manifest is None:
        print("note      no manifest found, so codes were read but not cross-checked")
        for scan in report.scans:
            for ref in scan.refs:
                print(f"  page {scan.index + 1}: {ref.notebook} p{ref.page}{ref.side} {ref.corner}")
        return 0 if report.decoded else 1

    print(f"expected  {report.expected} code(s) across the whole document")
    if report.tiled_pages or report.targeted_pages:
        print(
            f"retried   {report.tiled_pages} page(s) needed tiling, "
            f"{report.targeted_pages} needed a targeted crop "
            "(decoder limitation, not a defect in the page)"
        )
    if report.bit_verified_codes:
        print(
            f"bit-check {report.bit_verified_codes} code(s) this decoder cannot read "
            "at all, confirmed correct module by module against the expected payload"
        )
    if report.short_pages:
        listed = ", ".join(str(number) for number in report.short_pages[:10])
        print(f"short:    PDF page(s) {listed} read fewer codes than expected", file=sys.stderr)
    for token in report.missing[:10]:
        print(f"missing:  {token}", file=sys.stderr)
    for token in report.unexpected[:10]:
        print(f"unexpected: {token}", file=sys.stderr)

    if report.ok:
        scope = "every code" if report.complete else "every sampled code"
        print(f"ok        {scope} accounted for and matches the manifest")
        return 0
    print(
        "failed    codes did not read back cleanly. Try a higher --dpi, or "
        "increase qr.size and rebuild.",
        file=sys.stderr,
    )
    return 1


def command_scan(args: argparse.Namespace) -> int:
    from .capture import (
        BackendMissing,
        CaptureError,
        assemble_pdf,
        load_manifests,
        process,
    )

    photos: List[Path] = []
    for entry in args.photos:
        photos.extend(sorted(p for p in entry.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
                      if entry.is_dir() else [entry])
    if not photos:
        print("error: no images to scan", file=sys.stderr)
        return 1

    sources = args.manifest or _default_manifest_paths(photos)
    try:
        manifests = load_manifests(sources)
    except (CaptureError, BackendMissing) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not manifests:
        print(
            "error: no manifests found. Pass --manifest with the .manifest.json "
            "written next to the journal PDF.",
            file=sys.stderr,
        )
        return 1

    print(f"photos    {len(photos)}")
    print(f"notebooks {', '.join(sorted(manifests))}")
    try:
        run = process(
            photos, manifests, args.out_dir, dpi=args.dpi, enhancement=args.enhance
        )
    except BackendMissing as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for record in run.records:
        page = record.page
        flag = "" if page.confident else "  (low confidence)"
        where = f" -> {record.path}" if record.path else ""
        print(
            f"  {page.name}  {len(page.codes)} code(s), fit {page.residual_mm:.2f}mm"
            f"{flag}{where}"
        )
        for note in page.warnings:
            print(f"      {note}", file=sys.stderr)
    for record in run.superseded:
        print(f"  skipped {record.source.name}: a better shot of {record.page.name} exists")
    for path, reason in run.failures:
        print(f"  failed  {path.name}: {reason}", file=sys.stderr)

    for notebook, gaps in run.missing_pages(manifests).items():
        listed = ", ".join(str(page) for page in gaps[:20])
        more = " ..." if len(gaps) > 20 else ""
        print(f"missing   {notebook}: no photo of page(s) {listed}{more}")

    if args.pdf:
        for notebook, records in run.by_notebook().items():
            target = args.pdf if len(run.by_notebook()) == 1 else args.pdf.with_name(
                f"{args.pdf.stem}-{notebook}{args.pdf.suffix}"
            )
            assemble_pdf(records, target, manifests[notebook.upper()])
            print(f"bound     {target}")

    print(f"recovered {len(run.records)} page(s), {len(run.failures)} photo(s) failed")
    return 1 if run.failures and not run.records else 0


def _default_manifest_paths(photos: Sequence[Path]) -> List[Path]:
    """Look for manifests beside the photos and in the working directory."""
    seen: List[Path] = []
    for folder in [Path.cwd()] + [photo.parent for photo in photos]:
        if folder not in seen:
            seen.append(folder)
    return seen


def command_ui(args: argparse.Namespace) -> int:
    from .webui import serve

    serve(host=args.host, port=args.port, open_browser=not args.no_browser)
    return 0


def command_presets(args: argparse.Namespace) -> int:
    print("presets:")
    for name, preset in sorted(PRESETS.items()):
        ruling = preset.get("ruling", {})
        detail = f"{preset['page_size']}, {ruling.get('style', 'ruled')}, {preset['pages']} pages"
        if preset.get("imposition") == "booklet":
            detail += ", booklet"
        print(f"  {name:<16} {detail}")
    print("\npage sizes:")
    for name, (width, height) in sorted(PAGE_SIZES.items()):
        print(f"  {name:<20} {width / MM:.0f} × {height / MM:.0f} mm")
    print("\nrulings:  " + ", ".join(RULING_STYLES))
    return 0


def command_init(args: argparse.Namespace) -> int:
    text = SAMPLE_CONFIG.format(notebook_id=new_notebook_id())
    if args.out and str(args.out) != "-":
        path = Path(args.out)
        if path.exists() and not args.force:
            print(f"{path} already exists; pass --force to overwrite", file=sys.stderr)
            return 1
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path}")
    else:
        sys.stdout.write(text)
    return 0


def command_new_id(args: argparse.Namespace) -> int:
    for _ in range(max(args.count, 1)):
        print(new_notebook_id())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paperlog",
        description="Generate printable journals whose pages carry scannable QR identifiers.",
    )
    parser.add_argument("--version", action="version", version=f"paper-log {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_cmd = subparsers.add_parser("build", help="render a journal PDF")
    _add_build_arguments(build_cmd)
    build_cmd.set_defaults(func=command_build)

    decode_cmd = subparsers.add_parser("decode", help="parse and verify a page token")
    decode_cmd.add_argument("token", help="token or URL scanned off a page")
    decode_cmd.add_argument("--json", action="store_true")
    decode_cmd.set_defaults(func=command_decode)

    verify_cmd = subparsers.add_parser(
        "verify", help="read the codes back off a rendered PDF and check them"
    )
    verify_cmd.add_argument("pdf", type=Path)
    verify_cmd.add_argument("--manifest", type=Path, help="default: alongside the PDF")
    verify_cmd.add_argument("--dpi", type=int, default=300, help="rasterisation dpi (default: 300)")
    verify_cmd.add_argument("--pages", type=int, help="only scan the first N pages")
    verify_cmd.set_defaults(func=command_verify)

    scan_cmd = subparsers.add_parser(
        "scan", help="flatten and identify photographs of written pages"
    )
    scan_cmd.add_argument("photos", nargs="+", type=Path, help="image files or directories")
    scan_cmd.add_argument("-m", "--manifest", type=Path, action="append",
                          help="manifest file or directory (repeatable)")
    scan_cmd.add_argument("-o", "--out-dir", type=Path, default=Path("pages"),
                          help="where to write flattened pages (default: pages/)")
    scan_cmd.add_argument("--dpi", type=int, default=300, help="output resolution (default: 300)")
    scan_cmd.add_argument("--enhance", choices=("flatten", "scan", "none"), default="flatten",
                          help="even out the lighting (default: flatten)")
    scan_cmd.add_argument("--pdf", type=Path, help="also bind the pages into a PDF, in order")
    scan_cmd.set_defaults(func=command_scan)

    ui_cmd = subparsers.add_parser("ui", help="design a notebook in the browser, with a live preview")
    ui_cmd.add_argument("--port", type=int, default=8765)
    ui_cmd.add_argument("--host", default="127.0.0.1", help="loopback by default; there is no auth")
    ui_cmd.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    ui_cmd.set_defaults(func=command_ui)

    presets_cmd = subparsers.add_parser("presets", help="list presets, page sizes and rulings")
    presets_cmd.set_defaults(func=command_presets)

    init_cmd = subparsers.add_parser("init", help="write a starter config file")
    init_cmd.add_argument("-o", "--out", type=Path, default=Path("journal.yaml"), help="'-' for stdout")
    init_cmd.add_argument("--force", action="store_true")
    init_cmd.set_defaults(func=command_init)

    id_cmd = subparsers.add_parser("new-id", help="generate notebook ids")
    id_cmd.add_argument("-n", "--count", type=int, default=1)
    id_cmd.set_defaults(func=command_new_id)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ConfigError, TokenError, UnitError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"error: {exc.filename}: no such file", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
