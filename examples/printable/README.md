# Sample notebook — ready to print

A 10-page A5 ruled notebook at paper-log's defaults, for a real print-and-scan
test run.

| File | What it is |
|---|---|
| `PAPER1.pdf` | The thing to print. 10 pages, A5 (148 × 210 mm), portrait. |
| `PAPER1.manifest.json` | **Keep this.** It says where the codes sit on the paper; `paperlog scan` cannot turn a photo back into a page without it. |
| `notebook.yaml` | The config that produced them, so you can change something and rebuild. |

Everything here is at its defaults — 16mm codes in all four corners, 0.64mm per
module, error correction Q. If this prints and scans well, so will a notebook
you generate yourself.

## Printing it

The one thing that matters, whether you use your own printer or a shop:

> **Print at 100% / "actual size". Do not use "fit to page", "shrink to fit", or
> "scale to fit printable area".**

A page reduced to 96% to fit a printer's margins still looks right, and the
scanner will not complain: everything on the page shrinks together, so the
recovered image comes out correctly proportioned either way. What you lose is
decoding margin. Every module shrinks with the page, and the codes are already
sized for a phone camera at arm's length rather than for a comfortable surplus.
Print small enough and pages simply stop being recognised, with nothing to point
at as the cause.

For a print shop, the order is:

- **Paper size:** A5 (148 × 210 mm), portrait
- **Scaling:** none — 100%, actual size, borderless not required
- **Sides:** double-sided, **flip on the long edge** (this is the standard for a
  portrait book; short-edge flip would print every other page upside down)
- **Colour:** greyscale is fine — the ruling is grey and the codes are black
- **Paper:** ordinary 80–120 gsm. Avoid anything glossy: a shiny surface throws
  specular highlights straight back at a phone flash and can blow out a code.

10 pages double-sided is 5 sheets.

If you print it at home on A4, choose A5 booklet or 2-up **only if** your printer
driver does it without scaling — most scale slightly to fit A4's unprintable
margins. Cutting an A4 sheet down to A5 by hand after printing at 100% is safer.

## Checking it came out right

1. **Measure the page with a ruler.** It should be 148 × 210 mm. This is the
   only way to catch a scaled print — nothing downstream can, because a photo
   carries no sense of absolute size, so a 96% print of an A5 page and a real A5
   page an inch further from the lens are the same picture.
2. **Photograph a page** and run it back through:

   ```bash
   paperlog notebooks --add examples/printable/PAPER1.manifest.json
   paperlog scan my-photo.jpg -o pages/
   ```

   Four codes and a `fit` under about 0.6mm means the page was read cleanly.
   `fit` measures whether the four corners agree with each other about a flat
   page, so it catches a curled or creased sheet, a misread code, or a photo
   taken at too steep an angle — not scale.

You do not need to write in it first — the codes are what is being tested, and
a blank page tests them just as well.

## A note on the id

This notebook's id is `PAPER1`, fixed so the committed PDF and manifest always
agree. That is fine for a test run, but **generate your own id for a real
notebook** — two notebooks sharing an id get filed as one:

```bash
paperlog new-id
paperlog build -c examples/printable/notebook.yaml --notebook-id <your-id> --out mine.pdf
```
