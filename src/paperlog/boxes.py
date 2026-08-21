"""Find the boxes you drew around tool calls, and cut them out.

The writing surface is paper, so the markup has to be something a hand can make
without breaking flow. A drawn rectangle is the cheapest such gesture: no
counting brackets, no spelling a keyword, and no way to get it half right.

**Why this is computer vision and not a language model.** Cropping the box out
of the image *before* anything reads it is what keeps the surrounding prose out
of the request. A model that reads the whole page and then picks out the boxed
part has already seen the paragraph around it, and can quietly tailor the
request to fit -- which takes authorial control away from the writer. Cutting
the pixels is a guarantee rather than an instruction: what is outside the
rectangle cannot reach the model, because it is not in the image that is sent.

**How a drawn box is told from printed ruling.** Not by darkness -- flattening
pushes faint ruling to white in some places and leaves it grey in others, so
tone is unreliable. By *thickness*: printed ruling is a fraction of a
millimetre, a pen line is several times that, and an erosion sized between the
two removes one and leaves the other. Underlines and margin rules survive that
test too, so they are then rejected for not enclosing anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from .scancheck import load_backend

#: Ink is anything meaningfully darker than paper. Deliberately generous: the
#: thickness test below is what does the discriminating, not this.
INK_LEVEL = 200

#: Erosion that removes printed ruling and leaves pen. At 300dpi the ruling is
#: about 1.7px wide and a pen line 4-6px, so a 3x3 lands cleanly between them.
THIN_KERNEL = 3

#: Closes the wobble and the skips a real pen leaves.
JOIN_KERNEL = 9

#: A box has to be worth drawing. Below this it is likely a doodle or a heavily
#: ringed word; above it, someone has boxed the whole page.
MIN_PAGE_FRACTION = 0.02
MAX_PAGE_FRACTION = 0.75

#: How much of its own bounding rectangle the outline traces. A rectangle is
#: near 1.0; an L-shaped margin rule or a stray curve is far below.
MIN_RECTANGULARITY = 0.7

#: How much ink is allowed *inside*. A box is an outline around some writing;
#: a filled scribble or a QR code is not, and this is what separates them.
MAX_INSIDE_DENSITY = 0.25


@dataclass(frozen=True)
class Box:
    """A drawn rectangle on a flattened page, in page pixels."""

    x: int
    y: int
    width: int
    height: int
    #: Diagnostics, kept because when this misfires these are what say why.
    rectangularity: float = 0.0
    inside_density: float = 0.0

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def reading_order(self) -> Tuple[int, int]:
        """Down the page, then across -- how a person would meet them."""
        return (self.y, self.x)


def _furniture_mask(numpy, shape, geometry, dpi: int):
    """Blank out the corner codes, so a code can never read as a drawing.

    The density test already rejects them, but geometry is exact and free: the
    manifest says where the codes are, so there is no reason to reason about
    their pixels at all.
    """
    mask = numpy.ones(shape[:2], numpy.uint8)
    if geometry is None:
        return mask
    height, width = shape[:2]
    scale = dpi / 25.4
    for corner, (x_mm, y_mm, size_mm) in geometry.boxes.items():
        left = int(x_mm * scale)
        size = int(size_mm * scale)
        # Page y grows upward; image rows grow downward.
        top = int((geometry.height_mm - y_mm - size_mm) * scale)
        pad = int(2 * scale)  # a little slack for fit error
        mask[
            max(top - pad, 0) : min(top + size + pad, height),
            max(left - pad, 0) : min(left + size + pad, width),
        ] = 0
    return mask


def find_boxes(
    image,
    *,
    geometry=None,
    dpi: int = 300,
    min_fraction: float = MIN_PAGE_FRACTION,
    max_fraction: float = MAX_PAGE_FRACTION,
) -> List[Box]:
    """Find hand-drawn rectangles on a flattened page, in reading order."""
    cv2, numpy, _ = load_backend()
    gray = numpy.asarray(image)
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    ink = (gray < INK_LEVEL).astype(numpy.uint8)
    ink &= _furniture_mask(numpy, gray.shape, geometry, dpi)

    thick = cv2.erode(
        ink, cv2.getStructuringElement(cv2.MORPH_RECT, (THIN_KERNEL, THIN_KERNEL))
    )
    joined = cv2.dilate(
        thick, cv2.getStructuringElement(cv2.MORPH_RECT, (JOIN_KERNEL, JOIN_KERNEL))
    )

    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    page_area = gray.shape[0] * gray.shape[1]
    found: List[Box] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not (page_area * min_fraction <= area <= page_area * max_fraction):
            continue
        x, y, width, height = cv2.boundingRect(contour)
        rectangularity = area / float(width * height)
        inset_x, inset_y = width // 8, height // 8
        inside = thick[y + inset_y : y + height - inset_y, x + inset_x : x + width - inset_x]
        density = float(inside.mean()) if inside.size else 1.0
        if rectangularity < MIN_RECTANGULARITY or density > MAX_INSIDE_DENSITY:
            continue
        found.append(Box(x, y, width, height, rectangularity, density))

    found.sort(key=lambda box: box.reading_order)
    return found


def crop(image, box: Box):
    """Cut a box out of the page.

    Exactly the bounding rectangle, never padded outward. Padding would be the
    obvious kindness -- a few millimetres of margin so nothing is clipped -- and
    it is precisely wrong here: every pixel of padding is a chance for the line
    of prose above the box to appear in the request. The drawn line sits on the
    boundary, so cropping to it keeps the writing inside and everything else
    out.
    """
    _, numpy, _ = load_backend()
    gray = numpy.asarray(image)
    return gray[box.y : box.y + box.height, box.x : box.x + box.width]
