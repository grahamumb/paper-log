"""Sheet imposition for booklets.

A saddle-stitched booklet is printed two-up on sheets that are folded down the
middle and nested inside each other, so the page order on the press bears no
resemblance to the reading order. This module works out that order.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

Sheet = Tuple[Optional[int], Optional[int]]  # (left page, right page); None = blank


def padded_count(pages: int) -> int:
    """Round a page count up to a whole number of folded sheets (4 sides)."""
    return ((pages + 3) // 4) * 4


def booklet_sheets(pages: int) -> List[Sheet]:
    """Return sheet faces in print order for a saddle-stitched booklet.

    Each entry is one side of one physical sheet, as ``(left, right)`` page
    numbers. Print the list in order, double-sided, flipping on the short edge;
    stack, fold, staple through the spine.

    For 8 pages: ``[(8, 1), (2, 7), (6, 3), (4, 5)]``.
    """
    if pages < 1:
        return []
    total = padded_count(pages)

    def existing(number: int) -> Optional[int]:
        return number if number <= pages else None

    sheets: List[Sheet] = []
    for index in range(total // 4):
        first = 1 + 2 * index
        last = total - 2 * index
        # Front of the sheet: highest remaining page on the left, lowest on the
        # right. Back: the two pages that fall between them.
        sheets.append((existing(last), existing(first)))
        sheets.append((existing(first + 1), existing(last - 1)))
    return sheets
