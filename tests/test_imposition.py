import pytest

from paperlog.imposition import booklet_sheets, padded_count


def test_padded_count_rounds_up_to_whole_sheets():
    assert [padded_count(n) for n in (1, 4, 5, 8, 30)] == [4, 4, 8, 8, 32]


def test_eight_page_booklet_order():
    assert booklet_sheets(8) == [(8, 1), (2, 7), (6, 3), (4, 5)]


def test_four_page_booklet_order():
    assert booklet_sheets(4) == [(4, 1), (2, 3)]


def test_short_run_leaves_blanks_rather_than_inventing_pages():
    # Five pages needs two sheets; the three surplus slots stay empty.
    assert booklet_sheets(5) == [(None, 1), (2, None), (None, 3), (4, 5)]


@pytest.mark.parametrize("pages", [4, 8, 12, 16, 32, 64])
def test_every_page_is_placed_exactly_once(pages):
    placed = [number for sheet in booklet_sheets(pages) for number in sheet if number]
    assert sorted(placed) == list(range(1, pages + 1))


@pytest.mark.parametrize("pages", [4, 8, 12, 16, 32])
def test_reading_order_survives_folding(pages):
    """Fold the sheets and walk the stack: pages must come out 1, 2, 3, ...

    Sheet ``i`` sits at nesting depth ``i``; its front contributes the left
    then right page and its back the same, and reading a folded, nested stack
    means walking the outermost sheet's front-right, then its back-left...
    which is exactly what this reconstruction does.
    """
    sheets = booklet_sheets(pages)
    order = []
    count = len(sheets) // 2
    # Front half of the reading order: right-hand side of each front face,
    # left-hand side of each back face, working inwards.
    for index in range(count):
        front_left, front_right = sheets[2 * index]
        back_left, back_right = sheets[2 * index + 1]
        order.append(front_right)
        order.append(back_left)
    # Second half: work back outwards along the other side of the fold.
    for index in reversed(range(count)):
        front_left, front_right = sheets[2 * index]
        back_left, back_right = sheets[2 * index + 1]
        order.append(back_right)
        order.append(front_left)
    assert order == list(range(1, pages + 1))


def test_no_pages_no_sheets():
    assert booklet_sheets(0) == []
