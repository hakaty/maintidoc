import re

import pymupdf
import pytest

from app.ingestion._captions import TextBlock, find_caption
from app.ingestion.repeated_lines import normalize_line


@pytest.fixture
def keyword_pattern() -> re.Pattern[str]:
    return re.compile(r"figure|fig\.", re.IGNORECASE)


def _block(y: float, text: str) -> TextBlock:
    return TextBlock(bbox=pymupdf.Rect(0, y, 100, y + 10), text=text)


def test_repeated_header_block_is_never_returned(keyword_pattern) -> None:
    header = "fr - Traduction de la notice originale"
    real_caption = "Figure 1 : schema de principe"
    blocks = [_block(0, header), _block(50, real_caption)]
    target = pymupdf.Rect(5, 5, 50, 45)  # closest to the header block
    repeated = {normalize_line(header): header}

    caption = find_caption(target, blocks, keyword_pattern, repeated_patterns=repeated)

    assert caption == real_caption


def test_falls_back_to_none_when_every_block_is_repeated(keyword_pattern) -> None:
    header = "fr - Traduction de la notice originale"
    footer = "Page 3"
    blocks = [_block(0, header), _block(90, footer)]
    target = pymupdf.Rect(0, 40, 100, 50)
    repeated = {normalize_line(header): header, normalize_line(footer): footer}

    caption = find_caption(target, blocks, keyword_pattern, repeated_patterns=repeated)

    assert caption is None


def test_multiline_block_entirely_repeated_lines_is_excluded(keyword_pattern) -> None:
    """A footer block can combine a page number and a running title on two
    lines ("2 \nSéries CEA, CIE, CA - Instructions..."); if every line in the
    block is itself a repeated pattern, the whole block must be excluded."""
    footer_block = "2\nSeries CEA - Instructions d'installation"
    real_caption = "Figure 2 : vue eclatee"
    blocks = [_block(0, footer_block), _block(50, real_caption)]
    target = pymupdf.Rect(5, 5, 50, 45)
    repeated = {
        normalize_line("2"): "2",
        normalize_line("Series CEA - Instructions d'installation"): (
            "Series CEA - Instructions d'installation"
        ),
    }

    caption = find_caption(target, blocks, keyword_pattern, repeated_patterns=repeated)

    assert caption == real_caption


def test_block_with_some_repeated_and_some_real_lines_is_kept(keyword_pattern) -> None:
    """A block is only excluded if ALL of its lines are repeated -- one real
    line alongside a repeated one still makes it a valid candidate."""
    header = "fr - Traduction de la notice originale"
    mixed_block = f"{header}\nFigure 3 : detail du joint"
    blocks = [_block(0, mixed_block)]
    target = pymupdf.Rect(0, 0, 100, 10)
    repeated = {normalize_line(header): header}

    caption = find_caption(target, blocks, keyword_pattern, repeated_patterns=repeated)

    assert caption == mixed_block


def test_no_repeated_patterns_behaves_as_before(keyword_pattern) -> None:
    blocks = [_block(0, "Figure 1 : test")]
    target = pymupdf.Rect(0, 0, 100, 10)

    caption = find_caption(target, blocks, keyword_pattern)

    assert caption == "Figure 1 : test"
