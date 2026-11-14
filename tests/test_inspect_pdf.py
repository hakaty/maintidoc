from pathlib import Path

import pytest
from scripts.inspect_pdf import NotAPdfError, inspect_pdf


def test_inspect_sample_pdf(sample_pdf: Path) -> None:
    report = inspect_pdf(sample_pdf)

    assert report.page_count == 8
    assert report.scanned_pages == []
    assert all(page.char_count > 0 for page in report.pages)

    # >= 3 kept-size images (3 diagrams + the logo repeated on every page) and
    # >= 2 tables: the fixture must clear the "suitable" thresholds.
    assert report.total_images >= 3
    assert report.total_tables == 2
    assert report.is_suitable is True


def test_inspect_rejects_non_pdf(tmp_path: Path) -> None:
    fake_pdf = tmp_path / "fake.pdf"
    fake_pdf.write_bytes(b"this is not a pdf")

    with pytest.raises(NotAPdfError):
        inspect_pdf(fake_pdf)
