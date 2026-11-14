"""The find_tables() bbox cache in app.ingestion._captions is keyed by
(path, sha256, page) -- not path alone -- so a changed file reusing the same
path is never served another file's stale bboxes."""

import hashlib
from pathlib import Path

import pymupdf
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate
from scripts.make_sample_pdf import _plain_table

from app.ingestion._captions import get_table_bboxes


def _one_table_pdf(path: Path, header: list[str], rows: list[list[str]]) -> str:
    # find_tables() needs some other text content on the page to anchor its
    # detection -- a bare table with nothing else on the page is never found.
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(path), pagesize=A4)
    doc.build([Paragraph("Table", styles["Heading2"]), _plain_table(header, rows)])
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_same_path_different_content_is_not_served_stale_bboxes(tmp_path: Path) -> None:
    shared_path = tmp_path / "reused_name.pdf"

    sha_a = _one_table_pdf(shared_path, ["A", "B"], [["1", "2"], ["3", "4"]])
    with pymupdf.open(shared_path) as doc:
        bboxes_a = get_table_bboxes(str(shared_path), sha_a, 1, doc[0])

    # Overwrite the SAME path with a different table (3 columns, more rows):
    # a real scenario when re-ingesting an updated PDF kept at a stable path.
    sha_b = _one_table_pdf(
        shared_path, ["X", "Y", "Z"], [["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9"]]
    )
    assert sha_b != sha_a
    with pymupdf.open(shared_path) as doc:
        bboxes_b = get_table_bboxes(str(shared_path), sha_b, 1, doc[0])

    assert len(bboxes_a) == 1
    assert len(bboxes_b) == 1
    # The second table has more rows: its bbox must be taller, not identical
    # to the (stale, wrong) first table's bbox.
    assert bboxes_b[0].height > bboxes_a[0].height

    # Re-requesting the first (path, sha) pair still returns its own cached
    # value, unaffected by the second file having reused the same path.
    with pymupdf.open(shared_path) as doc:
        bboxes_a_again = get_table_bboxes(str(shared_path), sha_a, 1, doc[0])
    assert bboxes_a_again == bboxes_a
