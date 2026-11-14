"""Inspect a PDF (PyMuPDF only) and report whether it suits Partie I's RAG use case.

Suitability requires native text (no scanned pages), >= 3 sizeable images and
>= 2 tables across the whole document.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import pymupdf

PDF_MAGIC = b"%PDF"
MIN_IMAGE_SIDE_PX = 50
MIN_IMAGES = 3
MIN_TABLES = 2


class NotAPdfError(Exception):
    pass


@dataclass
class PageStats:
    page_number: int
    char_count: int
    image_count: int
    table_count: int

    @property
    def has_text(self) -> bool:
        return self.char_count > 0


@dataclass
class InspectionReport:
    path: Path
    pages: list[PageStats]

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def total_images(self) -> int:
        return sum(page.image_count for page in self.pages)

    @property
    def total_tables(self) -> int:
        return sum(page.table_count for page in self.pages)

    @property
    def scanned_pages(self) -> list[int]:
        return [page.page_number for page in self.pages if not page.has_text]

    @property
    def is_suitable(self) -> bool:
        return (
            not self.scanned_pages
            and self.total_images >= MIN_IMAGES
            and self.total_tables >= MIN_TABLES
        )


def check_pdf_magic(path: Path) -> None:
    with path.open("rb") as fh:
        header = fh.read(len(PDF_MAGIC))
    if header != PDF_MAGIC:
        raise NotAPdfError(f"{path}: not a PDF (bad magic bytes)")


def _count_large_images(doc: pymupdf.Document, page: pymupdf.Page) -> int:
    count = 0
    seen_xrefs: set[int] = set()
    for image in page.get_images(full=True):
        xref = image[0]
        if xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)
        info = doc.extract_image(xref)
        if info["width"] >= MIN_IMAGE_SIDE_PX and info["height"] >= MIN_IMAGE_SIDE_PX:
            count += 1
    return count


def inspect_pdf(path: Path) -> InspectionReport:
    check_pdf_magic(path)
    pages: list[PageStats] = []
    with pymupdf.open(path) as doc:
        for index, page in enumerate(doc):
            text = page.get_text().strip()
            pages.append(
                PageStats(
                    page_number=index + 1,
                    char_count=len(text),
                    image_count=_count_large_images(doc, page),
                    table_count=len(page.find_tables().tables),
                )
            )
    return InspectionReport(path=path, pages=pages)


def print_report(report: InspectionReport) -> None:
    print(f"PDF: {report.path}")
    print(f"Pages: {report.page_count}")
    for page in report.pages:
        print(
            f"  page {page.page_number}: {page.char_count} chars, "
            f"{page.image_count} image(s), {page.table_count} table(s)"
        )
    print(f"Total images (>= {MIN_IMAGE_SIDE_PX}px): {report.total_images}")
    print(f"Total tables: {report.total_tables}")
    if report.scanned_pages:
        print(f"Scanned pages (no native text): {report.scanned_pages}")
    print(f"Verdict: {'suitable' if report.is_suitable else 'not suitable'}")


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/inspect_pdf.py <path>", file=sys.stderr)
        return 2

    path = Path(sys.argv[1])
    try:
        report = inspect_pdf(path)
    except (NotAPdfError, FileNotFoundError) as exc:
        print(f"Cannot inspect PDF: {exc}", file=sys.stderr)
        return 1

    print_report(report)
    return 0 if report.is_suitable else 1


if __name__ == "__main__":
    raise SystemExit(main())
