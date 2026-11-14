"""CLI: python -m app.ingestion extract <pdf_path> [--force]"""

import argparse
import sys
from pathlib import Path

from app.config import get_settings
from app.ingestion.pipeline import CorruptedPdfError, ExtractionResult, run_extraction


def _print_summary(result: ExtractionResult) -> None:
    manifest = result.manifest
    counts = manifest["counts"]

    print(f"PDF: {manifest['source_file']}")
    print(f"sha256: {manifest['pdf_sha256'][:12]}...")
    print(f"pages: {manifest['total_pages']}")
    print(f"text chunks: {counts['text_chunks']}")
    print(f"tables: {counts['tables']}")
    print(f"images kept: {counts['images_kept']}")

    skipped_by_reason: dict[str, int] = {}
    for skipped in manifest["skipped_images"]:
        skipped_by_reason[skipped["reason"]] = skipped_by_reason.get(skipped["reason"], 0) + 1
    if skipped_by_reason:
        print("images skipped:")
        for reason, count in skipped_by_reason.items():
            print(f"  {reason}: {count}")

    if manifest["scanned_pages"]:
        print(f"scanned pages (no native text): {manifest['scanned_pages']}")

    print(f"status: {'unchanged' if result.unchanged else 'extracted'}")
    print(f"output: {result.processed_dir}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.ingestion")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser(
        "extract", help="Extract text/tables/images from a PDF"
    )
    extract_parser.add_argument("pdf_path", type=Path)
    extract_parser.add_argument("--force", action="store_true")

    args = parser.parse_args(argv)
    if args.command != "extract":
        return 2

    settings = get_settings()
    try:
        result = run_extraction(
            args.pdf_path,
            processed_root=Path("data/processed"),
            media_dir=settings.media_dir,
            force=args.force,
        )
    except CorruptedPdfError as exc:
        print(f"Extraction failed: {exc}", file=sys.stderr)
        return 1

    _print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
