"""CLI: python -m app.indexing build|search|search-image|search-image-text.

No LLM involved anywhere in this module: embeddings and vector search only.
"""

import argparse
import sys
from pathlib import Path

from app.config import get_settings
from app.indexing.stores import (
    BuildResult,
    ImageSearchResult,
    SearchResult,
    build_index,
    search_image,
    search_image_text,
    search_text,
)


def _print_build_summary(result: BuildResult) -> None:
    print(f"pdf_sha256: {result.pdf_sha256}")
    print(
        f"text index: {result.text_nodes_added} added "
        f"(of {result.text_nodes_total} built), {result.text_count_in_db} in DB"
    )
    print(
        f"image index: {result.image_nodes_added} added "
        f"(of {result.image_nodes_total} built), {result.image_count_in_db} in DB"
    )


def _print_search_results(results: list[SearchResult]) -> None:
    if not results:
        print("(no results)")
        return
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. page={result.page} type={result.content_type} "
            f"score={result.score:.4f} | {result.snippet}"
        )


def _print_image_search_results(results: list[ImageSearchResult]) -> None:
    if not results:
        print("(no results)")
        return
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. image_id={result.image_id} page={result.page} "
            f"score={result.score:.4f} caption={result.caption!r}"
        )


def main(argv: list[str] | None = None) -> int:
    # Document text/captions are untrusted PDF content and may contain characters
    # outside the Windows console's cp1252 codepage (e.g. U+2264 "<="): without
    # this, printing them crashes with UnicodeEncodeError instead of just
    # degrading the display.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(prog="python -m app.indexing")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser(
        "build", help="Embed and insert a processed document's nodes into PGVector"
    )
    build_parser.add_argument("processed_dir", type=Path)
    build_parser.add_argument("--force", action="store_true")

    search_parser = subparsers.add_parser("search", help="Search the text index")
    search_parser.add_argument("query")
    search_parser.add_argument("--k", type=int, default=5)
    search_parser.add_argument(
        "--type", dest="content_type", choices=["text", "table", "image_description"], default=None
    )
    search_parser.add_argument("--mode", choices=["vector", "hybrid"], default="vector")

    search_image_parser = subparsers.add_parser(
        "search-image", help="CLIP image-to-image search over the image index"
    )
    search_image_parser.add_argument("image_path", type=Path)
    search_image_parser.add_argument("--k", type=int, default=5)

    search_image_text_parser = subparsers.add_parser(
        "search-image-text", help="CLIP text-to-image search (English query only)"
    )
    search_image_text_parser.add_argument("query")
    search_image_text_parser.add_argument("--k", type=int, default=5)

    args = parser.parse_args(argv)
    settings = get_settings()

    if args.command == "build":
        result = build_index(args.processed_dir, force=args.force, settings=settings)
        _print_build_summary(result)
    elif args.command == "search":
        results = search_text(
            args.query, k=args.k, content_type=args.content_type, mode=args.mode, settings=settings
        )
        _print_search_results(results)
    elif args.command == "search-image":
        results = search_image(args.image_path, k=args.k, settings=settings)
        _print_image_search_results(results)
    elif args.command == "search-image-text":
        results = search_image_text(args.query, k=args.k, settings=settings)
        _print_image_search_results(results)
    else:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
