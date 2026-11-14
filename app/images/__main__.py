"""CLI: python -m app.images enrich <processed_dir> [--no-clip] [--no-vlm] [--force] [--limit N]"""

import argparse
from pathlib import Path

from app.config import get_settings
from app.images.pipeline import EnrichmentResult, run_enrichment


def _print_summary(result: EnrichmentResult) -> None:
    enrichment = result.manifest["images_enrichment"]
    print(f"device: {enrichment['device']} ({enrichment['gpu_name'] or 'n/a'})")
    print(
        f"clip model: {enrichment['clip_model']}/{enrichment['clip_pretrained']} "
        f"(dim={enrichment['clip_dim']})"
    )
    print(f"vlm model: {enrichment['vlm_model']} (prompt {enrichment['prompt_version']})")
    print(f"images: {enrichment['images_total']}")
    if enrichment["clip_label_distribution"]:
        print("clip label distribution:")
        for label, count in sorted(enrichment["clip_label_distribution"].items()):
            print(f"  {label}: {count}")
    if enrichment["description_status_counts"]:
        print("description status:", enrichment["description_status_counts"])
        print(
            f"vlm time: {enrichment['total_vlm_seconds']}s total, "
            f"{enrichment['avg_vlm_seconds']}s/image avg"
        )
    print(f"output: {result.processed_dir}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.images")
    subparsers = parser.add_subparsers(dest="command", required=True)

    enrich_parser = subparsers.add_parser(
        "enrich", help="Compute CLIP embeddings/labels and VLM descriptions for extracted images"
    )
    enrich_parser.add_argument("processed_dir", type=Path)
    enrich_parser.add_argument("--no-clip", action="store_true")
    enrich_parser.add_argument("--no-vlm", action="store_true")
    enrich_parser.add_argument("--force", action="store_true")
    enrich_parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args(argv)
    if args.command != "enrich":
        return 2

    get_settings()
    result = run_enrichment(
        args.processed_dir,
        use_clip=not args.no_clip,
        use_vlm=not args.no_vlm,
        force=args.force,
        limit=args.limit,
    )
    _print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
