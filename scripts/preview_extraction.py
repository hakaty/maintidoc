"""Manual-check helper: preview sample nodes from an extraction's nodes.jsonl."""

import sys
from pathlib import Path

from llama_index.core.schema import BaseNode

from app.ingestion.pipeline import node_from_line


def load_nodes(processed_dir: Path) -> list[BaseNode]:
    nodes_path = processed_dir / "nodes.jsonl"
    with nodes_path.open("r", encoding="utf-8") as fh:
        return [node_from_line(line) for line in fh if line.strip()]


def _truncate_markdown_rows(markdown: str, max_rows: int = 5) -> str:
    lines = markdown.splitlines()
    return "\n".join(lines[: max_rows + 2])  # header + separator + up to max_rows


def preview(processed_dir: Path) -> None:
    nodes = load_nodes(processed_dir)
    text_chunks = [node for node in nodes if node.metadata.get("content_type") == "text"]
    tables = [node for node in nodes if node.metadata.get("content_type") == "table"]
    images = [node for node in nodes if node.metadata.get("content_type") == "image"]

    print(f"=== {len(text_chunks)} text chunk(s), showing up to 3 ===")
    for node in text_chunks[:3]:
        print(f"-- page {node.metadata.get('page')} chunk {node.metadata.get('chunk_index')} --")
        print(node.text[:300])
        print()

    print(f"=== {len(tables)} table(s) ===")
    for node in tables:
        print(f"-- {node.metadata.get('table_id')} (page {node.metadata.get('page')}) --")
        print(f"caption: {node.metadata.get('caption')}")
        print(_truncate_markdown_rows(node.text))
        print()

    print(f"=== {len(images)} image(s) ===")
    for node in images:
        print(
            f"- {node.metadata.get('image_id')} (page {node.metadata.get('page')}): "
            f"caption={node.metadata.get('caption')!r} path={node.image_path}"
        )


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/preview_extraction.py <processed_dir>", file=sys.stderr)
        return 2
    preview(Path(sys.argv[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
