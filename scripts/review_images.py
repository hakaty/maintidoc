"""Manual review helper for app.images enrichment output.

python scripts/review_images.py <processed_dir>
- writes <processed_dir>/images_review.html (self-contained, git-ignored):
  thumbnail, page, caption, CLIP label + score, description, visible_text,
  status -- for human checking.
- runs 3 English CLIP sanity queries against clip_embeddings.npy and prints
  the top-3 matching image ids + scores.
"""

import base64
import html
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from app.images.clip_encoder import ClipEncoder

THUMBNAIL_MAX_PX = 200

SANITY_QUERIES = [
    "cross-section of a centrifugal pump",
    "electrical wiring diagram",
    "safety warning sign",
]


def _load_image_nodes(processed_dir: Path) -> list[dict]:
    nodes_path = processed_dir / "nodes.jsonl"
    nodes = []
    for line in nodes_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        if data.get("metadata", {}).get("content_type") == "image":
            nodes.append(data)
    return nodes


def _thumbnail_data_uri(image_path: Path, max_size: int = THUMBNAIL_MAX_PX) -> str:
    with PILImage.open(image_path) as img:
        img = img.convert("RGB")
        img.thumbnail((max_size, max_size))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=80)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _row_html(data: dict) -> str:
    meta = data["metadata"]
    thumb = _thumbnail_data_uri(Path(data["image_path"]))
    score = meta.get("clip_label_score")
    score_str = f"{score:.3f}" if isinstance(score, int | float) else ""
    return (
        "<tr>"
        f'<td><img src="{thumb}" alt=""></td>'
        f"<td>{html.escape(meta.get('image_id', ''))}</td>"
        f"<td>{meta.get('page', '')}</td>"
        f"<td>{html.escape(meta.get('caption') or '')}</td>"
        f"<td>{html.escape(meta.get('clip_label') or '')}<br>{score_str}</td>"
        f"<td>{html.escape(meta.get('description') or '')}</td>"
        f"<td>{html.escape(meta.get('visible_text') or '')}</td>"
        f"<td>{html.escape(meta.get('description_status') or '')}</td>"
        "</tr>"
    )


def build_html(processed_dir: Path) -> str:
    image_nodes = _load_image_nodes(processed_dir)
    rows = "\n".join(_row_html(data) for data in image_nodes)
    return f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Revue des images extraites</title>
<style>
body {{ font-family: sans-serif; margin: 1.5rem; }}
table {{ border-collapse: collapse; width: 100%; }}
td, th {{ border: 1px solid #ccc; padding: 6px; vertical-align: top; font-size: 13px; }}
td, th {{ text-align: left; }}
img {{ max-width: {THUMBNAIL_MAX_PX}px; max-height: {THUMBNAIL_MAX_PX}px; }}
</style>
</head>
<body>
<h1>Revue des images extraites ({len(image_nodes)} image(s))</h1>
<table>
<tr>
<th>Image</th><th>ID</th><th>Page</th><th>Legende</th><th>Label CLIP</th>
<th>Description</th><th>Texte visible</th><th>Statut</th>
</tr>
{rows}
</table>
</body>
</html>
"""


def run_sanity_queries(
    processed_dir: Path, encoder: ClipEncoder | None = None
) -> dict[str, list[tuple[str, float]]]:
    index_path = processed_dir / "clip_index.json"
    npy_path = processed_dir / "clip_embeddings.npy"
    if not index_path.is_file() or not npy_path.is_file():
        print(
            "No clip_embeddings.npy/clip_index.json found; run "
            "`python -m app.images enrich <processed_dir>` first.",
            file=sys.stderr,
        )
        return {}

    index = json.loads(index_path.read_text(encoding="utf-8"))
    vectors = np.load(npy_path)
    image_ids = index["image_ids"]
    if not image_ids:
        return {}

    encoder = encoder or ClipEncoder()
    query_embeddings = encoder.encode_texts(SANITY_QUERIES)

    results = {}
    similarities = query_embeddings @ vectors.T
    for query, row in zip(SANITY_QUERIES, similarities, strict=True):
        top3 = np.argsort(row)[::-1][:3]
        results[query] = [(image_ids[i], float(row[i])) for i in top3]
    return results


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/review_images.py <processed_dir>", file=sys.stderr)
        return 2
    processed_dir = Path(sys.argv[1])

    html_content = build_html(processed_dir)
    output_path = processed_dir / "images_review.html"
    output_path.write_text(html_content, encoding="utf-8")
    print(f"Generated {output_path}")

    print("\nCLIP sanity queries:")
    for query, results in run_sanity_queries(processed_dir).items():
        print(f"  '{query}':")
        for image_id, score in results:
            print(f"    {image_id}: {score:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
