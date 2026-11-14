"""Direct (no-browser) timing profile of app.rag.retriever.retrieve_by_image,
step by step -- model load, image embedding, DB query, rendering data prep.

Usage: python -m scripts._profile_image_search [image_path]
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llama_index.core.vector_stores.types import (  # noqa: E402
    FilterCondition,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
)

from app.config import get_settings  # noqa: E402
from app.images.clip_encoder import ClipEncoder  # noqa: E402
from app.indexing.stores import get_image_store, get_text_store  # noqa: E402
from app.rag import retriever  # noqa: E402

DEFAULT_IMAGE = Path("data/media/5c1a4a3e34a4/5c1a4a3e_p1_i0.png")


def _step(label: str, fn):
    t0 = time.monotonic()
    result = fn()
    elapsed_ms = (time.monotonic() - t0) * 1000
    print(f"{label:<28} {elapsed_ms:9.1f} ms")
    return result, elapsed_ms


def run(image_path: Path, label: str) -> None:
    print(f"\n=== {label} ({image_path.name}) ===")
    settings = get_settings()
    total_ms = 0.0

    def _load():
        encoder = ClipEncoder(settings)
        encoder._ensure_loaded()  # force load now, not lazily inside encode_images
        return encoder

    encoder, ms = _step("1. CLIP model load", _load)
    total_ms += ms

    def _embed():
        return encoder.encode_images([image_path])[0].tolist()

    vector, ms = _step("2. Image embedding", _embed)
    total_ms += ms

    def _query_image_store():
        image_store = get_image_store(settings)
        vs_query = VectorStoreQuery(query_embedding=vector, similarity_top_k=3)
        return image_store.query(vs_query)

    result, ms = _step("3. Image-store DB query", _query_image_store)
    total_ms += ms

    def _query_text_store():
        text_store = get_text_store(settings)
        matches = []
        for node, score in zip(
            result.nodes or [], result.similarities or [0.0] * len(result.nodes or []), strict=True
        ):
            image_id = node.metadata.get("image_id")
            if not image_id:
                continue
            filters = MetadataFilters(
                filters=[
                    MetadataFilter(key="content_type", value="image_description"),
                    MetadataFilter(key="image_id", value=image_id),
                ],
                condition=FilterCondition.AND,
            )
            found = text_store.get_nodes(filters=filters)
            if found:
                matches.append((found[0], score))
        return matches

    matches, ms = _step("4. Text-store lookup (per match)", _query_text_store)
    total_ms += ms

    def _render_prep():
        return [
            retriever._to_retrieved_node(node, score=float(score)) for node, score in matches
        ]

    nodes, ms = _step("5. Rendering data prep", _render_prep)
    total_ms += ms

    print(f"{'TOTAL':<28} {total_ms:9.1f} ms")
    for n in nodes:
        print(f"  -> page={n.page} score={n.score:.4f}")

    encoder.unload()


def run_real_retrieve_by_image(image_path: Path, label: str) -> None:
    """Calls the actual app.rag.retriever.retrieve_by_image() end to end
    (the real UI/service code path, with the encoder cache fix applied)."""
    settings = get_settings()
    t0 = time.monotonic()
    matches = retriever.retrieve_by_image(image_path, k=3, settings=settings)
    elapsed_ms = (time.monotonic() - t0) * 1000
    print(f"\n=== {label} ({image_path.name}) ===")
    print(f"{'TOTAL (retrieve_by_image)':<28} {elapsed_ms:9.1f} ms")
    for n in matches:
        print(f"  -> page={n.page} score={n.score:.4f}")


def main() -> int:
    positional = [a for a in sys.argv[1:] if not a.startswith("--")]
    image_path = Path(positional[0]) if positional else DEFAULT_IMAGE
    if not image_path.is_file():
        print(f"Image not found: {image_path}")
        return 1

    if "--after-fix" in sys.argv:
        run_real_retrieve_by_image(image_path, "Call 1 (cold, cache fix applied)")
        run_real_retrieve_by_image(image_path, "Call 2 (warm, cache fix applied)")
        run_real_retrieve_by_image(image_path, "Call 3 (warm, cache fix applied)")
        return 0

    run(image_path, "Run 1 (cold -- fresh ClipEncoder per call, before fix)")
    run(image_path, "Run 2 (same process, fresh ClipEncoder again -- repeat search, before fix)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
