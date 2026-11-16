"""Diagnosis script (M7.4 step 1): print every stored row/chunk for page 5,
the retrieval results for the pictogram question, and the exact context
block sent to the LLM. Evidence only -- no changes made here.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llama_index.core.vector_stores.types import (  # noqa: E402
    FilterCondition,
    MetadataFilter,
    MetadataFilters,
)

from app.config import get_settings  # noqa: E402
from app.indexing.stores import get_text_store  # noqa: E402
from app.rag.retriever import retrieve  # noqa: E402
from app.rag.service import build_prompt  # noqa: E402

QUESTION = (
    "Quel pictogramme de sécurité, parmi les symboles complémentaires, signale "
    "un risque d'atmosphère explosive ?"
)


def main() -> int:
    settings = get_settings()
    store = get_text_store(settings)

    print("=" * 80)
    print("(a) ALL STORED NODES FOR PAGE 5")
    print("=" * 80)
    filters = MetadataFilters(
        filters=[MetadataFilter(key="page", value=5)], condition=FilterCondition.AND
    )
    nodes = store.get_nodes(filters=filters)
    print(f"n={len(nodes)} nodes on page 5\n")
    for n in nodes:
        meta = n.metadata
        print(f"--- node_id={n.node_id} content_type={meta.get('content_type')} ---")
        print(f"metadata: {meta}")
        print("content:")
        print(n.get_content())
        print()

    print("=" * 80)
    print("(b) RETRIEVAL RESULTS FOR THE PICTOGRAM QUESTION")
    print("=" * 80)
    result = retrieve(QUESTION, settings=settings)
    print(f"best_score={result.best_score} refused={result.refused} n_kept={len(result.nodes)}")
    for n in result.nodes:
        print(
            f"- id={n.node_id[:12]} type={n.content_type} page={n.page} score={n.score:.4f}"
        )
        print(f"  text[:200]={n.text[:200]!r}")

    print("\n" + "=" * 80)
    print("(c) EXACT CONTEXT BLOCK SENT TO THE LLM")
    print("=" * 80)
    prompt = build_prompt(QUESTION, result.nodes)
    print(prompt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
