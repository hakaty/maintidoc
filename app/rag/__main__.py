"""CLI: python -m app.rag ask "question" [--image path] [--show-context]
[--provider ollama|openrouter] [--k N] [--mode vector|hybrid]
"""

import argparse
import sys
from pathlib import Path

from app.config import Settings, get_settings
from app.rag.retriever import retrieve
from app.rag.service import RAGAnswer, answer


def _print_context(question: str, k: int | None, mode: str, settings: Settings) -> None:
    result = retrieve(question, k=k, mode=mode, settings=settings)
    print(f"--- context (best_score={result.best_score:.4f}, refused={result.refused}) ---")
    for node in result.nodes:
        print(f"[p. {node.page}] ({node.content_type}, score={node.score:.4f}) {node.text[:200]}")
    print("--- end context ---\n")


def _print_answer(result: RAGAnswer) -> None:
    print(result.answer)
    if result.safety_notice:
        print()
        print(result.safety_notice)
    if result.citations:
        print("\nCitations:")
        for citation in result.citations:
            print(f"  [p. {citation.page}] {citation.content_type} node_id={citation.node_id}")
    if result.images:
        print("\nImages citées:")
        for image_path in result.images:
            print(f"  {image_path}")
    print(
        f"\nrefused={result.refused} citations_corrected={result.citations_corrected} "
        f"error={result.error}"
    )
    print(f"model={result.model} provider={result.provider} prompt_version={result.prompt_version}")
    print(f"timings={result.timings}")


def main(argv: list[str] | None = None) -> int:
    # Answers may contain characters outside the Windows console's cp1252
    # codepage: degrade the display instead of crashing (see app/indexing/__main__.py).
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(prog="python -m app.rag")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask_parser = subparsers.add_parser("ask", help="Ask a question answered from the manual")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--image", type=Path, default=None)
    ask_parser.add_argument("--show-context", action="store_true")
    ask_parser.add_argument("--provider", choices=["ollama", "openrouter"], default=None)
    ask_parser.add_argument("--k", type=int, default=None)
    ask_parser.add_argument("--mode", choices=["vector", "hybrid"], default="hybrid")

    args = parser.parse_args(argv)
    if args.command != "ask":
        return 2

    settings = get_settings()
    if args.provider and args.provider != settings.llm_provider:
        settings = Settings(**{**settings.model_dump(), "llm_provider": args.provider})

    if args.show_context:
        _print_context(args.question, args.k, args.mode, settings)

    result = answer(
        args.question, image_path=args.image, k=args.k, mode=args.mode, settings=settings
    )
    _print_answer(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
