"""Detect and strip header/footer lines repeated across most pages.

Manual titles, company names, "Page N / M" footers and revision codes repeat
on (almost) every page; they add no RAG value as text chunks and are kept
once in manifest.json instead.
"""

import re

from llama_index.core.schema import Document as LlamaDocument

DEFAULT_MIN_PAGE_RATIO = 0.5

_DIGIT_PATTERN = re.compile(r"\d+")


def normalize_line(line: str) -> str:
    """Collapse digits so "Page 3 / 48" and "Page 4 / 48" compare equal."""
    return _DIGIT_PATTERN.sub("#", line.strip())


def detect_repeated_lines(
    documents: list[LlamaDocument], min_page_ratio: float = DEFAULT_MIN_PAGE_RATIO
) -> dict[str, str]:
    """Normalized line pattern -> a representative (first-seen) raw example."""
    total_pages = len(documents)
    if total_pages == 0:
        return {}

    page_count: dict[str, int] = {}
    example: dict[str, str] = {}
    for document in documents:
        seen_on_page: set[str] = set()
        for raw_line in document.text.split("\n"):
            stripped = raw_line.strip()
            if not stripped:
                continue
            normalized = normalize_line(stripped)
            seen_on_page.add(normalized)
            example.setdefault(normalized, stripped)
        for normalized in seen_on_page:
            page_count[normalized] = page_count.get(normalized, 0) + 1

    return {
        normalized: example[normalized]
        for normalized, count in page_count.items()
        if count / total_pages >= min_page_ratio
    }


def remove_repeated_lines(
    documents: list[LlamaDocument], repeated_patterns: dict[str, str]
) -> list[LlamaDocument]:
    """New Document list with repeated lines dropped; does not mutate `documents`."""
    if not repeated_patterns:
        return documents

    cleaned: list[LlamaDocument] = []
    for document in documents:
        kept_lines = [
            raw_line
            for raw_line in document.text.split("\n")
            if normalize_line(raw_line) not in repeated_patterns
        ]
        cleaned.append(
            LlamaDocument(text="\n".join(kept_lines).strip(), metadata=dict(document.metadata))
        )
    return cleaned
