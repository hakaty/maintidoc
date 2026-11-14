from llama_index.core.schema import Document as LlamaDocument

from app.ingestion.repeated_lines import detect_repeated_lines, remove_repeated_lines


def _documents(pages: list[str]) -> list[LlamaDocument]:
    return [
        LlamaDocument(text=text, metadata={"page": index + 1})
        for index, text in enumerate(pages)
    ]


def test_detects_header_repeated_on_every_page() -> None:
    documents = _documents(
        [
            "ACME Corp\nIntroduction du produit.",
            "ACME Corp\nCaracteristiques techniques.",
            "ACME Corp\nMaintenance preventive.",
        ]
    )

    repeated = detect_repeated_lines(documents)

    assert "ACME Corp" in repeated.values()


def test_detects_footer_with_varying_page_number() -> None:
    documents = _documents(
        [
            "Contenu page 1.\nPage 1 / 3",
            "Contenu page 2.\nPage 2 / 3",
            "Contenu page 3.\nPage 3 / 3",
        ]
    )

    repeated = detect_repeated_lines(documents)

    # digits are normalized to "#" for comparison, so "Page 1 / 3" and
    # "Page 2 / 3" count as the same repeated pattern
    assert any(value.startswith("Page ") for value in repeated.values())


def test_line_below_threshold_is_not_removed() -> None:
    documents = _documents(
        [
            "Rare line.\nUnique content A.",
            "Content B, no repeat here.",
            "Content C, also unique.",
            "Content D, still unique.",
        ]
    )

    repeated = detect_repeated_lines(documents, min_page_ratio=0.5)

    assert repeated == {}


def test_remove_repeated_lines_strips_only_matching_lines() -> None:
    documents = _documents(
        [
            "ACME Corp\nIntroduction du produit.",
            "ACME Corp\nCaracteristiques techniques.",
            "ACME Corp\nMaintenance preventive.",
        ]
    )
    repeated = detect_repeated_lines(documents)

    cleaned = remove_repeated_lines(documents, repeated)

    assert "ACME Corp" not in cleaned[0].text
    assert "Introduction du produit." in cleaned[0].text
    assert "ACME Corp" not in cleaned[1].text
    assert "Caracteristiques techniques." in cleaned[1].text


def test_remove_repeated_lines_does_not_mutate_input() -> None:
    documents = _documents(["ACME Corp\nA.", "ACME Corp\nB."])
    original_texts = [doc.text for doc in documents]
    repeated = detect_repeated_lines(documents)

    remove_repeated_lines(documents, repeated)

    assert [doc.text for doc in documents] == original_texts


def test_no_repeated_lines_returns_same_list() -> None:
    documents = _documents(["Unique A.", "Unique B."])

    result = remove_repeated_lines(documents, {})

    assert result is documents
