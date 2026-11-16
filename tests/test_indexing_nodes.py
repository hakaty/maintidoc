from app.indexing.nodes import build_image_index_nodes, build_text_index_nodes, text_node_id

PDF_SHA = "deadbeef" * 8
SOURCE_FILE = "data/pdf/document.pdf"


def _text_raw(chunk_index: int = 0, page: int = 1) -> dict:
    return {
        "id_": f"abcd1234_p{page}_c{chunk_index}",
        "text": "Le pompe doit etre installee verticalement.",
        "metadata": {
            "content_type": "text",
            "page": page,
            "chunk_index": chunk_index,
            "pdf_sha256": PDF_SHA,
            "source_file": SOURCE_FILE,
            "related_image_ids": ["abcd1234_p1_i0"],
            "related_table_ids": [],
        },
    }


def _table_raw(page: int = 2, caption: str | None = "Sommaire") -> dict:
    return {
        "id_": "abcd1234_p2_t0",
        "text": "|a|b|\n|---|---|\n|1|2|",
        "metadata": {
            "content_type": "table",
            "table_id": "abcd1234_p2_t0",
            "page": page,
            "row_count": 2,
            "col_count": 2,
            "caption": caption,
            "pdf_sha256": PDF_SHA,
            "source_file": SOURCE_FILE,
        },
    }


def _image_raw(
    page: int = 1,
    caption: str | None = "Figure 1",
    description: str | None = "Une pompe centrifuge vue de face.",
    visible_text: str | None = "",
    kind_fr: str | None = "photo",
    safety_relevant: bool = False,
    description_status: str = "ok",
) -> dict:
    return {
        "image_path": "data/media/abcd1234/abcd1234_p1_i0.png",
        "text": caption or "",
        "metadata": {
            "content_type": "image",
            "image_id": "abcd1234_p1_i0",
            "sha256": "feedface",
            "page": page,
            "caption": caption,
            "kind_fr": kind_fr,
            "clip_label": "a photo of a pump",
            "description": description,
            "visible_text": visible_text,
            "safety_relevant": safety_relevant,
            "description_status": description_status,
            "pdf_sha256": PDF_SHA,
            "source_file": SOURCE_FILE,
        },
    }


def test_text_chunk_becomes_text_node_with_common_metadata() -> None:
    nodes = build_text_index_nodes([_text_raw()], PDF_SHA, SOURCE_FILE)

    assert len(nodes) == 1
    node = nodes[0]
    assert node.metadata["content_type"] == "text"
    assert node.text == "Le pompe doit etre installee verticalement."
    assert node.metadata["page"] == 1
    assert node.metadata["chunk_index"] == 0
    assert node.metadata["related_image_ids"] == ["abcd1234_p1_i0"]
    assert node.metadata["pdf_sha256"] == PDF_SHA
    assert node.metadata["source_file"] == SOURCE_FILE
    assert node.id_ == text_node_id(PDF_SHA, "text", "abcd1234_p1_c0")


def test_table_node_text_is_prefixed_and_keeps_markdown_intact() -> None:
    raw = _table_raw(page=2, caption="Sommaire")
    nodes = build_text_index_nodes([raw], PDF_SHA, SOURCE_FILE)

    # One whole-table node + one table_row node (section 2.2: row-level
    # chunks are indexed ALONGSIDE the whole-table chunk, not instead of it).
    assert len(nodes) == 2
    node = next(n for n in nodes if n.metadata["content_type"] == "table")
    assert node.text == "Tableau (page 2) — Sommaire\n|a|b|\n|---|---|\n|1|2|"
    assert node.metadata["caption"] == "Sommaire"


def test_table_node_without_caption_omits_the_dash() -> None:
    raw = _table_raw(page=3, caption=None)
    nodes = build_text_index_nodes([raw], PDF_SHA, SOURCE_FILE)

    assert nodes[0].text.startswith("Tableau (page 3)\n")
    assert "—" not in nodes[0].text


def test_image_description_node_includes_all_non_empty_parts() -> None:
    raw = _image_raw(
        page=5, caption="Vue de la pompe", description="Pompe centrifuge.", visible_text="EX"
    )
    nodes = build_text_index_nodes([raw], PDF_SHA, SOURCE_FILE)

    assert len(nodes) == 1
    node = nodes[0]
    assert node.metadata["content_type"] == "image_description"
    assert "Figure (page 5) — photo." in node.text
    assert "Légende : Vue de la pompe." in node.text
    assert "Description : Pompe centrifuge." in node.text
    assert "Texte visible : EX." in node.text
    assert node.metadata["image_id"] == "abcd1234_p1_i0"
    assert node.metadata["safety_relevant"] is False
    assert node.metadata["description_status"] == "ok"


def test_image_description_node_omits_empty_parts() -> None:
    raw = _image_raw(caption=None, description=None, visible_text="")
    nodes = build_text_index_nodes([raw], PDF_SHA, SOURCE_FILE)

    text = nodes[0].text
    assert "Légende" not in text
    assert "Description" not in text
    assert "Texte visible" not in text


def test_build_text_index_nodes_skips_unknown_content_types() -> None:
    raw = {"id_": "x", "text": "", "metadata": {"content_type": "something_else"}}
    nodes = build_text_index_nodes([raw], PDF_SHA, SOURCE_FILE)

    assert nodes == []


def test_node_ids_are_deterministic_across_calls() -> None:
    raw_nodes = [_text_raw(), _table_raw(), _image_raw()]

    first = build_text_index_nodes(raw_nodes, PDF_SHA, SOURCE_FILE)
    second = build_text_index_nodes(raw_nodes, PDF_SHA, SOURCE_FILE)

    assert [n.id_ for n in first] == [n.id_ for n in second]
    # text, table, table_row (one row in _table_raw), image_description.
    assert len({n.id_ for n in first}) == 4


def test_build_image_index_nodes_attaches_precomputed_embedding() -> None:
    raw = _image_raw()
    clip_vectors = {"abcd1234_p1_i0": [0.1, 0.2, 0.3]}

    nodes = build_image_index_nodes([raw], clip_vectors, PDF_SHA, SOURCE_FILE)

    assert len(nodes) == 1
    node = nodes[0]
    assert node.embedding == [0.1, 0.2, 0.3]
    assert node.metadata["content_type"] == "image"
    assert node.metadata["image_id"] == "abcd1234_p1_i0"
    assert node.metadata["kind_fr"] == "photo"


def test_build_image_index_nodes_skips_images_without_a_vector() -> None:
    raw = _image_raw()

    nodes = build_image_index_nodes([raw], {}, PDF_SHA, SOURCE_FILE)

    assert nodes == []


# --- M7.4 section 2.1: pictogram table cells -------------------------------


def _pictogram_table_raw(page: int = 5) -> dict:
    return {
        "id_": "abcd1234_p5_t0",
        "text": "|Symbole|Description|\n|---|---|\n||Risque d'atmosphère explosive|",
        "metadata": {
            "content_type": "table",
            "table_id": "abcd1234_p5_t0",
            "page": page,
            "row_count": 2,
            "col_count": 2,
            "caption": "Symboles complémentaires",
            # row 0 = header (bboxes irrelevant here), row 1 = the one data row.
            "cell_bboxes": [
                [[0, 0, 10, 10], [10, 0, 100, 10]],
                [[0, 10, 10, 20], [10, 10, 100, 20]],
            ],
            "pdf_sha256": PDF_SHA,
            "source_file": SOURCE_FILE,
        },
    }


def _pictogram_image_raw(
    page: int = 5, bbox=(1, 11, 9, 19), description: str | None = "un triangle jaune avec EX"
) -> dict:
    return {
        "image_path": "data/media/abcd1234/abcd1234_p5_i7.png",
        "text": "",
        "metadata": {
            "content_type": "image",
            "image_id": "abcd1234_p5_i7",
            "sha256": "cafef00d",
            "page": page,
            "bbox": list(bbox),
            "caption": None,
            "kind_fr": "pictogramme de sécurité",
            "description": description,
            "description_status": "ok",
            "pdf_sha256": PDF_SHA,
            "source_file": SOURCE_FILE,
        },
    }


def test_empty_cell_overlapping_a_described_image_gets_a_pictogram_placeholder() -> None:
    nodes = build_text_index_nodes(
        [_pictogram_table_raw(), _pictogram_image_raw()], PDF_SHA, SOURCE_FILE
    )
    table_node = next(n for n in nodes if n.metadata["content_type"] == "table")

    assert "[pictogramme : un triangle jaune avec EX]" in table_node.text
    assert "Risque d'atmosphère explosive" in table_node.text


def test_empty_cell_with_no_overlapping_image_is_left_alone() -> None:
    raw = _pictogram_table_raw()
    nodes = build_text_index_nodes([raw], PDF_SHA, SOURCE_FILE)
    table_node = next(n for n in nodes if n.metadata["content_type"] == "table")

    assert "[pictogramme" not in table_node.text


def test_empty_cell_overlapping_an_undescribed_image_gets_a_bare_placeholder() -> None:
    nodes = build_text_index_nodes(
        [
            _pictogram_table_raw(),
            _pictogram_image_raw(description=None),
        ],
        PDF_SHA,
        SOURCE_FILE,
    )
    table_node = next(n for n in nodes if n.metadata["content_type"] == "table")

    assert "[pictogramme]" in table_node.text


def test_a_cell_that_already_has_real_text_is_never_overwritten() -> None:
    raw = _pictogram_table_raw()
    raw["text"] = "|Symbole|Description|\n|---|---|\n|déjà du texte|Risque d'atmosphère explosive|"
    nodes = build_text_index_nodes([raw, _pictogram_image_raw()], PDF_SHA, SOURCE_FILE)
    table_node = next(n for n in nodes if n.metadata["content_type"] == "table")

    assert "déjà du texte" in table_node.text
    assert "[pictogramme" not in table_node.text
