import json
import uuid
from pathlib import Path

import numpy as np
import psycopg
import pytest
from PIL import Image as PILImage
from scripts.init_db import ensure_vector_extension

from app.config import Settings, get_settings
from app.indexing import stores as stores_module
from app.indexing.stores import (
    build_index,
    get_image_store,
    get_text_store,
    search_image,
    search_image_text,
    search_text,
)

FAKE_TEXT_DIM = 8
FAKE_IMAGE_DIM = 4


def _deterministic_vector(text: str, dim: int) -> np.ndarray:
    seed = (sum(ord(c) for c in text) % (2**31)) or 1
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=dim).astype(np.float32)
    return vector / np.linalg.norm(vector)


class _FakeTextEmbedder:
    def __init__(self, settings: Settings | None = None, for_query: bool = False) -> None:
        self.settings = settings
        self.for_query = for_query

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, FAKE_TEXT_DIM), dtype=np.float32)
        return np.stack([_deterministic_vector(t, FAKE_TEXT_DIM) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return _deterministic_vector(text, FAKE_TEXT_DIM)

    def unload(self) -> None:
        pass


class _FakeClipEncoder:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings

    def encode_images(self, paths: list[Path]) -> np.ndarray:
        return np.stack([_deterministic_vector(str(p), FAKE_IMAGE_DIM) for p in paths])

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        return np.stack([_deterministic_vector(t, FAKE_IMAGE_DIM) for t in texts])

    def unload(self) -> None:
        pass


@pytest.fixture(autouse=True)
def _patch_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stores_module, "TextEmbedder", _FakeTextEmbedder)
    monkeypatch.setattr(stores_module, "ClipEncoder", _FakeClipEncoder)
    monkeypatch.setattr(stores_module, "unload_ollama_models", lambda settings=None: None)
    monkeypatch.setattr(
        stores_module, "get_query_embedder", lambda settings=None: _FakeTextEmbedder(settings)
    )


@pytest.fixture
def db_settings() -> Settings:
    base = get_settings()
    try:
        ensure_vector_extension(base.database_url)
    except psycopg.OperationalError:
        pytest.skip("Database not reachable (run `docker compose up -d db` first)")

    suffix = uuid.uuid4().hex[:8]
    settings = Settings(
        _env_file=None,
        database_url=base.database_url,
        pgvector_text_table=f"test_text_{suffix}",
        pgvector_image_table=f"test_image_{suffix}",
        pgvector_text_dim=FAKE_TEXT_DIM,
        pgvector_image_dim=FAKE_IMAGE_DIM,
    )
    yield settings

    with psycopg.connect(base.database_url, autocommit=True) as conn:
        conn.execute(f'DROP TABLE IF EXISTS "data_{settings.pgvector_text_table}" CASCADE')
        conn.execute(f'DROP TABLE IF EXISTS "data_{settings.pgvector_image_table}" CASCADE')


def _make_processed_dir(tmp_path: Path, pdf_sha256: str) -> Path:
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    image_path = tmp_path / "abcd_p1_i0.png"
    PILImage.new("RGB", (16, 16), color=(10, 20, 30)).save(image_path)

    nodes = [
        {
            "id_": "abcd_p1_c0",
            "text": "La pompe doit etre entretenue regulierement par un technicien.",
            "metadata": {
                "content_type": "text",
                "page": 1,
                "chunk_index": 0,
                "pdf_sha256": pdf_sha256,
                "source_file": "doc.pdf",
                "related_image_ids": ["abcd_p1_i0"],
                "related_table_ids": [],
            },
            "class_name": "TextNode",
        },
        {
            "id_": "abcd_p2_t0",
            "text": "|a|b|\n|---|---|\n|1|2|",
            "metadata": {
                "content_type": "table",
                "table_id": "abcd_p2_t0",
                "page": 2,
                "row_count": 2,
                "col_count": 2,
                "caption": "Sommaire",
                "pdf_sha256": pdf_sha256,
                "source_file": "doc.pdf",
            },
            "class_name": "TextNode",
        },
        {
            "image_path": str(image_path),
            "text": "Figure 1",
            "metadata": {
                "content_type": "image",
                "image_id": "abcd_p1_i0",
                "sha256": "feedface",
                "page": 1,
                "caption": "Figure 1",
                "kind_fr": "photo",
                "clip_label": "a photo of a pump",
                "description": "Une pompe centrifuge.",
                "visible_text": "",
                "safety_relevant": False,
                "description_status": "ok",
                "pdf_sha256": pdf_sha256,
                "source_file": "doc.pdf",
            },
            "class_name": "ImageNode",
        },
    ]
    with (processed_dir / "nodes.jsonl").open("w", encoding="utf-8") as fh:
        for node in nodes:
            fh.write(json.dumps(node, ensure_ascii=False) + "\n")

    manifest = {"pdf_sha256": pdf_sha256, "source_file": "doc.pdf"}
    (processed_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    vector = _deterministic_vector("abcd_p1_i0", FAKE_IMAGE_DIM)
    np.save(processed_dir / "clip_embeddings.npy", np.array([vector], dtype=np.float32))
    (processed_dir / "clip_index.json").write_text(
        json.dumps(
            {"model": "x", "pretrained": "y", "dim": FAKE_IMAGE_DIM, "image_ids": ["abcd_p1_i0"]}
        ),
        encoding="utf-8",
    )
    return processed_dir


def test_get_text_store_passes_configured_table_dim_and_hybrid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    def _fake_from_params(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(stores_module.PGVectorStore, "from_params", staticmethod(_fake_from_params))

    settings = Settings(_env_file=None)
    get_text_store(settings)

    assert captured["table_name"] == settings.pgvector_text_table
    assert captured["embed_dim"] == settings.pgvector_text_dim
    assert captured["hybrid_search"] == settings.pgvector_hybrid_search
    assert captured["text_search_config"] == settings.pgvector_text_search_config


def test_get_image_store_passes_configured_table_and_dim(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def _fake_from_params(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(stores_module.PGVectorStore, "from_params", staticmethod(_fake_from_params))

    settings = Settings(_env_file=None)
    get_image_store(settings)

    assert captured["table_name"] == settings.pgvector_image_table
    assert captured["embed_dim"] == settings.pgvector_image_dim
    assert captured["hybrid_search"] is False


@pytest.mark.integration
def test_build_index_inserts_text_and_image_nodes(tmp_path: Path, db_settings: Settings) -> None:
    pdf_sha256 = "sha_" + uuid.uuid4().hex
    processed_dir = _make_processed_dir(tmp_path, pdf_sha256)

    result = build_index(processed_dir, settings=db_settings)

    assert result.text_nodes_total == 3  # text chunk + table + image_description
    assert result.text_nodes_added == 3
    assert result.text_count_in_db == 3
    assert result.image_nodes_total == 1
    assert result.image_nodes_added == 1
    assert result.image_count_in_db == 1


@pytest.mark.integration
def test_build_index_is_idempotent(tmp_path: Path, db_settings: Settings) -> None:
    pdf_sha256 = "sha_" + uuid.uuid4().hex
    processed_dir = _make_processed_dir(tmp_path, pdf_sha256)

    build_index(processed_dir, settings=db_settings)
    second = build_index(processed_dir, settings=db_settings)

    assert second.text_nodes_added == 0
    assert second.image_nodes_added == 0
    assert second.text_count_in_db == 3
    assert second.image_count_in_db == 1


@pytest.mark.integration
def test_build_index_force_rebuilds_without_duplicating(
    tmp_path: Path, db_settings: Settings
) -> None:
    pdf_sha256 = "sha_" + uuid.uuid4().hex
    processed_dir = _make_processed_dir(tmp_path, pdf_sha256)

    build_index(processed_dir, settings=db_settings)
    result = build_index(processed_dir, force=True, settings=db_settings)

    assert result.text_nodes_added == 3
    assert result.text_count_in_db == 3
    assert result.image_count_in_db == 1


@pytest.mark.integration
def test_search_text_filters_by_content_type(tmp_path: Path, db_settings: Settings) -> None:
    pdf_sha256 = "sha_" + uuid.uuid4().hex
    processed_dir = _make_processed_dir(tmp_path, pdf_sha256)
    build_index(processed_dir, settings=db_settings)

    table_results = search_text(
        "peu importe la requete", k=5, content_type="table", settings=db_settings
    )

    assert len(table_results) == 1
    assert table_results[0].content_type == "table"
    assert table_results[0].page == 2


@pytest.mark.integration
def test_hybrid_search_finds_french_keyword_match(tmp_path: Path, db_settings: Settings) -> None:
    """The fake embedder's vectors are unrelated to text content, so a hit here
    can only come from the French full-text (tsvector) side of hybrid search."""
    pdf_sha256 = "sha_" + uuid.uuid4().hex
    processed_dir = _make_processed_dir(tmp_path, pdf_sha256)
    build_index(processed_dir, settings=db_settings)

    results = search_text("entretenue", k=5, mode="hybrid", settings=db_settings)

    assert any("entretenue" in r.snippet for r in results)


@pytest.mark.integration
def test_vector_mode_does_not_use_hybrid(tmp_path: Path, db_settings: Settings) -> None:
    pdf_sha256 = "sha_" + uuid.uuid4().hex
    processed_dir = _make_processed_dir(tmp_path, pdf_sha256)
    build_index(processed_dir, settings=db_settings)

    results = search_text("entretenue", k=5, mode="vector", settings=db_settings)

    assert len(results) <= 3


@pytest.mark.integration
def test_search_image_and_search_image_text(tmp_path: Path, db_settings: Settings) -> None:
    pdf_sha256 = "sha_" + uuid.uuid4().hex
    processed_dir = _make_processed_dir(tmp_path, pdf_sha256)
    build_index(processed_dir, settings=db_settings)

    query_image = tmp_path / "query.png"
    PILImage.new("RGB", (16, 16), color=(10, 20, 30)).save(query_image)

    image_results = search_image(query_image, k=5, settings=db_settings)
    assert len(image_results) == 1
    assert image_results[0].image_id == "abcd_p1_i0"

    text_results = search_image_text("a pump", k=5, settings=db_settings)
    assert len(text_results) == 1
    assert text_results[0].image_id == "abcd_p1_i0"
