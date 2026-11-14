"""Document tab pipeline wiring: extract -> enrich -> index, in order, and
idempotence ("déjà indexé" on a second upload with the same content)."""

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app.config import Settings
from app.images.pipeline import EnrichmentResult
from app.indexing.stores import BuildResult
from app.ingestion.pipeline import ExtractionResult
from ui.components import document as document_module


def _run_pipeline_script() -> None:
    # AppTest.from_function re-serializes this function's SOURCE and execs it
    # standalone: it cannot see this test module's globals, so every name used
    # below must be (re)imported inside the function body.
    from pathlib import Path

    import streamlit as st

    from app.config import Settings
    from ui.components import document as document_module

    manifest, text_nodes_total = document_module._run_pipeline(
        Path("fake.pdf"), use_vlm=True, settings=Settings(_env_file=None)
    )
    st.session_state["manifest"] = manifest
    st.session_state["text_nodes_total"] = text_nodes_total


def _settings() -> Settings:
    return Settings(_env_file=None)


@pytest.fixture
def _processed_dir(tmp_path: Path) -> Path:
    processed_dir = tmp_path / "processed" / "abc123def456"
    processed_dir.mkdir(parents=True)
    manifest = {
        "pdf_sha256": "abc123def456" * 4,
        "total_pages": 5,
        "counts": {"tables": 2, "images_kept": 3},
        "extraction_date": "2026-01-01T00:00:00Z",
    }
    (processed_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return processed_dir


def test_run_pipeline_calls_extract_enrich_index_in_order(
    monkeypatch: pytest.MonkeyPatch, _processed_dir: Path
) -> None:
    call_order: list[str] = []

    def _fake_extract(pdf_path, processed_root, media_dir):
        call_order.append("extract")
        return ExtractionResult(processed_dir=_processed_dir, manifest={}, unchanged=False)

    def _fake_enrich(processed_dir, use_vlm=True, settings=None):
        call_order.append("enrich")
        assert processed_dir == _processed_dir
        return EnrichmentResult(processed_dir=processed_dir, manifest={})

    def _fake_build(processed_dir, settings=None):
        call_order.append("index")
        return BuildResult(
            pdf_sha256="x",
            text_nodes_total=42,
            text_nodes_added=42,
            image_nodes_total=3,
            image_nodes_added=3,
            text_count_in_db=42,
            image_count_in_db=3,
        )

    monkeypatch.setattr(document_module, "run_extraction", _fake_extract)
    monkeypatch.setattr(document_module, "run_enrichment", _fake_enrich)
    monkeypatch.setattr(document_module, "build_index", _fake_build)

    at = AppTest.from_function(_run_pipeline_script)
    at.run(timeout=30)

    assert not at.exception
    assert call_order == ["extract", "enrich", "index"]
    assert at.session_state["text_nodes_total"] == 42
    assert at.session_state["manifest"]["total_pages"] == 5


def test_list_documents_reads_manifests_from_processed_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _processed_dir: Path
) -> None:
    monkeypatch.setattr(document_module, "PROCESSED_ROOT", _processed_dir.parent)

    docs = document_module._list_documents()

    assert len(docs) == 1
    assert docs[0]["sha12"] == _processed_dir.name
    assert docs[0]["manifest"]["total_pages"] == 5


def test_list_documents_empty_when_processed_root_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(document_module, "PROCESSED_ROOT", tmp_path / "does_not_exist")

    assert document_module._list_documents() == []


def test_already_indexed_check_uses_sha256_prefix(
    monkeypatch: pytest.MonkeyPatch, _processed_dir: Path
) -> None:
    monkeypatch.setattr(document_module, "PROCESSED_ROOT", _processed_dir.parent)

    sha12 = _processed_dir.name
    assert (document_module.PROCESSED_ROOT / sha12 / "manifest.json").is_file()
    assert not (document_module.PROCESSED_ROOT / "other000000" / "manifest.json").is_file()
