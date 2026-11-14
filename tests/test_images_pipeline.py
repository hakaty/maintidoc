import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage

from app.config import Settings
from app.images.describe import PROMPT_VERSION, DescribeResult
from app.images.pipeline import (
    CACHE_FILENAME,
    CLIP_EMBEDDINGS_FILENAME,
    CLIP_INDEX_FILENAME,
    run_enrichment,
)
from app.ingestion.pipeline import node_from_line

FAKE_DIM = 4


class _FakeClipEncoder:
    """Counts calls so tests can assert on how much work each run actually did."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings
        self.unloaded = False

    @property
    def embedding_dim(self) -> int:
        return FAKE_DIM

    @property
    def logit_scale(self) -> float:
        return 10.0

    @property
    def peak_vram_mb(self) -> float | None:
        return 123.4

    def encode_images(self, paths: list[Path]) -> np.ndarray:
        CLIP_CALLS.extend(str(p) for p in paths)
        return np.array(
            [[float(i + 1), 0.0, 0.0, 0.0] for i in range(len(paths))], dtype=np.float32
        )

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        return np.array([[1.0, 0.0, 0.0, 0.0] for _ in texts], dtype=np.float32)

    def unload(self) -> None:
        self.unloaded = True


CLIP_CALLS: list[str] = []
VLM_CALLS: list[str] = []


def _fake_describe_image(
    image_path: Path,
    caption: str | None,
    page: int,
    context_text: str,
    fallback_kind: str = "photo",
    settings: Settings | None = None,
) -> DescribeResult:
    VLM_CALLS.append(str(image_path))
    return DescribeResult(
        description=f"description for {image_path.name}",
        visible_text="",
        kind=fallback_kind,
        safety_relevant=False,
        confidence="high",
        status="ok",
        seconds=0.01,
    )


@pytest.fixture(autouse=True)
def _patch_models(monkeypatch: pytest.MonkeyPatch) -> None:
    CLIP_CALLS.clear()
    VLM_CALLS.clear()
    monkeypatch.setattr("app.images.pipeline.ClipEncoder", _FakeClipEncoder)
    monkeypatch.setattr("app.images.pipeline.get_embedding_dim", lambda model: FAKE_DIM)
    monkeypatch.setattr("app.images.pipeline.describe_image", _fake_describe_image)
    monkeypatch.setattr("app.images.pipeline.unload_ollama_vlm", lambda settings=None: None)
    monkeypatch.setattr("app.images.pipeline.get_ollama_vram_usage", lambda settings=None: 56.7)


def _make_processed_dir(
    tmp_path: Path, image_specs: list[tuple[str, str, int, list[int]]]
) -> Path:
    processed_dir = tmp_path / "processed"
    media_dir = processed_dir / "media"
    media_dir.mkdir(parents=True)

    nodes = []
    for image_id, sha256, page, other_pages in image_specs:
        image_path = media_dir / f"{image_id}.png"
        PILImage.new("RGB", (16, 16), color=(50, 50, 50)).save(image_path)
        nodes.append(
            {
                "image_path": str(image_path),
                "text": f"caption for {image_id}",
                "metadata": {
                    "content_type": "image",
                    "image_id": image_id,
                    "sha256": sha256,
                    "page": page,
                    "bbox": [0, 0, 10, 10],
                    "caption": f"caption for {image_id}",
                    "width": 16,
                    "height": 16,
                    "other_pages": other_pages,
                    "source": "embedded",
                    "kind": "figure",
                    "pdf_sha256": "deadbeef",
                    "source_file": "dummy.pdf",
                    "description": "",
                    "clip_embedding": None,
                },
                "class_name": "ImageNode",
            }
        )
    nodes.append(
        {
            "text": "Some surrounding page text.",
            "metadata": {"content_type": "text", "page": 1, "chunk_index": 0},
            "class_name": "TextNode",
        }
    )

    nodes_path = processed_dir / "nodes.jsonl"
    with nodes_path.open("w", encoding="utf-8") as fh:
        for node in nodes:
            fh.write(json.dumps(node, ensure_ascii=False) + "\n")

    manifest = {"pdf_sha256": "deadbeef", "total_pages": 1, "counts": {}}
    (processed_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    return processed_dir


def _settings() -> Settings:
    return Settings(_env_file=None)


def test_fresh_run_processes_all_images(tmp_path: Path) -> None:
    processed_dir = _make_processed_dir(
        tmp_path, [("img1", "sha1", 1, []), ("img2", "sha2", 2, [])]
    )

    run_enrichment(processed_dir, settings=_settings())

    assert len(CLIP_CALLS) == 2
    assert len(VLM_CALLS) == 2


def test_second_run_is_idempotent_no_new_calls(tmp_path: Path) -> None:
    processed_dir = _make_processed_dir(
        tmp_path, [("img1", "sha1", 1, []), ("img2", "sha2", 2, [])]
    )

    run_enrichment(processed_dir, settings=_settings())
    CLIP_CALLS.clear()
    VLM_CALLS.clear()
    run_enrichment(processed_dir, settings=_settings())

    assert CLIP_CALLS == []
    assert VLM_CALLS == []


def test_resumable_after_interruption(tmp_path: Path) -> None:
    processed_dir = _make_processed_dir(
        tmp_path, [("img1", "sha1", 1, []), ("img2", "sha2", 2, [])]
    )
    # Simulate an interrupted run: only img1's cache entry exists.
    cache_path = processed_dir / CACHE_FILENAME
    cache_path.write_text(
        json.dumps(
            {
                "image_id": "img1",
                "sha256": "sha1",
                "clip": {
                    "model": _settings().clip_model,
                    "pretrained": _settings().clip_pretrained,
                    "label": "a logo",
                    "label_score": 0.9,
                    "category_fr": "logo",
                    "top3": [["a logo", 0.9]],
                },
                "vlm": {
                    "model": _settings().ollama_vlm_model,
                    "prompt_version": PROMPT_VERSION,
                    "description": "already done",
                    "visible_text": "",
                    "kind": "logo",
                    "safety_relevant": False,
                    "confidence": "high",
                    "status": "ok",
                    "seconds": 1.0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    run_enrichment(processed_dir, settings=_settings())

    assert [str(processed_dir / "media" / "img2.png")] == CLIP_CALLS
    assert [str(processed_dir / "media" / "img2.png")] == VLM_CALLS


def test_force_recomputes_everything(tmp_path: Path) -> None:
    processed_dir = _make_processed_dir(tmp_path, [("img1", "sha1", 1, [])])

    run_enrichment(processed_dir, settings=_settings())
    CLIP_CALLS.clear()
    VLM_CALLS.clear()
    run_enrichment(processed_dir, force=True, settings=_settings())

    assert len(CLIP_CALLS) == 1
    assert len(VLM_CALLS) == 1


def test_limit_restricts_images_processed(tmp_path: Path) -> None:
    processed_dir = _make_processed_dir(
        tmp_path, [("img1", "sha1", 1, []), ("img2", "sha2", 2, []), ("img3", "sha3", 3, [])]
    )

    run_enrichment(processed_dir, limit=1, settings=_settings())

    assert len(CLIP_CALLS) == 1
    assert len(VLM_CALLS) == 1


def test_no_vlm_flag_skips_descriptions(tmp_path: Path) -> None:
    processed_dir = _make_processed_dir(tmp_path, [("img1", "sha1", 1, [])])

    run_enrichment(processed_dir, use_vlm=False, settings=_settings())

    assert VLM_CALLS == []
    assert len(CLIP_CALLS) == 1


def test_no_clip_flag_skips_embeddings(tmp_path: Path) -> None:
    processed_dir = _make_processed_dir(tmp_path, [("img1", "sha1", 1, [])])

    run_enrichment(processed_dir, use_clip=False, settings=_settings())

    assert CLIP_CALLS == []
    assert len(VLM_CALLS) == 1
    assert not (processed_dir / CLIP_EMBEDDINGS_FILENAME).exists()


def test_deduped_image_gets_single_shared_enrichment(tmp_path: Path) -> None:
    """An image kept once but appearing on other_pages must get exactly one
    cache entry/description, covering all the pages it appears on."""
    processed_dir = _make_processed_dir(
        tmp_path, [("img1", "sha1", 4, [6, 7, 12, 18])]
    )

    run_enrichment(processed_dir, settings=_settings())

    cache_lines = (processed_dir / CACHE_FILENAME).read_text(encoding="utf-8").splitlines()
    assert len(cache_lines) == 1
    assert len(CLIP_CALLS) == 1
    assert len(VLM_CALLS) == 1

    nodes = [
        json.loads(line)
        for line in (processed_dir / "nodes.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    image_nodes = [n for n in nodes if n["metadata"].get("content_type") == "image"]
    assert len(image_nodes) == 1
    assert image_nodes[0]["metadata"]["other_pages"] == [6, 7, 12, 18]
    assert image_nodes[0]["metadata"]["description"]


def test_nodes_jsonl_reloadable_after_enrichment(tmp_path: Path) -> None:
    processed_dir = _make_processed_dir(tmp_path, [("img1", "sha1", 1, [])])

    run_enrichment(processed_dir, settings=_settings())

    lines = (processed_dir / "nodes.jsonl").read_text(encoding="utf-8").splitlines()
    nodes = [node_from_line(line) for line in lines if line.strip()]
    image_node = next(n for n in nodes if n.metadata["content_type"] == "image")
    assert image_node.metadata["description"] == "description for img1.png"
    assert image_node.metadata["description_status"] == "ok"


def test_npy_and_index_are_consistent(tmp_path: Path) -> None:
    processed_dir = _make_processed_dir(
        tmp_path, [("img1", "sha1", 1, []), ("img2", "sha2", 2, [])]
    )

    run_enrichment(processed_dir, settings=_settings())

    index = json.loads((processed_dir / CLIP_INDEX_FILENAME).read_text(encoding="utf-8"))
    vectors = np.load(processed_dir / CLIP_EMBEDDINGS_FILENAME)

    assert vectors.shape == (2, FAKE_DIM)
    assert index["dim"] == FAKE_DIM
    assert sorted(index["image_ids"]) == ["img1", "img2"]
    assert len(index["image_ids"]) == vectors.shape[0]


def test_manifest_gains_images_enrichment_section(tmp_path: Path) -> None:
    processed_dir = _make_processed_dir(tmp_path, [("img1", "sha1", 1, [])])

    result = run_enrichment(processed_dir, settings=_settings())

    section = result.manifest["images_enrichment"]
    assert section["images_total"] == 1
    assert section["clip_dim"] == FAKE_DIM
    assert section["description_status_counts"] == {"ok": 1}
    assert "device" in section
    assert section["clip_peak_vram_mb"] == 123.4
    assert section["vlm_peak_vram_mb"] == 56.7
