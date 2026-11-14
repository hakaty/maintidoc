"""Image enrichment pipeline: CLIP embeddings + zero-shot labels + VLM descriptions.

Reads data/processed/<sha12>/nodes.jsonl (written by app.ingestion), enriches
each ImageNode in place, and writes CLIP vectors to clip_embeddings.npy /
clip_index.json. Resumable: data/processed/<sha12>/image_enrichment.jsonl
caches each image's result, keyed by (image sha256, model, prompt version),
so an interrupted run picks up where it stopped unless --force is given.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from app.config import Settings, get_settings
from app.images.clip_encoder import ClipEncoder, get_embedding_dim
from app.images.describe import (
    PROMPT_VERSION,
    DescribeResult,
    describe_image,
    get_ollama_vram_usage,
    unload_ollama_vlm,
)
from app.images.labels import LabelResult, classify_embeddings, load_labels

logger = logging.getLogger(__name__)

CACHE_FILENAME = "image_enrichment.jsonl"
CLIP_EMBEDDINGS_FILENAME = "clip_embeddings.npy"
CLIP_INDEX_FILENAME = "clip_index.json"


@dataclass
class ImageRecord:
    image_id: str
    sha256: str
    page: int
    caption: str | None
    image_path: Path
    other_pages: list[int]


@dataclass
class EnrichmentResult:
    processed_dir: Path
    manifest: dict


def _vlm_model_name(settings: Settings) -> str:
    if settings.llm_provider == "ollama":
        return settings.ollama_vlm_model
    return settings.openrouter_vlm_model


def _load_raw_nodes(nodes_path: Path) -> list[dict]:
    lines = nodes_path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _image_records(raw_nodes: list[dict]) -> list[ImageRecord]:
    records = []
    for raw in raw_nodes:
        meta = raw.get("metadata", {})
        if meta.get("content_type") != "image":
            continue
        records.append(
            ImageRecord(
                image_id=meta["image_id"],
                sha256=meta["sha256"],
                page=meta["page"],
                caption=meta.get("caption"),
                image_path=Path(raw["image_path"]),
                other_pages=meta.get("other_pages", []),
            )
        )
    return records


def _page_context_text(raw_nodes: list[dict], page: int) -> str:
    chunks = [
        raw.get("text", "")
        for raw in raw_nodes
        if raw.get("metadata", {}).get("content_type") == "text"
        and raw.get("metadata", {}).get("page") == page
    ]
    return " ".join(chunks)


def _load_cache(cache_path: Path) -> dict[str, dict]:
    if not cache_path.is_file():
        return {}
    entries: dict[str, dict] = {}
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        entries[data["image_id"]] = data
    return entries


def _write_cache(cache_path: Path, entries: dict[str, dict]) -> None:
    with cache_path.open("w", encoding="utf-8") as fh:
        for image_id in sorted(entries):
            fh.write(json.dumps(entries[image_id], ensure_ascii=False) + "\n")


def _clip_cache_valid(entry: dict | None, settings: Settings) -> bool:
    clip = (entry or {}).get("clip")
    if not clip:
        return False
    return (
        clip.get("model") == settings.clip_model
        and clip.get("pretrained") == settings.clip_pretrained
    )


def _vlm_cache_valid(entry: dict | None, settings: Settings) -> bool:
    vlm = (entry or {}).get("vlm")
    if not vlm:
        return False
    return (
        vlm.get("model") == _vlm_model_name(settings)
        and vlm.get("prompt_version") == PROMPT_VERSION
    )


def _load_existing_clip_vectors(processed_dir: Path) -> dict[str, np.ndarray]:
    index_path = processed_dir / CLIP_INDEX_FILENAME
    npy_path = processed_dir / CLIP_EMBEDDINGS_FILENAME
    if not index_path.is_file() or not npy_path.is_file():
        return {}
    index = json.loads(index_path.read_text(encoding="utf-8"))
    vectors = np.load(npy_path)
    return dict(zip(index["image_ids"], vectors, strict=True))


def _save_clip_outputs(
    processed_dir: Path, vectors_by_image_id: dict[str, np.ndarray], settings: Settings, dim: int
) -> None:
    image_ids = sorted(vectors_by_image_id)
    if image_ids:
        matrix = np.stack([vectors_by_image_id[i] for i in image_ids])
    else:
        matrix = np.empty((0, dim))
    np.save(processed_dir / CLIP_EMBEDDINGS_FILENAME, matrix.astype(np.float32))
    index = {
        "model": settings.clip_model,
        "pretrained": settings.clip_pretrained,
        "dim": dim,
        "image_ids": image_ids,
    }
    (processed_dir / CLIP_INDEX_FILENAME).write_text(
        json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _run_clip_stage(
    images: list[ImageRecord],
    cache: dict[str, dict],
    existing_vectors: dict[str, np.ndarray],
    settings: Settings,
    force: bool,
) -> tuple[dict[str, np.ndarray], list[LabelResult | None], float | None]:
    todo = [
        img for img in images if force or not _clip_cache_valid(cache.get(img.image_id), settings)
    ]
    vectors_by_id = dict(existing_vectors)
    results_by_id: dict[str, LabelResult] = {}
    peak_vram_mb: float | None = None

    if todo:
        encoder = ClipEncoder(settings)
        labels = load_labels()
        image_embeddings = encoder.encode_images([img.image_path for img in todo])
        label_embeddings = encoder.encode_texts([label.text for label in labels])
        label_results = classify_embeddings(
            image_embeddings, label_embeddings, labels, encoder.logit_scale
        )
        for img, embedding, label_result in zip(todo, image_embeddings, label_results, strict=True):
            vectors_by_id[img.image_id] = embedding
            results_by_id[img.image_id] = label_result
            cache[img.image_id] = {
                **cache.get(img.image_id, {}),
                "image_id": img.image_id,
                "sha256": img.sha256,
                "clip": {
                    "model": settings.clip_model,
                    "pretrained": settings.clip_pretrained,
                    "label": label_result.label,
                    "label_score": label_result.score,
                    "category_fr": label_result.category_fr,
                    "top3": label_result.top3,
                },
            }
        peak_vram_mb = encoder.peak_vram_mb
        encoder.unload()

    all_results = [results_by_id.get(img.image_id) for img in images]
    return vectors_by_id, all_results, peak_vram_mb


def _run_vlm_stage(
    images: list[ImageRecord],
    raw_nodes: list[dict],
    clip_results: list["LabelResult | None"],
    cache: dict[str, dict],
    settings: Settings,
    force: bool,
) -> float | None:
    todo = [
        (img, clip_result)
        for img, clip_result in zip(images, clip_results, strict=True)
        if force or not _vlm_cache_valid(cache.get(img.image_id), settings)
    ]
    peak_vram_mb: float | None = None

    for img, clip_result in todo:
        fallback_kind = clip_result.category_fr if clip_result else "photo"
        context_text = _page_context_text(raw_nodes, img.page)
        result: DescribeResult = describe_image(
            img.image_path,
            img.caption,
            img.page,
            context_text,
            fallback_kind=fallback_kind,
            settings=settings,
        )
        logger.info("Described %s in %.2fs (%s)", img.image_id, result.seconds, result.status)
        cache[img.image_id] = {
            **cache.get(img.image_id, {}),
            "image_id": img.image_id,
            "sha256": img.sha256,
            "vlm": {
                "model": _vlm_model_name(settings),
                "prompt_version": PROMPT_VERSION,
                "description": result.description,
                "visible_text": result.visible_text,
                "kind": result.kind,
                "safety_relevant": result.safety_relevant,
                "confidence": result.confidence,
                "status": result.status,
                "seconds": result.seconds,
            },
        }
        # `ollama ps` point-in-time reading, sampled per image since Ollama
        # (a separate process) exposes no peak-memory API of its own; the
        # max seen across the batch approximates the peak.
        sample = get_ollama_vram_usage(settings)
        if sample is not None:
            peak_vram_mb = sample if peak_vram_mb is None else max(peak_vram_mb, sample)

    if todo:
        unload_ollama_vlm(settings)

    return peak_vram_mb


def _apply_enrichment_to_nodes(raw_nodes: list[dict], cache: dict[str, dict]) -> None:
    for raw in raw_nodes:
        meta = raw.get("metadata", {})
        if meta.get("content_type") != "image":
            continue
        entry = cache.get(meta["image_id"])
        if entry is None:
            continue
        clip = entry.get("clip")
        if clip:
            meta["clip_label"] = clip["label"]
            meta["clip_label_score"] = clip["label_score"]
            meta["kind_fr"] = clip["category_fr"]
        vlm = entry.get("vlm")
        if vlm:
            meta["description"] = vlm["description"]
            meta["visible_text"] = vlm["visible_text"]
            meta["safety_relevant"] = vlm["safety_relevant"]
            meta["description_confidence"] = vlm["confidence"]
            meta["description_status"] = vlm["status"]
            meta["vlm_model"] = vlm["model"]
            meta["prompt_version"] = vlm["prompt_version"]
            if "kind_fr" not in meta:
                meta["kind_fr"] = vlm["kind"]


def _write_nodes(nodes_path: Path, raw_nodes: list[dict]) -> None:
    with nodes_path.open("w", encoding="utf-8") as fh:
        for raw in raw_nodes:
            fh.write(json.dumps(raw, default=str, ensure_ascii=False) + "\n")


def _build_manifest_section(
    images: list[ImageRecord],
    cache: dict[str, dict],
    settings: Settings,
    dim: int,
    clip_peak_vram_mb: float | None = None,
    vlm_peak_vram_mb: float | None = None,
) -> dict:
    label_distribution: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    vlm_seconds: list[float] = []

    for img in images:
        entry = cache.get(img.image_id, {})
        clip = entry.get("clip")
        if clip:
            category = clip["category_fr"]
            label_distribution[category] = label_distribution.get(category, 0) + 1
        vlm = entry.get("vlm")
        if vlm:
            status_counts[vlm["status"]] = status_counts.get(vlm["status"], 0) + 1
            vlm_seconds.append(vlm["seconds"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "clip_model": settings.clip_model,
        "clip_pretrained": settings.clip_pretrained,
        "clip_dim": dim,
        "vlm_model": _vlm_model_name(settings),
        "prompt_version": PROMPT_VERSION,
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "images_total": len(images),
        "clip_label_distribution": label_distribution,
        "description_status_counts": status_counts,
        "total_vlm_seconds": round(sum(vlm_seconds), 2),
        "avg_vlm_seconds": round(sum(vlm_seconds) / len(vlm_seconds), 2) if vlm_seconds else 0.0,
        "clip_peak_vram_mb": round(clip_peak_vram_mb, 1) if clip_peak_vram_mb is not None else None,
        "vlm_peak_vram_mb": round(vlm_peak_vram_mb, 1) if vlm_peak_vram_mb is not None else None,
    }


def run_enrichment(
    processed_dir: Path,
    use_clip: bool = True,
    use_vlm: bool = True,
    force: bool = False,
    limit: int | None = None,
    settings: Settings | None = None,
) -> EnrichmentResult:
    settings = settings or get_settings()
    nodes_path = processed_dir / "nodes.jsonl"
    manifest_path = processed_dir / "manifest.json"
    cache_path = processed_dir / CACHE_FILENAME

    raw_nodes = _load_raw_nodes(nodes_path)
    images = _image_records(raw_nodes)
    if limit is not None:
        images = images[:limit]

    cache = {} if force else _load_cache(cache_path)

    dim = get_embedding_dim(settings.clip_model) or 0
    clip_results: list[LabelResult | None] = [None] * len(images)
    clip_peak_vram_mb: float | None = None
    vlm_peak_vram_mb: float | None = None
    if use_clip:
        existing_vectors = {} if force else _load_existing_clip_vectors(processed_dir)
        vectors_by_id, clip_results, clip_peak_vram_mb = _run_clip_stage(
            images, cache, existing_vectors, settings, force
        )
        _save_clip_outputs(processed_dir, vectors_by_id, settings, dim)

    if use_vlm:
        vlm_peak_vram_mb = _run_vlm_stage(images, raw_nodes, clip_results, cache, settings, force)

    _write_cache(cache_path, cache)
    _apply_enrichment_to_nodes(raw_nodes, cache)
    _write_nodes(nodes_path, raw_nodes)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["images_enrichment"] = _build_manifest_section(
        images, cache, settings, dim, clip_peak_vram_mb, vlm_peak_vram_mb
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    return EnrichmentResult(processed_dir=processed_dir, manifest=manifest)
