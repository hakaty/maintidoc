"""Zero-shot image category labelling via CLIP image/text cosine similarity."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from app.images.clip_encoder import ClipEncoder

DEFAULT_LABELS_PATH = Path("prompts/clip_labels_v1.yaml")


@dataclass(frozen=True)
class LabelSpec:
    text: str
    category_fr: str


@dataclass
class LabelResult:
    label: str
    category_fr: str
    score: float
    top3: list[tuple[str, float]]


def load_labels(path: Path = DEFAULT_LABELS_PATH) -> list[LabelSpec]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [
        LabelSpec(text=item["text"], category_fr=item["category_fr"]) for item in data["labels"]
    ]


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def classify_embeddings(
    image_embeddings: np.ndarray,
    label_embeddings: np.ndarray,
    labels: list[LabelSpec],
    logit_scale: float,
) -> list[LabelResult]:
    """Zero-shot classification from already-computed, L2-normalised embeddings."""
    cosine_sim = image_embeddings @ label_embeddings.T  # (n_images, n_labels)
    probs = _softmax(cosine_sim * logit_scale)

    results = []
    for row in probs:
        order = np.argsort(row)[::-1]
        top_index = int(order[0])
        top3 = [(labels[i].text, float(row[i])) for i in order[:3]]
        results.append(
            LabelResult(
                label=labels[top_index].text,
                category_fr=labels[top_index].category_fr,
                score=float(row[top_index]),
                top3=top3,
            )
        )
    return results


def classify_images(
    encoder: ClipEncoder, image_paths: list[Path], labels: list[LabelSpec] | None = None
) -> list[LabelResult]:
    labels = labels if labels is not None else load_labels()
    image_embeddings = encoder.encode_images(image_paths)
    label_embeddings = encoder.encode_texts([label.text for label in labels])
    return classify_embeddings(image_embeddings, label_embeddings, labels, encoder.logit_scale)
