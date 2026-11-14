import numpy as np
import pytest

from app.images.labels import (
    DEFAULT_LABELS_PATH,
    LabelSpec,
    classify_embeddings,
    load_labels,
)


def test_load_labels_from_real_file() -> None:
    labels = load_labels()

    assert len(labels) == 10
    texts = {label.text for label in labels}
    categories = {label.category_fr for label in labels}
    assert "a technical diagram or schematic" in texts
    assert "a safety warning pictogram" in texts
    assert "schéma" in categories
    assert "pictogramme de sécurité" in categories
    assert DEFAULT_LABELS_PATH.is_file()


def test_each_label_has_text_and_category() -> None:
    for label in load_labels():
        assert label.text
        assert label.category_fr


@pytest.fixture
def three_labels() -> list[LabelSpec]:
    return [
        LabelSpec(text="a photo of a cat", category_fr="chat"),
        LabelSpec(text="a photo of a dog", category_fr="chien"),
        LabelSpec(text="a logo", category_fr="logo"),
    ]


def test_classify_embeddings_picks_closest_label(three_labels: list[LabelSpec]) -> None:
    # 2D embeddings for clarity: label 0 points along +x, label 1 along +y,
    # label 2 along -x. One image embedding close to each.
    label_embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    image_embeddings = np.array(
        [
            [0.95, 0.05],  # closest to "chat"
            [0.05, 0.95],  # closest to "chien"
            [-0.9, 0.1],  # closest to "logo"
        ],
        dtype=np.float32,
    )

    results = classify_embeddings(
        image_embeddings, label_embeddings, three_labels, logit_scale=10.0
    )

    assert [r.category_fr for r in results] == ["chat", "chien", "logo"]
    for result in results:
        assert 0.0 <= result.score <= 1.0
        assert len(result.top3) == 3
        # top3 sorted descending by score
        scores = [score for _, score in result.top3]
        assert scores == sorted(scores, reverse=True)


def test_classify_embeddings_score_is_softmax_probability(three_labels: list[LabelSpec]) -> None:
    # Identical embeddings for all 3 labels -> uniform 1/3 probability regardless of scale.
    label_embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    image_embeddings = np.array([[1.0, 0.0]], dtype=np.float32)

    results = classify_embeddings(
        image_embeddings, label_embeddings, three_labels, logit_scale=50.0
    )

    assert results[0].score == pytest.approx(1 / 3, abs=1e-5)


def test_classify_embeddings_empty_images_returns_empty(three_labels: list[LabelSpec]) -> None:
    label_embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    image_embeddings = np.empty((0, 2), dtype=np.float32)

    results = classify_embeddings(
        image_embeddings, label_embeddings, three_labels, logit_scale=10.0
    )

    assert results == []
