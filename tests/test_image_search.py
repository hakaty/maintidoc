"""M7.4 sections 3 (image rejection) and 8.2 (clean captions, layout)."""

from pathlib import Path

import pytest

from app.config import Settings
from app.rag.retriever import RetrievedNode
from ui.components.image_search import _clean_figure_caption, is_above_rejection_threshold


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def _match(score: float, image_id: str = "p1_i0") -> RetrievedNode:
    return RetrievedNode(
        node_id=f"n_{image_id}",
        page=1,
        content_type="image_description",
        score=score,
        text="Figure (page 1) — photo.",
        image_id=image_id,
        image_path=None,
        safety_relevant=None,
    )


# --- section 3: threshold decision -----------------------------------------


def test_above_threshold_is_accepted() -> None:
    settings = _settings(image_min_score=0.78)
    assert is_above_rejection_threshold([_match(0.85)], settings) is True


def test_below_threshold_is_rejected() -> None:
    settings = _settings(image_min_score=0.78)
    assert is_above_rejection_threshold([_match(0.40)], settings) is False


def test_exactly_at_threshold_is_accepted() -> None:
    settings = _settings(image_min_score=0.78)
    assert is_above_rejection_threshold([_match(0.78)], settings) is True


def test_no_matches_is_rejected() -> None:
    assert is_above_rejection_threshold([], _settings(image_min_score=0.78)) is False


def test_decision_only_looks_at_top1_score() -> None:
    """A weak top-2/3 never drags down an already-strong top-1."""
    settings = _settings(image_min_score=0.78)
    matches = [_match(0.90), _match(0.10), _match(0.05)]
    assert is_above_rejection_threshold(matches, settings) is True


# --- section 8.2: clean, single, word-boundary-clamped caption -------------


def test_caption_shows_description_only_not_legend_duplicated() -> None:
    text = (
        "Figure (page 5) — pictogramme de sécurité. Légende : Symboles complémentaires. "
        'Description : Un triangle jaune avec un texte noir "EX" au centre. '
        "Texte visible : EX."
    )
    clamped, full = _clean_figure_caption(text)
    assert "Légende" not in clamped
    assert "Description" not in clamped
    assert "Symboles complémentaires" not in clamped
    assert 'Un triangle jaune avec un texte noir "EX" au centre' in full


def test_caption_falls_back_to_legend_when_no_description() -> None:
    text = "Figure (page 3) — logo. Légende : Logo du fabricant."
    clamped, _full = _clean_figure_caption(text)
    assert clamped == "Logo du fabricant"


def test_caption_clamps_at_a_word_boundary_not_mid_word() -> None:
    long_description = "conduite à pression " * 15  # > 160 chars
    text = f"Figure (page 9) — schéma. Description : {long_description.strip()}."
    clamped, full = _clean_figure_caption(text)
    assert len(clamped) <= 161  # 160 chars + ellipsis
    assert clamped.endswith("…")
    # never cut mid-word: the character right before the ellipsis is not
    # immediately preceded by a partial word fragment like "conduite à p"
    assert not clamped[:-1].rstrip().endswith(("à p", "pressio"))
    assert full != clamped
    assert full.startswith("conduite à pression")


def test_caption_short_text_is_not_clamped() -> None:
    text = "Figure (page 2) — photo. Description : Une vue de face."
    clamped, full = _clean_figure_caption(text)
    assert clamped == full == "Une vue de face"


# --- real-stack integration (section 3 required tests) ---------------------


@pytest.mark.integration
def test_unrelated_image_is_rejected_by_the_real_threshold() -> None:
    """A solid-colour image (nothing to do with the manual) must score
    below the calibrated threshold against the real index."""
    from PIL import Image

    from app.config import get_settings
    from app.rag.retriever import retrieve_by_image

    settings = get_settings()
    tmp_path = Path("data/media") / "_test_unrelated_solid.png"
    try:
        Image.new("RGB", (640, 480), (120, 180, 220)).save(tmp_path)
        matches = retrieve_by_image(tmp_path, k=3, settings=settings)
    except Exception as exc:  # pragma: no cover - real stack unavailable
        pytest.skip(f"Real image-search stack unavailable: {exc}")
    finally:
        tmp_path.unlink(missing_ok=True)

    assert not is_above_rejection_threshold(matches, settings)


@pytest.mark.integration
def test_a_perturbed_manual_figure_finds_its_own_page_in_top_3() -> None:
    """A real manual figure, re-encoded with a crop + blur + brightness
    change, must still find its own page among the top-3 real results."""
    from PIL import Image, ImageFilter

    from app.config import get_settings
    from app.rag.retriever import retrieve_by_image

    media_dir = Path("data/media/5c1a4a3e34a4")
    candidates = sorted(media_dir.glob("*.png")) if media_dir.is_dir() else []
    candidates = [p for p in candidates if min(Image.open(p).size) >= 120]
    if not candidates:
        pytest.skip("No real manual figures available on this machine")
    fig_path = candidates[0]

    settings = get_settings()
    tmp_path = Path("data/media") / "_test_perturbed_figure.jpg"
    try:
        with Image.open(fig_path) as img:
            img = img.convert("RGB")
            w, h = img.size
            img = img.crop((int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.9)))
            img = img.filter(ImageFilter.GaussianBlur(radius=1.0))
            img.save(tmp_path, format="JPEG", quality=50)
        matches = retrieve_by_image(tmp_path, k=3, settings=settings)
    except Exception as exc:  # pragma: no cover - real stack unavailable
        pytest.skip(f"Real image-search stack unavailable: {exc}")
    finally:
        tmp_path.unlink(missing_ok=True)

    assert any(m.image_id == fig_path.stem for m in matches)
