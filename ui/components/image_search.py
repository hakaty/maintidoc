"""Image search page: photo -> similar figures (+ optional question).

Bug fix (section 0d): the camera used to open automatically because
st.camera_input was rendered unconditionally on every run. It now only
renders after the user clicks the "Prendre une photo" ghost button, gated
by a session_state flag.
"""

import html
import re
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image as PILImage

from app.config import Settings
from app.rag import service
from app.rag.retriever import RetrievedNode, retrieve_by_image
from ui.components.cards import render_answer
from ui.components.uploads import UploadValidationError, strip_exif_and_save, validate_image_bytes
from ui.format import fr_decimal

TEMP_UPLOAD_DIR = Path(tempfile.gettempdir()) / "maintidoc_uploads"
_CONTENT_LABELS = {"text": "passage", "table": "tableau", "image_description": "figure"}

CLEAN_CAPTION_MAX_CHARS = 160
# ui/components/image_search.py renders a node's FULL pre-formatted text
# ("Figure (page N) — kind. Légende : X. Description : Y. Texte visible :
# Z."), not the raw legend/description fields -- app.indexing.nodes doesn't
# carry those separately onto RetrievedNode. Parsing them back out here
# (section 8.2 fix) avoids showing "Légende : ... Description : ..." twice
# when both happen to start with the same words, and avoids a mid-word cut.
_DESCRIPTION_PATTERN = re.compile(r"Description\s*:\s*(.+?)(?:\s*Texte visible\s*:|$)", re.DOTALL)
_LEGEND_PATTERN = re.compile(r"Légende\s*:\s*(.+?)(?:\s*Description\s*:|$)", re.DOTALL)


def _clean_figure_caption(full_text: str) -> tuple[str, str]:
    """Returns (clamped, full) -- ONE caption (the description if present,
    else the legend), never both concatenated, clamped at a WORD boundary
    (never mid-word) with an ellipsis."""
    match = _DESCRIPTION_PATTERN.search(full_text) or _LEGEND_PATTERN.search(full_text)
    full = match.group(1).strip().rstrip(".") if match else full_text.strip()
    if len(full) <= CLEAN_CAPTION_MAX_CHARS:
        return full, full
    clamped = full[:CLEAN_CAPTION_MAX_CHARS].rsplit(" ", 1)[0] + "…"
    return clamped, full


def _save_uploaded_image(data: bytes) -> Path:
    validate_image_bytes(data)
    return strip_exif_and_save(data, TEMP_UPLOAD_DIR)


def _flatten_to_white(image_path: str) -> PILImage.Image | str:
    """M7.4b section 4 fix: a figure PNG with an alpha channel was rendered
    on a BLACK background by st.image (the browser's default behind a
    transparent PNG), making several figures (p. 23, p. 24, p. 10) almost
    unreadable. Composite onto an opaque WHITE canvas before display -- the
    same fix belongs at extraction time too, but this guarantees the UI
    never shows black regardless of where a transparent PNG came from.
    Falls back to the raw path (unchanged behaviour) if the file can't be
    opened as an image, instead of crashing."""
    try:
        with PILImage.open(image_path) as img:
            if img.mode not in ("RGBA", "LA", "PA"):
                return image_path
            rgba = img.convert("RGBA")
            background = PILImage.new("RGB", rgba.size, (255, 255, 255))
            background.paste(rgba, mask=rgba.split()[-1])
            return background
    except OSError:
        return image_path


def _render_match_card(match: RetrievedNode, is_best: bool, needs_review: bool = False) -> None:
    label = _CONTENT_LABELS.get(match.content_type or "", match.content_type or "")
    clamped_caption, full_caption = _clean_figure_caption(match.text)
    with st.container(border=True):
        if is_best:
            st.badge("Meilleure correspondance", icon=":material/star:", color="blue")
        elif needs_review:
            # M7.4b section 4 fix: a result scoring between image_min_score
            # and image_review_score must NOT look identical to a confident
            # match -- a muted card + a small amber-outline label instead.
            st.markdown(
                '<span style="display:inline-flex;align-items:center;gap:4px;'
                "font-size:12px;color:#92600A;border:1px solid #E8B339;"
                'border-radius:9999px;padding:2px 10px;">À vérifier</span>',
                unsafe_allow_html=True,
            )
        cols = st.columns([1, 2])
        with cols[0]:
            if match.image_path:
                st.image(
                    _flatten_to_white(match.image_path),
                    width="stretch",
                    alt=f"Figure page {match.page}",
                )
        with cols[1]:
            # M7.4b section 4 fix: "p. N" and "figure" at the SAME font size,
            # with a non-breaking space between "p." and the number (same
            # pattern as ui/components/cards.py::render_source_row).
            st.markdown(
                f'<span class="md-mono" style="font-size:14px;">p.&nbsp;{match.page}</span>'
                f'<span style="font-size:14px;"> · {html.escape(label)}</span>',
                unsafe_allow_html=True,
            )
            st.progress(max(0.0, min(1.0, float(match.score))))
            # French decimal format everywhere (section 4 fix): "0,846", not
            # "0.846".
            st.markdown(
                f'<span class="md-mono">{fr_decimal(match.score)}</span>', unsafe_allow_html=True
            )
            st.caption(clamped_caption)
            if full_caption != clamped_caption:
                # M7.4c fix: keyed by node_id -- several result cards render
                # in the SAME run (up to 3 matches), and an unkeyed expander
                # with an identical label+icon across them is a duplicate-
                # element risk (same root cause as the chat history crash).
                with st.expander(
                    "Voir plus", icon=":material/expand_more:", key=f"figure_more_{match.node_id}"
                ):
                    st.write(full_caption)


def is_above_rejection_threshold(matches: list[RetrievedNode], settings: Settings) -> bool:
    """Pure decision function (section 3): the top-1 CLIP score must clear
    `settings.image_min_score` (calibrated on real data, see
    scripts/_calibrate_image_threshold.py) for this to be treated as a real
    match rather than a coincidental/unrelated one."""
    if not matches:
        return False
    return matches[0].score >= settings.image_min_score


def needs_review(score: float, settings: Settings) -> bool:
    """Pure decision function (M7.4b section 2/4 "middle zone"): a score
    at/above image_min_score but below image_review_score is a real
    candidate, not confidently a match -- shown as "À vérifier" rather than
    treated the same as a confident result."""
    return score < settings.image_review_score


def _render_rejection_empty_state() -> None:
    """Section 3 UI: below IMAGE_MIN_SCORE, a clear "no match" state --
    never a false "Meilleure correspondance" on a weak/unrelated image."""
    st.markdown(
        '<div class="md-neutral-card" style="display:flex;flex-direction:column;gap:4px;">'
        '<span style="font-size:14px;font-weight:700;">'
        "Aucune figure du manuel ne ressemble à cette image.</span>"
        '<span style="font-size:14px;color:#334155;">Essayez une photo de l\'équipement, '
        "d'une pièce ou d'une plaque signalétique.</span></div>",
        unsafe_allow_html=True,
    )


def _render_results(matches: list[RetrievedNode], settings: Settings) -> None:
    """Section 3 UI: the "Meilleure correspondance" badge, and the results
    themselves, only appear above IMAGE_MIN_SCORE -- below it, the match is
    shown (if at all) only inside a COLLAPSED "Résultats peu fiables"
    expander (section 4 fix: `expanded=False` made explicit), with no badge
    on any of them. M7.4b section 4 "middle zone": once accepted, a result
    scoring between image_min_score and image_review_score is still shown,
    but flagged "À vérifier" instead of looking identical to the best
    match."""
    if not matches or not is_above_rejection_threshold(matches, settings):
        _render_rejection_empty_state()
        if matches:
            with st.expander(
                f"Résultats peu fiables ({len(matches)})",
                icon=":material/info:",
                expanded=False,
            ):
                st.caption(
                    f"Score sous le seuil de fiabilité ({fr_decimal(settings.image_min_score, 2)}) "
                    "-- probablement pas une figure du manuel."
                )
                for match in matches:
                    _render_match_card(match, is_best=False)
        return

    for i, match in enumerate(matches):
        review = i > 0 and needs_review(match.score, settings)
        _render_match_card(match, is_best=(i == 0), needs_review=review)


def render_image_search_tab(settings: Settings) -> None:
    st.markdown("#### Votre image")

    # Section 8.2 fix: a fixed ~360px upload column (was too narrow,
    # truncating filenames like "fig3_deplo…") -- ratio relative to this
    # page's 1120px max-width (.st-key-wide_content), matching the
    # reference's proportions.
    left, right = st.columns([360, 760])

    with left, st.container(border=True):
        uploaded = st.file_uploader(
            "Glissez une photo ou un schéma ici",
            type=["jpg", "jpeg", "png", "webp"],
            help="JPG, PNG ou WebP · 10 Mo max",
        )
        st.caption("JPG, PNG ou WebP · 10 Mo max")

        show_camera = st.session_state.get("image_search_show_camera", False)
        camera_image = None
        if uploaded is None:
            if not show_camera:
                if st.button(
                    "Prendre une photo", icon=":material/photo_camera:", type="tertiary"
                ):
                    st.session_state["image_search_show_camera"] = True
                    st.rerun()
            else:
                camera_image = st.camera_input("Prendre une photo", label_visibility="collapsed")
                if st.button("Annuler la photo", type="tertiary"):
                    st.session_state["image_search_show_camera"] = False
                    st.rerun()

        raw = uploaded.getvalue() if uploaded is not None else None
        raw = raw or (camera_image.getvalue() if camera_image is not None else None)

        if raw is not None:
            st.caption("Vos photos ne sont jamais envoyées à un service externe.")

    if raw is None:
        with right:
            st.markdown("##### Figures similaires dans le manuel")
            st.caption("Importez une image pour lancer la recherche.")
        return

    try:
        image_path = _save_uploaded_image(raw)
    except UploadValidationError as exc:
        with right:
            st.error(str(exc))
        return

    try:
        matches = retrieve_by_image(image_path, k=3, settings=settings)
    finally:
        image_path.unlink(missing_ok=True)

    with right:
        st.markdown("##### Figures similaires dans le manuel")
        st.caption("Similarité visuelle (CLIP), de 0 à 1")
        _render_results(matches, settings)

        st.markdown("###### Une question sur cette image ?")
        st.caption("Posez une question au sujet de ce que montre la photo.")
        question = st.text_input(
            "Une question sur cette image ?", label_visibility="collapsed"
        )
        if st.button("Demander à l'assistant", icon=":material/send:", type="primary") and question:
            image_path2 = _save_uploaded_image(raw)
            try:
                result = service.answer(question, image_path=image_path2, settings=settings)
            finally:
                image_path2.unlink(missing_ok=True)
            render_answer(result, key_prefix="image_search_answer")
