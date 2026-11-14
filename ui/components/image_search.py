"""Image search page: photo -> similar figures (+ optional question).

Bug fix (section 0d): the camera used to open automatically because
st.camera_input was rendered unconditionally on every run. It now only
renders after the user clicks the "Prendre une photo" ghost button, gated
by a session_state flag.
"""

import tempfile
from pathlib import Path

import streamlit as st

from app.config import Settings
from app.rag import service
from app.rag.retriever import retrieve_by_image
from ui.components.cards import render_answer
from ui.components.uploads import UploadValidationError, strip_exif_and_save, validate_image_bytes

TEMP_UPLOAD_DIR = Path(tempfile.gettempdir()) / "maintidoc_uploads"
_CONTENT_LABELS = {"text": "passage", "table": "tableau", "image_description": "figure"}


def _save_uploaded_image(data: bytes) -> Path:
    validate_image_bytes(data)
    return strip_exif_and_save(data, TEMP_UPLOAD_DIR)


def _render_match_card(match, is_best: bool) -> None:
    label = _CONTENT_LABELS.get(match.content_type or "", match.content_type or "")
    with st.container(border=True):
        if is_best:
            st.badge("Meilleure correspondance", icon=":material/star:", color="blue")
        cols = st.columns([1, 2])
        with cols[0]:
            if match.image_path:
                st.image(match.image_path, width="stretch", alt=f"Figure page {match.page}")
        with cols[1]:
            st.markdown(
                f'<span class="md-mono">p. {match.page}</span> · {label}',
                unsafe_allow_html=True,
            )
            st.progress(max(0.0, min(1.0, float(match.score))))
            st.markdown(f'<span class="md-mono">{match.score:.3f}</span>', unsafe_allow_html=True)
            st.caption(match.text[:200])


def render_image_search_tab(settings: Settings) -> None:
    st.markdown("#### Votre image")

    left, right = st.columns([1, 1.4])

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
        if not matches:
            st.info("Aucune figure similaire trouvée dans le manuel actif.", icon=":material/info:")
            return
        for i, match in enumerate(matches):
            _render_match_card(match, is_best=(i == 0))

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
