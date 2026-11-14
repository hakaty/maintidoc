"""Image search page (file-based st.Page, dispatched by ui/streamlit_app.py)."""

import streamlit as st

from ui.components.image_search import render_image_search_tab

st.markdown(
    '<div class="md-h1">Recherche par image</div>'
    '<div class="md-muted">Importez une photo pour retrouver les figures similaires '
    "dans le manuel.</div>",
    unsafe_allow_html=True,
)

with st.container(key="wide_content"):
    render_image_search_tab(st.session_state["_resolved_settings"])
