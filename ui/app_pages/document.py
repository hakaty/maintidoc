"""Document page (file-based st.Page, dispatched by ui/streamlit_app.py)."""

import streamlit as st

from ui.components.document import render_document_tab

st.markdown(
    '<div class="md-h1">Document</div>'
    '<div class="md-muted">Importer un nouveau manuel ou choisir le document actif.</div>',
    unsafe_allow_html=True,
)

with st.container(key="wide_content"):
    render_document_tab(st.session_state["_resolved_settings"])
