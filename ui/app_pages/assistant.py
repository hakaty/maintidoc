"""Assistant page (file-based st.Page, dispatched by ui/streamlit_app.py).

Thin wrapper only: all business logic lives in ui/components/chat.py and
app/rag|agent -- this file is deliberately kept to layout/dispatch.
"""

import streamlit as st

from ui.components.chat import render_assistant_tab

# Section 7.1 fix: render_assistant_tab() now calls st.chat_input() itself
# BEFORE opening its own `st.container(key="chat_content")` (required for
# Streamlit to pin the input to the real bottom of the page) -- this page
# must not wrap it in a container of its own.
render_assistant_tab(st.session_state["_resolved_settings"])
