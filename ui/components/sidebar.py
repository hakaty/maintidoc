"""Sidebar extras: service status, advanced settings, new conversation,
recents. Logo and page navigation themselves are rendered by
ui/streamlit_app.py via st.logo()/st.navigation() before this module runs --
this module only adds the content that follows them in the sidebar.

Bug fix (section 0b): the previous stylesheet forced
`color: white !important` on every sidebar descendant on top of a dark
sidebar background. That made the "Fournisseur" select's white internal
background collide with forced-white text (invisible), and did the same to
the "Nouvelle conversation" button's label. The sidebar now uses the same
(light) theme as the main area -- no per-descendant colour override.

Bug fix (section 0d of the pixel-match task): status rows used to build a
`:material/check_circle:`/`:material/error:` shortcode string inside a raw
`<div>` block passed to `st.markdown(unsafe_allow_html=True)`. Streamlit's
markdown parser does not substitute shortcodes inside a literal multi-line
HTML block (it is passed through as raw HTML), so the shortcode text
rendered literally (observed in a real run: "``:material_error: Base de
données · hors ligne``"). The reference design doesn't use an icon there
at all -- just a coloured dot (see ui/icons.py::status_dot) -- so the fix
matches it exactly instead of fighting the shortcode-in-HTML limitation.
"""

from dataclasses import dataclass

import psycopg
import streamlit as st

from app.config import Settings, get_settings
from app.device import resolve_device
from ui.icons import status_dot

# Section 0f: short timeouts so a dead DB/Ollama never stalls page render.
CHECK_TIMEOUT_SECONDS = 5


@dataclass
class SidebarState:
    provider: str
    k: int
    search_mode: str
    query_expansion: bool
    active_pdf_sha256: str | None
    new_conversation: bool = False


def _check_db(settings: Settings) -> bool:
    try:
        psycopg.connect(settings.database_url, connect_timeout=CHECK_TIMEOUT_SECONDS).close()
        return True
    except psycopg.OperationalError:
        return False


def _check_ollama(settings: Settings) -> bool:
    import httpx

    try:
        response = httpx.get(
            f"{settings.ollama_base_url}/api/tags", timeout=CHECK_TIMEOUT_SECONDS
        )
        return response.status_code == 200
    except httpx.HTTPError:
        return False


def _gpu_or_cpu_label(settings: Settings) -> str:
    try:
        device = resolve_device(settings, "UI")
    except RuntimeError:
        return "Mode CPU"
    if device != "cuda":
        return "Mode CPU"
    import torch

    return torch.cuda.get_device_name(0)


def _status_row(ok: bool, online_label: str, offline_label: str) -> None:
    # Colour is never the only signal: the dot colour is accompanied by the
    # explicit "en ligne"/"hors ligne" text (never relying on colour alone).
    import html as _html

    label = _html.escape(online_label if ok else offline_label)
    dot = status_dot("#157347" if ok else "#B42318")
    st.markdown(
        f'<span style="display:flex;align-items:center;gap:8px;font-size:14px;">'
        f"{dot}{label}</span>",
        unsafe_allow_html=True,
    )


def _render_new_conversation_and_recents() -> bool:
    clicked = st.button(
        "Nouvelle conversation",
        icon=":material/add_comment:",
        width="stretch",
        type="secondary",
    )
    if clicked:
        from app.agent.agent import reset_session

        session_id = st.session_state.get("session_id")
        if session_id:
            reset_session(session_id)
        st.session_state["chat_history"] = []
        st.rerun()

    recent = [entry["question"] for entry in st.session_state.get("chat_history", [])]
    if recent:
        st.caption("Récentes")
        for question in recent[-3:]:
            st.badge(question, icon=":material/history:")
    return clicked


def render_sidebar(
    settings: Settings | None = None, show_new_conversation: bool = False
) -> SidebarState:
    settings = settings or get_settings()

    with st.sidebar:
        new_conversation = False
        if show_new_conversation:
            new_conversation = _render_new_conversation_and_recents()

        with st.container(border=True):
            st.caption("État des services")
            db_ok = _check_db(settings)
            _status_row(db_ok, "Base de données · en ligne", "Base de données · hors ligne")
            if settings.llm_provider == "ollama":
                ollama_ok = _check_ollama(settings)
                _status_row(
                    ollama_ok, "Qwen 7B · Ollama · en ligne", "Qwen 7B · Ollama · hors ligne"
                )
            else:
                _status_row(True, "OpenRouter configuré", "OpenRouter configuré")
            st.caption(_gpu_or_cpu_label(settings))

        with st.expander("Réglages avancés", icon=":material/tune:"):
            provider_options = ["ollama"]
            if settings.openrouter_api_key:
                provider_options.append("openrouter")
            provider = st.selectbox(
                "Fournisseur",
                provider_options,
                index=provider_options.index(settings.llm_provider)
                if settings.llm_provider in provider_options
                else 0,
                help="OpenRouter n'apparaît que si une clé API est configurée dans .env.",
            )
            k = st.slider("Nombre de passages (k)", min_value=3, max_value=15, value=settings.rag_k)
            search_mode = st.segmented_control(
                "Mode de recherche", ["hybride", "vecteur"], default="hybride"
            )
            query_expansion = st.toggle(
                "Reformulation de requête", value=settings.rag_query_expansion
            )

    return SidebarState(
        provider=provider,
        k=k,
        search_mode="hybrid" if search_mode == "hybride" else "vector",
        query_expansion=query_expansion,
        active_pdf_sha256=st.session_state.get("active_pdf_sha256"),
        new_conversation=new_conversation,
    )
