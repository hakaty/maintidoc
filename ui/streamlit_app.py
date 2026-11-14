"""MaintiDoc Streamlit app. Launch with: python -m streamlit run ui/streamlit_app.py

This file is deliberately NOT named app.py: Streamlit puts this script's own
directory (ui/) first on sys.path, so a file named ui/app.py would shadow
the top-level app/ package and break every "from app.* import ..." below
with "ModuleNotFoundError: No module named 'app.config'; 'app' is not a
package" -- a real bug hit on the actual launch command. Inserting the
project root on sys.path (next) is a second, independent safeguard.

Section 0f fix: the page used to go blank for a long time at startup
because model warm-up ran before anything else was drawn. The page shell
(logo, navigation, sidebar) is now rendered FIRST; warm-up runs afterwards
inside a visible st.status, cached once per process via st.cache_resource.

Bug fix (section 0h of the pixel-match task): `st.logo()` + `st.navigation
(position="sidebar")` put Streamlit's OWN logo/nav widgets at fixed
positions it controls, so our custom wordmark ("Mainti"+"Doc") and
per-manual subtitle ended up squeezed in below the nav links instead of
above them, and only the small square icon (not the wordmark) showed. This
now uses `st.navigation(position="hidden")` (no default nav UI at all) and
builds the sidebar ourselves: one single HTML block with the real logo SVG
+ wordmark + subtitle, THEN `st.page_link` for each page -- same order as
the reference.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import contextlib  # noqa: E402
import json  # noqa: E402

import streamlit as st  # noqa: E402

from app.config import Settings, get_settings  # noqa: E402
from app.indexing.embedder import get_query_embedder  # noqa: E402
from app.llm_factory import get_llm  # noqa: E402
from ui.components.cards import render_error_card  # noqa: E402
from ui.components.sidebar import (  # noqa: E402
    CHECK_TIMEOUT_SECONDS,
    _check_db,
    _check_ollama,
    render_sidebar,  # noqa: E402
)
from ui.theme import build_css  # noqa: E402

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
LOGO_PATH = str(ASSETS_DIR / "logo.svg")
LOGO_SVG_MARKUP = (ASSETS_DIR / "logo.svg").read_text(encoding="utf-8")

# Manuals whose PDF has no usable title metadata: sha12 -> display name, so
# the sidebar subtitle never shows the generic file stem "document" (bug
# fix, section 0h). Editable on the Document page (manifest["display_name"]
# takes priority over this fallback).
_KNOWN_MANUAL_NAMES = {"5c1a4a3e34a4": "Lowara"}

st.set_page_config(page_title="MaintiDoc", page_icon=LOGO_PATH, layout="wide")


@st.cache_resource(show_spinner=False)
def _warm_up_embedder() -> bool:
    get_query_embedder()
    return True


@st.cache_resource(show_spinner=False)
def _warm_up_llm(settings: Settings) -> bool:
    """One throwaway call so the first real question doesn't pay for Ollama
    loading the model into VRAM -- best-effort, a failure here must not
    block the app (the real question will surface the error normally)."""
    with contextlib.suppress(Exception):  # best-effort warm-up, never fatal
        get_llm(settings).invoke("Bonjour")
    return True


@st.cache_resource(show_spinner=False)
def _warm_up_clip(settings: Settings) -> bool:
    """Loads MobileCLIP once at startup (populating app.rag.retriever's
    module-level encoder cache -- the UI section 2 performance fix) so the
    first real "Recherche par image" search doesn't pay the ~6-7s model
    load on top of the one-time CUDA warm-up."""
    from app.rag.retriever import _get_cached_image_encoder

    with contextlib.suppress(Exception):  # best-effort warm-up, never fatal
        _get_cached_image_encoder(settings)._ensure_loaded()
    return True


def _settings_for_provider(base: Settings, provider: str) -> Settings:
    if provider == base.llm_provider:
        return base
    return Settings(**{**base.model_dump(), "llm_provider": provider})


def _manual_display_name(manifest: dict, sha12: str) -> str:
    """manifest["display_name"] (editable on the Document page) wins; else
    a known mapping for manuals whose PDF has no usable title metadata
    (bug fix, section 0h -- never show the generic file stem "document")."""
    if manifest.get("display_name"):
        return manifest["display_name"]
    if sha12 in _KNOWN_MANUAL_NAMES:
        return _KNOWN_MANUAL_NAMES[sha12]
    return Path(manifest.get("source_file", "manuel.pdf")).stem


def _active_manual_subtitle() -> str:
    processed_root = Path("data/processed")
    if not processed_root.is_dir():
        return "Aucun manuel actif"
    manifests = sorted(processed_root.glob("*/manifest.json"))
    if not manifests:
        return "Aucun manuel actif"
    manifest = json.loads(manifests[-1].read_text(encoding="utf-8"))
    title = _manual_display_name(manifest, manifests[-1].parent.name)
    pages = manifest.get("total_pages", "?")
    return f"Manuel {title} · {pages} pages"


def _render_logo_header() -> None:
    """One single HTML block (logo SVG + wordmark + subtitle), rendered
    ABOVE the page_link navigation -- static markup only, no user/document
    text interpolated unescaped (the subtitle comes from our own manifest
    resolution above, not from raw user input)."""
    import html as _html

    subtitle = _html.escape(_active_manual_subtitle())
    st.markdown(
        '<div style="display:flex;align-items:center;gap:12px;padding:0 8px 8px;">'
        f'<span style="flex:none;">{LOGO_SVG_MARKUP}</span>'
        '<span style="display:flex;flex-direction:column;gap:2px;min-width:0;">'
        '<span style="font-size:18px;font-weight:700;letter-spacing:-0.01em;">'
        'Mainti<span style="color:#0B5C7A;">Doc</span></span>'
        f'<span style="font-size:13px;color:#5B6675;">{subtitle}</span>'
        "</span></div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.markdown(f"<style>{build_css()}</style>", unsafe_allow_html=True)

    pages = [
        st.Page(
            "app_pages/assistant.py",
            title="Assistant",
            icon=":material/forum:",
            default=True,
        ),
        st.Page(
            "app_pages/image_search.py",
            title="Recherche par image",
            icon=":material/photo_camera:",
        ),
        st.Page("app_pages/document.py", title="Document", icon=":material/description:"),
    ]
    # position="hidden": Streamlit draws no nav UI of its own -- the sidebar
    # below is built entirely by us (logo+wordmark ABOVE the nav links,
    # matching the reference; st.navigation(position="sidebar") instead put
    # Streamlit's own nav widget at a fixed spot that pushed our subtitle
    # below the links -- see module docstring, section 0h).
    page = st.navigation(pages, position="hidden")

    with st.sidebar:
        _render_logo_header()
        for p in pages:
            st.page_link(p)

    settings = get_settings()
    sidebar_state = render_sidebar(settings, show_new_conversation=(page.title == "Assistant"))
    settings = _settings_for_provider(settings, sidebar_state.provider)
    settings.rag_k = sidebar_state.k
    settings.rag_query_expansion = sidebar_state.query_expansion
    st.session_state["rag_mode"] = sidebar_state.search_mode

    # Page shell (sidebar, nav, header) is already drawn above -- warm-up
    # runs now, visibly, instead of blocking before anything renders.
    already_warm = st.session_state.get("_warm_up_done", False)
    if not already_warm:
        db_ok = _check_db(settings)
        ollama_ok = settings.llm_provider != "ollama" or _check_ollama(settings)
        if not db_ok or not ollama_ok:
            fix = (
                "docker compose up -d db"
                if not db_ok
                else "ollama serve (puis vérifiez `ollama list`)"
            )
            render_error_card(
                f"Service indisponible (vérifié avec un délai de {CHECK_TIMEOUT_SECONDS} s) : "
                f"{'base de données' if not db_ok else 'Ollama'}. Commande de dépannage : {fix}",
                retry_key="startup_retry",
            )
        else:
            with st.status(
                "Chargement des modèles… (première ouverture : ~30 s)", expanded=False
            ) as status:
                st.write("Base de données : connectée")
                st.write("Modèle d'embeddings (BGE-M3)…")
                _warm_up_embedder()
                st.write(":material/check_circle: Modèle d'embeddings prêt")
                st.write("Modèle de langage…")
                _warm_up_llm(settings)
                st.write(":material/check_circle: Modèle de langage prêt")
                st.write("Modèle de recherche par image (CLIP)…")
                _warm_up_clip(settings)
                st.write(":material/check_circle: Modèle CLIP prêt")
                status.update(label="Modèles prêts", state="complete")
        st.session_state["_warm_up_done"] = True

    st.session_state["_resolved_settings"] = settings
    page.run()


main()
