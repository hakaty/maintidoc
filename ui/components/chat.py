"""Assistant page: chat history, streaming answers, source cards, agent
trace, optional photo attachment for similar_figures.

RAG/agent business logic (app.rag.service, app.agent.agent) is untouched by
this UI redesign -- only rendering, widgets and layout changed.
"""

import json
import time
import uuid
from pathlib import Path

import streamlit as st

from app.config import Settings
from app.rag import service
from app.rag.service import RAGAnswer
from ui.components.cards import (
    render_answer_meta,
    render_answer_text,
    render_empty_state,
    render_trace,
    render_user_bubble,
)
from ui.components.uploads import UploadValidationError, strip_exif_and_save, validate_image_bytes

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"
ASSISTANT_AVATAR = str(ASSETS_DIR / "logo.svg")

# Verified (real stack, fresh-session) to get a "Réponse vérifiée" badge in
# the default mode -- see tests/test_ui_app.py::test_suggestions_are_verified_in_default_mode.
# Short chip label -> full gold question (sent on click, shown as a tooltip
# via the button's `help=`) -- bug fix, section 0j: the reference uses
# short pill labels ("Pictogramme atmosphère explosive ?"), not the full
# 20-word gold question as a button label.
SUGGESTIONS: dict[str, str] = {
    "Pictogramme atmosphère explosive ?": (
        "Quel pictogramme de sécurité, parmi les symboles complémentaires, signale "
        "un risque d'atmosphère explosive ?"
    ),
    "Classe de résistance du béton ?": (
        "Quelle classe de résistance à la compression le béton doit-il avoir ?"
    ),
    "Moteur monophasé refroidi ?": "Que risque un moteur monophasé après avoir refroidi ?",
}

MODE_OPTIONS = ["Simple", "Agent"]
_LEGACY_MODE_ALIASES = {"Simple (RAG)": "Simple"}
# Written by eval/run_qa.py (or by hand after reviewing a real eval run): the
# mode with the better expected-page-hit rate on the gold set, so the demo
# defaults to whichever mode actually answers more questions correctly on
# THIS document, instead of a hardcoded guess.
DEFAULT_MODE_PATH = Path("eval") / "default_mode.json"
RESULTS_PATHS = {
    "Simple": Path("eval") / "results_simple_after.json",
    "Agent": Path("eval") / "results_agent_after.json",
}

TEMP_UPLOAD_DIR = Path.cwd() / ".tmp_uploads"


def default_mode() -> str:
    try:
        data = json.loads(DEFAULT_MODE_PATH.read_text(encoding="utf-8"))
        mode = data.get("default_mode")
        mode = _LEGACY_MODE_ALIASES.get(mode, mode)
        if mode in MODE_OPTIONS:
            return mode
    except Exception:  # noqa: BLE001 -- missing/invalid file just falls back
        pass
    return "Simple"


def _hit_rate_pct(mode_label: str) -> int | None:
    """Expected-page-hit rate (%) for `mode_label`, read from the latest real
    eval output file -- used only for the mode-selector caption, never to
    change behaviour."""
    path = RESULTS_PATHS.get(mode_label)
    if path is None or not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        rate = data["report"]["overall"]["expected_page_hit_rate"]
    except Exception:  # noqa: BLE001 -- missing/malformed file just hides the caption
        return None
    return round(rate * 100) if rate is not None else None


def get_answer(
    question: str,
    settings: Settings,
    mode: str = "hybrid",
    history: list[tuple[str, str]] | None = None,
) -> RAGAnswer:
    """Simple RAG mode: retrieval + streamed generation + citation check.

    `history` (prior turns of this chat session) is used only to condense a
    follow-up into a standalone question for retrieval (app.rag.condense) --
    the answer is still generated/citation-checked from this call's own
    retrieval alone.

    On success, the answer TEXT is already rendered live by a manual
    accumulate-and-markdown loop (called here, inside the current
    st.chat_message block) -- the caller must only render the metadata
    (render_answer_meta), not the text again. st.write_stream() is
    deliberately avoided: it calls dataframe_util.is_dataframe_like() on the
    stream object, which imports pyarrow purely for a type check -- an
    unnecessary dependency on the hot path for a plain text stream.
    On refusal/error, nothing is rendered here: the caller renders the full
    answer (render_answer_text + render_answer_meta).
    Tests monkeypatch this single function instead of mocking the stream.
    """
    try:
        prepared = service.prepare_answer(question, mode=mode, settings=settings, history=history)
    except Exception as exc:  # noqa: BLE001 -- surfaced as a calm error card, never a traceback
        return RAGAnswer(
            answer="",
            error=f"Impossible d'effectuer la recherche dans le manuel : {exc}",
            model="",
            provider=settings.llm_provider,
        )
    if prepared.refusal is not None:
        return prepared.refusal

    placeholder = st.empty()
    try:
        gen_start = time.monotonic()
        full_text = ""
        with placeholder.container():
            text_slot = st.empty()
            for chunk in service.generate_stream(prepared.prompt, settings):
                full_text += chunk
                text_slot.markdown(full_text)
        generation_ms = (time.monotonic() - gen_start) * 1000
    except Exception as exc:  # noqa: BLE001 -- e.g. Ollama unreachable mid-stream
        return RAGAnswer(
            answer="",
            error=f"Le modèle n'a pas répondu : {exc}",
            model=prepared.model,
            provider=prepared.provider,
            injection_detected=prepared.injection_detected,
        )

    result = service.finalize_answer(full_text, prepared, settings, generation_ms)
    if not result.refused and not result.error:
        # Replace the raw streamed text (which still shows literal
        # "[p. N]" brackets) with the chip-rendered final version --
        # section 6.5 fix. The citation CHECK itself already ran inside
        # finalize_answer and is unaffected: this only changes what's shown.
        placeholder.empty()
        render_answer_text(result)
    return result


def get_agent_answer(
    question: str, settings: Settings, session_id: str, image_path: Path | None = None
) -> RAGAnswer:
    """Agent mode: not streamed (the agent may call several tools before its
    final answer), so the text is shown all at once by the caller."""
    from app.agent.agent import ask as agent_ask

    try:
        return agent_ask(question, image_path=image_path, settings=settings, session_id=session_id)
    except Exception as exc:  # noqa: BLE001 -- surfaced as a calm error card, never a traceback
        return RAGAnswer(
            answer="", error=f"L'agent n'a pas pu répondre : {exc}", provider=settings.llm_provider
        )


def _get_session_id() -> str:
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    return st.session_state["session_id"]


def _save_uploaded_photo(data: bytes) -> Path | None:
    try:
        validate_image_bytes(data)
    except UploadValidationError as exc:
        st.error(str(exc))
        return None
    return strip_exif_and_save(data, TEMP_UPLOAD_DIR)


def _render_history_message(role: str, content, avatar: str) -> None:
    with st.chat_message(role, avatar=avatar):
        content()


def render_assistant_tab(settings: Settings) -> None:
    """Section 6.1 fix: the header (title/description), messages and new
    exchange all render inside the SAME `st.container(key="chat_content")`
    (entered here, not by the caller) -- one shared box, same max-width and
    left edge throughout.

    Section 7.1 fix: `st.chat_input` is called here BEFORE entering that
    container. Streamlit only pins a chat_input to the real bottom of the
    page when it is called while NOT nested inside a `st.container`/
    `st.columns`/etc. -- the previous version called it deep inside the
    content container, so it rendered in-place in the document flow
    (between the suggestions and nothing, since nothing came after it in
    that script run) instead of pinned, and a brand-new exchange for the
    turn just submitted ended up rendered below it rather than above.
    `use_agent` must be known before this chat_input call (to pick the
    file-attach variant), but the mode selector widget itself must render
    INSIDE the content container (for the header-row layout) -- a
    `st.container` object can be entered with `with` more than once in the
    same run, appending content each time, so the header row (with the
    live segmented_control value, captured directly) is drawn first inside
    the container, then chat_input is called unnested, then the container
    is re-entered to append history/suggestions/the new exchange below."""
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    history = st.session_state["chat_history"]
    session_id = _get_session_id()

    chosen_default = default_mode()
    content_container = st.container(key="chat_content")
    with content_container:
        header = st.columns([3, 2], vertical_alignment="top")
        with header[0]:
            st.markdown(
                '<div class="md-h1">Assistant</div>'
                '<div class="md-muted">Posez une question en français, l\'assistant cite '
                "toujours la page du manuel.</div>",
                unsafe_allow_html=True,
            )
        with header[1]:
            selected = st.segmented_control(
                "Mode", MODE_OPTIONS, default=chosen_default, key="assistant_mode"
            )
            pct = _hit_rate_pct(chosen_default)
            caption = f"{chosen_default} : le plus fiable sur ce manuel"
            if pct is not None:
                caption += f" ({pct} % de bonnes pages)"
            st.caption(caption)

    use_agent = (selected or chosen_default) == "Agent"
    uploaded_photo = None
    if use_agent:
        chat_value = st.chat_input(
            "Posez votre question sur le manuel…",
            accept_file=True,
            file_type=["jpg", "jpeg", "png", "webp"],
        )
        question = chat_value.text if chat_value else None
        if chat_value and chat_value.files:
            uploaded_photo = chat_value.files[0]
    else:
        question = st.chat_input("Posez votre question sur le manuel…")

    with content_container:
        for entry in history:
            render_user_bubble(entry["question"])
            # M7.4c fix: every widget rendered per answer (Copier, 👍/👎,
            # "Voir le tableau"/"Voir le passage", the trace expander) must
            # have a key unique PER MESSAGE -- a real crash
            # (StreamlitDuplicateElementKey: key='answer_copy') hit on the
            # 2nd question in a conversation, because render_answer_meta()
            # was always called with its default key_prefix="answer" for
            # EVERY history entry. `message_id` is assigned ONCE per entry
            # (see below) and stored WITH the entry, not derived from the
            # loop index, so keys stay stable even if history is ever
            # filtered/reordered.
            message_id = entry["message_id"]

            def _render_past_answer(result=entry["result"], message_id=message_id) -> None:
                render_answer_text(result, key_prefix=f"msg_{message_id}")
                render_answer_meta(result, key_prefix=f"msg_{message_id}")
                render_trace(result, key_prefix=f"msg_{message_id}")

            _render_history_message("assistant", _render_past_answer, ASSISTANT_AVATAR)

        # Section 7.1 fix: the guided empty state (heading + suggestion
        # pills) shows ONLY when the conversation is empty -- it must never
        # reappear once messages exist. A suggestion click is resolved via
        # an on_click callback (see cards.py::_set_pending_suggestion),
        # which runs BEFORE this script body re-executes on that rerun --
        # so by the time we check `question` here, a suggestion click is
        # already indistinguishable from a typed chat_input submission,
        # and the suggestions never render at all on the run that resolves
        # either one (no post-hoc clearing needed).
        question = question or st.session_state.pop("_pending_suggestion", None)
        if not history and not question:
            render_empty_state(SUGGESTIONS, [])

        if question:
            # M7.4c fix: assigned BEFORE rendering, so the brand-new
            # exchange's widgets use the SAME unique key_prefix the history
            # loop above will use for it on every later rerun.
            message_id = uuid.uuid4().hex
            render_user_bubble(question)
            with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
                if use_agent:
                    image_path = None
                    if uploaded_photo:
                        image_path = _save_uploaded_photo(uploaded_photo.getvalue())
                    # M7.5 section 7.5/7.6 fix: a plain spinner, not
                    # st.status() -- it fully disappears once this `with`
                    # block exits instead of staying on screen with its "…"
                    # label never updated, which looked like a second,
                    # separate trace next to render_trace()'s expander
                    # below (the ONE real trace, "under the answer
                    # footer").
                    with st.spinner("Réflexion de l'agent…"):
                        result = get_agent_answer(question, settings, session_id, image_path)
                    # M7.5 section 7.4 fix: render_answer_text() applies the
                    # same inline "[p. N]" -> chip conversion as Simple mode
                    # (it used to be a bare st.markdown(result.answer) here,
                    # bypassing that conversion entirely for a successful
                    # agent answer).
                    render_answer_text(result, key_prefix=f"msg_{message_id}")
                    render_answer_meta(result, key_prefix=f"msg_{message_id}")
                    render_trace(result, key_prefix=f"msg_{message_id}")
                else:
                    mode = st.session_state.get("rag_mode", "hybrid")
                    history_pairs = [
                        (entry["question"], entry["result"].answer)
                        for entry in history[-3:]
                        if not entry["result"].error
                    ]
                    # Not wrapped in st.status: get_answer() renders the
                    # live streamed answer text itself (st.write_stream) --
                    # nesting that inside a collapsible status would hide
                    # the text once the status auto-collapses. The
                    # streaming text IS the loading indicator here.
                    result = get_answer(question, settings, mode=mode, history=history_pairs)
                    if result.error or result.refused:
                        render_answer_text(result, key_prefix=f"msg_{message_id}")
                    if result.standalone_question and result.standalone_question != question:
                        st.caption(f"Question reformulée : « {result.standalone_question} »")
                    render_answer_meta(result, key_prefix=f"msg_{message_id}")
            history.append({"question": question, "result": result, "message_id": message_id})
