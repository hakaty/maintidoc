from pathlib import Path

import pytest
import streamlit as st
from streamlit.proto.Block_pb2 import Block as BlockProto
from streamlit.testing.v1 import AppTest

from app.config import Settings
from app.rag.service import Citation, RAGAnswer
from ui.components import chat as chat_module
from ui.components import document as document_module
from ui.components import sidebar as sidebar_module

APP_PATH = str(Path(__file__).resolve().parent.parent / "ui" / "streamlit_app.py")
ASSISTANT_PAGE = "app_pages/assistant.py"
IMAGE_SEARCH_PAGE = "app_pages/image_search.py"
DOCUMENT_PAGE = "app_pages/document.py"
APP_TIMEOUT = 30


def _select_mode(at: AppTest, mode: str) -> AppTest:
    at.segmented_control(key="assistant_mode").set_value(mode).run(timeout=APP_TIMEOUT)
    return at


@pytest.fixture(autouse=True)
def _patch_heavy_io(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Mocked tests should never touch the real DB, Ollama, GPU or BGE-M3 --
    except the one test explicitly marked integration, which wants the real
    stack and would break if get_query_embedder returned None."""
    if request.node.get_closest_marker("integration"):
        return
    monkeypatch.setattr("app.indexing.embedder.get_query_embedder", lambda settings=None: None)
    monkeypatch.setattr(sidebar_module, "_check_db", lambda settings: True)
    monkeypatch.setattr(sidebar_module, "_check_ollama", lambda settings: True)
    monkeypatch.setattr(sidebar_module, "_gpu_or_cpu_label", lambda settings: "Mode CPU")


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def _answer(**overrides) -> RAGAnswer:
    base = dict(answer="Réponse test [p. 4].", model="qwen", provider="ollama")
    base.update(overrides)
    return RAGAnswer(**base)


def _fake_get_answer_rendering_text(result: RAGAnswer):
    """Mimics the real get_answer()'s contract: on success (not refused, not
    error), it already renders the answer text itself (st.write_stream in
    production); the caller must not render it again."""

    def _fake(question, settings, mode="hybrid", history=None):
        if not result.error and not result.refused:
            st.markdown(result.answer)
        return result

    return _fake


# --- Section 0: visual bug regressions -------------------------------------


def test_section0a_no_bare_icon_name_avatars_anywhere_in_ui_source() -> None:
    """Regression for the icon-font bug (0a): no chat avatar is a bare
    Material icon string ("face"/"smart_toy") -- avatars are image files."""
    ui_root = Path(__file__).resolve().parent.parent / "ui"
    offenders = []
    for path in ui_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if 'avatar=":material/face:"' in text or 'avatar=":material/smart_toy:"' in text:
            offenders.append(path)
    assert offenders == []


def test_section0a_avatars_are_real_images_not_icon_strings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The assistant turn uses a real st.chat_message (image avatar, never
    an icon-string fallback). The user turn is a plain petrol bubble (no
    avatar at all, no st.chat_message) since the 6.2 fix -- see
    test_section6_2_user_message_is_a_right_aligned_bubble_not_chat_message."""
    monkeypatch.setattr("app.config.get_settings", lambda: _settings())
    result = _answer()
    monkeypatch.setattr(chat_module, "get_answer", _fake_get_answer_rendering_text(result))

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=APP_TIMEOUT)
    _select_mode(at, "Simple")
    at.chat_input[0].set_value("Une question").run(timeout=APP_TIMEOUT)

    assert not at.exception
    assert len(at.chat_message) == 1
    for cm in at.chat_message:
        assert cm.proto.avatar_type == BlockProto.ChatMessage.AvatarType.IMAGE
        assert cm.proto.avatar.startswith("data:image/svg+xml")


def test_section0b_sidebar_css_never_forces_a_colour_on_every_descendant() -> None:
    """Regression for the white-on-white select/button bug (0b): the old
    stylesheet had `section[data-testid="stSidebar"] * { color: ... !important; }`."""
    from ui.theme import build_css

    css = build_css()
    assert "stSidebar" not in css or "!important" not in css


def test_section0c_verified_chip_is_a_single_markdown_element(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for the empty-pill bug (0c): icon+label are one escaped
    string in one st.markdown call, never split across an empty wrapper div
    and a separate sibling markdown element. The icon is a real inline SVG
    (section 0d/0e fix), not a Material shortcode -- shortcodes inside a
    raw HTML block render as literal text, see ui/icons.py."""
    monkeypatch.setattr("app.config.get_settings", lambda: _settings())
    citation = Citation(page=4, content_type="table", node_id="n1", snippet="Niveaux de danger")
    result = _answer(citations=[citation])
    monkeypatch.setattr(chat_module, "get_answer", _fake_get_answer_rendering_text(result))

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=APP_TIMEOUT)
    _select_mode(at, "Simple")
    at.chat_input[0].set_value("Quels sont les niveaux de danger ?").run(timeout=APP_TIMEOUT)

    assert not at.exception
    verified = [m for m in at.markdown if "Réponse vérifiée" in m.value]
    assert len(verified) == 1
    assert "md-chip-success" in verified[0].value
    assert "<svg" in verified[0].value
    assert ":material_" not in verified[0].value
    assert ":material/" not in verified[0].value


def test_section0d_camera_hidden_until_prendre_une_photo_clicked() -> None:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=APP_TIMEOUT)
    at.switch_page(IMAGE_SEARCH_PAGE).run(timeout=APP_TIMEOUT)

    assert not at.exception
    assert len(at.get("camera_input")) == 0

    take_photo = next(b for b in at.button if b.label == "Prendre une photo")
    take_photo.click().run(timeout=APP_TIMEOUT)

    assert not at.exception
    assert len(at.get("camera_input")) == 1


def test_section0e_document_page_reads_latest_eval_json_not_a_static_doc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simple_path = tmp_path / "results_simple_after.json"
    agent_path = tmp_path / "results_agent_after.json"
    simple_path.write_text(
        '{"report": {"overall": {"expected_page_hit_rate": 0.91, "citation_validity_rate": 1.0, '
        '"refusal_accuracy": 0.8, "groundedness_avg": 0.95, "latency_ms_p50": 1234}}}',
        encoding="utf-8",
    )
    agent_path.write_text(
        '{"report": {"overall": {"expected_page_hit_rate": 0.42, "citation_validity_rate": 1.0, '
        '"refusal_accuracy": 0.7, "groundedness_avg": 0.9, "latency_ms_p50": 999}}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        document_module, "RESULTS_PATHS", {"Simple": simple_path, "Agent": agent_path}
    )
    monkeypatch.setattr(document_module, "MULTITURN_PATH", tmp_path / "missing.json")

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=APP_TIMEOUT)
    at.switch_page(DOCUMENT_PAGE).run(timeout=APP_TIMEOUT)

    assert not at.exception
    joined = "\n".join(m.value for m in at.markdown)
    # French decimal formatting (section 0i): "91,0 %", never "91 %"/"0.91".
    assert "91,0 %" in joined
    assert "42,0 %" in joined


def test_section0f_error_card_shown_when_db_unreachable_at_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sidebar_module, "_check_db", lambda settings: False)

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=APP_TIMEOUT)

    assert not at.exception
    joined = "\n".join(m.value for m in at.markdown)
    assert "Une erreur est survenue" in joined
    assert "docker compose up -d db" in joined
    # The page shell (sidebar, navigation) still renders despite the failure.
    assert len(at.sidebar) > 0


def test_section0f_warmup_status_does_not_reappear_on_a_later_rerun() -> None:
    """AppTest only observes the FINAL state of a run (it doesn't reproduce
    progressive/streaming rendering -- see testing.md), so the status block's
    label is already "Modèles prêts" by the time .run() returns. What we can
    verify is that the warm-up status block (and the work it gates) only
    exists on the FIRST run of a session, never on a later rerun."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=APP_TIMEOUT)
    assert any("Modèles prêts" in s.label for s in at.get("status"))
    assert at.session_state["_warm_up_done"] is True

    at.run(timeout=APP_TIMEOUT)  # a plain rerun, same session
    assert len(at.get("status")) == 0


# --- Navigation / pages -----------------------------------------------------


def test_app_loads_without_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.config.get_settings", lambda: _settings())
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=APP_TIMEOUT)
    assert not at.exception


def test_sidebar_navigation_has_three_pages_reachable_by_switch_page() -> None:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=APP_TIMEOUT)
    assert not at.exception
    assert any("Assistant" in m.value for m in at.markdown)

    at.switch_page(IMAGE_SEARCH_PAGE).run(timeout=APP_TIMEOUT)
    assert not at.exception
    assert any("Recherche par image" in m.value for m in at.markdown)

    at.switch_page(DOCUMENT_PAGE).run(timeout=APP_TIMEOUT)
    assert not at.exception
    assert any("Document" in m.value for m in at.markdown)


def test_single_primary_button_per_page_source() -> None:
    """Static check: each page file declares at most one `type="primary"`
    button (the Assistant page's primary action is the native chat_input
    submit control, not a separate button)."""
    import ui.components.chat as chat_src
    import ui.components.document as document_src
    import ui.components.image_search as image_search_src

    for module in (chat_src, image_search_src, document_src):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert source.count('type="primary"') <= 1, module.__file__


# --- Assistant page behaviour (unchanged RAG wiring, new widgets) ----------


def test_empty_state_shows_three_suggestion_buttons(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.config.get_settings", lambda: _settings())
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=APP_TIMEOUT)

    labels = [b.label for b in at.button]
    for suggestion in chat_module.SUGGESTIONS:
        assert suggestion in labels


def test_chat_answer_shows_verified_chip_and_citation_chip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.config.get_settings", lambda: _settings())
    citation = Citation(page=4, content_type="table", node_id="n1", snippet="Niveaux de danger")
    result = _answer(citations=[citation])
    monkeypatch.setattr(chat_module, "get_answer", _fake_get_answer_rendering_text(result))

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=APP_TIMEOUT)
    _select_mode(at, "Simple")
    at.chat_input[0].set_value("Quels sont les niveaux de danger ?").run(timeout=APP_TIMEOUT)

    assert not at.exception
    full_text = " ".join(m.value for m in at.markdown)
    assert "Réponse vérifiée" in full_text
    assert "Réponse test" in full_text
    # &nbsp; (never a literal double space), section 6.3 fix; type shown
    # once in the Source(s) row, not repeated on the inline chip (7.2 fix).
    assert "p.&nbsp;4" in full_text
    assert "Tableau" in full_text


def test_refusal_shows_neutral_card_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.config.get_settings", lambda: _settings())
    result = _answer(answer="Le manuel ne contient pas d'information suffisante.", refused=True)
    monkeypatch.setattr(
        chat_module, "get_answer", lambda question, settings, mode="hybrid", history=None: result
    )

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=APP_TIMEOUT)
    _select_mode(at, "Simple")
    at.chat_input[0].set_value("Question hors sujet").run(timeout=APP_TIMEOUT)

    assert not at.exception
    # html.unescape: the refusal card is one escaped HTML block (section 0e
    # fix) -- the answer text's apostrophe is encoded as `&#x27;` in the raw
    # markup (renders as a normal apostrophe in a real browser).
    import html as _html

    full_text = _html.unescape(" ".join(m.value for m in at.markdown))
    assert "Le manuel ne répond pas à cette question" in full_text
    assert "ne contient pas d'information suffisante" in full_text


def test_safety_notice_rendered_as_amber_card(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.config.get_settings", lambda: _settings())
    result = _answer(
        safety_notice="⚠ Sécurité : risque de blessure [p. 26].",
        citations=[Citation(page=26, content_type="text", node_id="n1", snippet="moteur")],
    )
    monkeypatch.setattr(chat_module, "get_answer", _fake_get_answer_rendering_text(result))

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=APP_TIMEOUT)
    _select_mode(at, "Simple")
    at.chat_input[0].set_value("Question sécurité").run(timeout=APP_TIMEOUT)

    assert not at.exception
    full_text = " ".join(m.value for m in at.markdown)
    assert "md-safety-card" in full_text
    assert "risque de blessure" in full_text


def test_service_error_shows_error_card_with_retry_button(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.config.get_settings", lambda: _settings())

    def _boom(question, settings, mode="hybrid", history=None):
        return _answer(answer="", error="Ollama unreachable: connection refused")

    monkeypatch.setattr(chat_module, "get_answer", _boom)

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=APP_TIMEOUT)
    _select_mode(at, "Simple")
    at.chat_input[0].set_value("Une question").run(timeout=APP_TIMEOUT)

    assert not at.exception
    full_text = " ".join(m.value for m in at.markdown)
    assert "Une erreur est survenue" in full_text
    assert "Traceback" not in full_text
    assert any(b.label == "Réessayer" for b in at.button)


def test_clicking_a_suggestion_asks_the_question(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.config.get_settings", lambda: _settings())
    captured = {}

    def _fake_get_answer(question, settings, mode="hybrid", history=None):
        captured["question"] = question
        return _answer()

    monkeypatch.setattr(chat_module, "get_answer", _fake_get_answer)

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=APP_TIMEOUT)
    _select_mode(at, "Simple")
    short_label = next(iter(chat_module.SUGGESTIONS))
    full_question = chat_module.SUGGESTIONS[short_label]
    at.button(key=f"suggestion_{short_label}").click().run(timeout=APP_TIMEOUT)

    assert not at.exception
    assert captured["question"] == full_question


def test_provider_selector_hides_openrouter_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.config.get_settings", lambda: _settings(openrouter_api_key=""))
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=APP_TIMEOUT)

    provider_select = at.sidebar.selectbox[0]
    assert provider_select.options == ["ollama"]


def test_provider_selector_shows_openrouter_when_key_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.config.get_settings", lambda: _settings(openrouter_api_key="sk-test")
    )
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=APP_TIMEOUT)

    provider_select = at.sidebar.selectbox[0]
    assert provider_select.options == ["ollama", "openrouter"]


def test_new_conversation_button_only_on_assistant_page() -> None:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=APP_TIMEOUT)
    assert any(b.label == "Nouvelle conversation" for b in at.sidebar.button)

    at.switch_page(DOCUMENT_PAGE).run(timeout=APP_TIMEOUT)
    assert not any(b.label == "Nouvelle conversation" for b in at.sidebar.button)


def test_mode_selector_defaults_to_configured_default_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default mode is read from eval/default_mode.json (written by/after
    a real eval run, see eval/run_qa.py) -- whichever mode actually answers
    more gold-set questions correctly on this document, not a hardcoded
    guess."""
    monkeypatch.setattr("app.config.get_settings", lambda: _settings())
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=APP_TIMEOUT)

    mode_control = at.segmented_control(key="assistant_mode")
    assert set(mode_control.options) == {"Agent", "Simple"}
    assert mode_control.value == chat_module.default_mode()


def test_default_mode_reads_json_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    custom_path = tmp_path / "default_mode.json"
    custom_path.write_text('{"default_mode": "Agent"}', encoding="utf-8")
    monkeypatch.setattr(chat_module, "DEFAULT_MODE_PATH", custom_path)

    assert chat_module.default_mode() == "Agent"


def test_default_mode_maps_legacy_label(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    custom_path = tmp_path / "default_mode.json"
    custom_path.write_text('{"default_mode": "Simple (RAG)"}', encoding="utf-8")
    monkeypatch.setattr(chat_module, "DEFAULT_MODE_PATH", custom_path)

    assert chat_module.default_mode() == "Simple"


def test_default_mode_falls_back_when_file_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(chat_module, "DEFAULT_MODE_PATH", tmp_path / "missing.json")

    assert chat_module.default_mode() == "Simple"


def test_trace_expander_shown_for_agent_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.config.get_settings", lambda: _settings())
    result = _answer(
        mode_used="agent",
        citations=[Citation(page=35, content_type="table", node_id="n1", snippet="RCD")],
        tool_calls=[
            {"tool": "search_tables", "args": {"query": "RCD"}, "n_results": 1, "seconds": 0.5}
        ],
    )
    monkeypatch.setattr(chat_module, "get_agent_answer", lambda *a, **k: result)

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=APP_TIMEOUT)
    _select_mode(at, "Agent")
    at.chat_input[0].set_value("Question RCD").run(timeout=APP_TIMEOUT)

    assert not at.exception
    expander_labels = [e.label for e in at.expander]
    assert any("Recherches effectuées" in label for label in expander_labels)


def test_photo_attachment_passes_image_path_to_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """AppTest has no public API to simulate a real chat_input file
    attachment, so this checks the wiring directly: when a photo is
    attached, render_assistant_tab must pass its saved path through to
    get_agent_answer."""
    import io

    from PIL import Image as PILImage

    from app.config import Settings
    from ui.components import chat as chat_module_direct

    buf = io.BytesIO()
    PILImage.new("RGB", (16, 16), color=(10, 20, 30)).save(buf, format="PNG")
    png_bytes = buf.getvalue()

    class _FakeFile:
        def getvalue(self):
            return png_bytes

    class _FakeChatInputValue:
        text = "une question"
        files = [_FakeFile()]

    captured = {}

    def _fake_get_agent_answer(question, settings, session_id, image_path=None):
        captured["image_path"] = image_path
        return _answer()

    monkeypatch.setattr(chat_module_direct, "get_agent_answer", _fake_get_agent_answer)
    # render_assistant_tab() now reads the mode from the segmented_control
    # widget's OWN persisted session_state key (needed so it can resolve
    # use_agent BEFORE calling st.chat_input, which must not be nested
    # inside the content container -- section 7.1) rather than capturing
    # the widget call's return value directly, so seed that key instead of
    # mocking the widget's return value.
    chat_module_direct.st.session_state["assistant_mode"] = "Agent"
    monkeypatch.setattr(chat_module_direct.st, "segmented_control", lambda *a, **k: "Agent")
    monkeypatch.setattr(
        chat_module_direct.st, "chat_input", lambda *a, **k: _FakeChatInputValue()
    )

    chat_module_direct.render_assistant_tab(Settings(_env_file=None))

    assert captured["image_path"] is not None
    assert captured["image_path"].is_file()
    captured["image_path"].unlink()


# --- Real stack integration tests ------------------------------------------


@pytest.mark.integration
def test_real_app_answers_five_demo_questions() -> None:
    """Real DB + real Ollama, through the actual UI script (not mocked):
    the 5 demo questions from docs/05_rapport_rag.md."""
    questions = [
        "Que faut-il faire si le dispositif de protection a courant differentiel "
        "residuel (RCD) s'est declenche et que l'unite ne s'allume pas ?",
        "Quel pictogramme de securite, parmi les symboles complementaires, signale "
        "un risque d'atmosphere explosive ?",
        "Quel est le prix de vente recommande pour la pompe CA 1206/3 ?",
        "Que risque-t-il de se passer avec un moteur monophase equipe d'une "
        "protection thermique a reinitialisation automatique, apres qu'il a refroidi ?",
        "A quelle temperature ambiante l'unite emballee doit-elle etre stockee ?",
    ]

    statuses: dict[str, str] = {}
    for i, question in enumerate(questions, start=1):
        try:
            at = AppTest.from_file(APP_PATH)
            at.run(timeout=60)
        except Exception as exc:  # pragma: no cover - network/model/DB unavailable
            pytest.skip(f"Real stack unavailable: {exc}")

        at.chat_input[0].set_value(question).run(timeout=90)
        assert not at.exception, f"Q{i} raised: {at.exception}"

        joined = "\n".join(m.value for m in at.markdown)
        verified = "Réponse vérifiée" in joined
        refused = "Le manuel ne répond pas à cette question" in joined
        error = "Une erreur est survenue" in joined
        if error:
            status = "error"
        elif refused:
            status = "refused"
        elif verified:
            status = "verified"
        else:
            status = "unknown"
        statuses[f"Q{i}"] = status
        print(f"\n[Q{i}] {question[:70]}... -> {status}")
        assert status != "unknown", f"Q{i}: neither a verified answer nor a refusal was shown"

    print(f"\n[summary] {statuses}")


@pytest.mark.integration
def test_suggestions_are_verified_in_default_mode() -> None:
    """The 3 suggestion chips must each get a "Réponse vérifiée" badge when
    asked in the UI's default mode, fresh session per question (same
    rationale as test_real_app_answers_five_demo_questions)."""
    statuses: dict[str, str] = {}
    for i, question in enumerate(chat_module.SUGGESTIONS.values(), start=1):
        try:
            at = AppTest.from_file(APP_PATH)
            at.run(timeout=60)
        except Exception as exc:  # pragma: no cover - network/model/DB unavailable
            pytest.skip(f"Real stack unavailable: {exc}")

        at.chat_input[0].set_value(question).run(timeout=90)
        assert not at.exception, f"Q{i} raised: {at.exception}"
        joined = "\n".join(m.value for m in at.markdown)
        statuses[question] = "verified" if "Réponse vérifiée" in joined else "not verified"

    print(f"\n[suggestions] {statuses}")
    assert all(status == "verified" for status in statuses.values()), statuses


@pytest.mark.integration
def test_app_still_renders_every_page_when_pyarrow_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """M7.4 follow-up: on this machine, Windows Application Control blocked
    pyarrow's compiled lib.pyd, and st.write_stream()'s internal
    is_dataframe_like() check imported pyarrow just to type-sniff a plain
    text generator -- every streamed answer errored with "DLL load failed".
    chat.py now streams via a manual st.empty()+markdown loop instead (never
    imports pyarrow), and a repo-wide grep confirms no st.dataframe/st.table/
    data_editor/chart call exists anywhere in ui/ (all tables are hand-built
    markdown). This test makes pyarrow import fail on purpose (regardless of
    whether the real OS block is active) and drives all 3 pages -- a Simple
    question, an Agent question, and an image upload -- to prove the app
    never needs it. Inside Docker (Linux, M7.5) pyarrow imports normally;
    this only proves the app does not *require* it."""
    import sys

    for name in list(sys.modules):
        if name == "pyarrow" or name.startswith("pyarrow."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "pyarrow", None)

    try:
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=60)
    except Exception as exc:  # pragma: no cover - network/model/DB unavailable
        pytest.skip(f"Real stack unavailable: {exc}")
    assert not at.exception

    at.segmented_control[0].set_value("Simple").run(timeout=15)
    at.chat_input[0].set_value(
        "Quel pictogramme de securite, parmi les symboles complementaires, "
        "signale un risque d'atmosphere explosive ?"
    ).run(timeout=90)
    assert not at.exception, f"Simple mode raised: {at.exception}"
    joined = "\n".join(m.value for m in at.markdown)
    assert "DLL load failed" not in joined
    assert "Une erreur est survenue" not in joined
    assert "Réponse vérifiée" in joined

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    at.segmented_control[0].set_value("Agent").run(timeout=15)
    at.chat_input[0].set_value(
        "Que risque un moteur monophase apres avoir refroidi ?"
    ).run(timeout=90)
    assert not at.exception, f"Agent mode raised: {at.exception}"
    joined = "\n".join(m.value for m in at.markdown)
    assert "DLL load failed" not in joined
    assert "Une erreur est survenue" not in joined

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    at.switch_page(IMAGE_SEARCH_PAGE).run(timeout=APP_TIMEOUT)
    assert not at.exception
    import io

    from PIL import Image as PILImage

    buf = io.BytesIO()
    PILImage.new("RGB", (64, 64), color=(200, 60, 60)).save(buf, format="PNG")
    at.file_uploader[0].set_value(("query.png", buf.getvalue(), "image/png")).run(timeout=60)
    assert not at.exception, f"Image search raised: {at.exception}"
    joined = "\n".join(m.value for m in at.markdown)
    assert "DLL load failed" not in joined
    assert "Une erreur est survenue" not in joined

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    at.switch_page(DOCUMENT_PAGE).run(timeout=APP_TIMEOUT)
    assert not at.exception
    joined = "\n".join(m.value for m in at.markdown)
    assert "DLL load failed" not in joined
    assert "Une erreur est survenue" not in joined


# --- Pixel-match task: bugs b/c/d/h regression tests -----------------------


def test_section0b_each_page_renders_only_its_own_header() -> None:
    """Bug fix (0b): the entry script must do shared setup + sidebar +
    page.run() only -- no page content of its own -- so switching pages
    never leaves a previous page's header/body behind."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=APP_TIMEOUT)
    joined = "\n".join(m.value for m in at.markdown)
    assert 'class="md-h1">Assistant<' in joined
    assert 'class="md-h1">Document<' not in joined
    assert 'class="md-h1">Recherche par image<' not in joined

    at.switch_page(DOCUMENT_PAGE).run(timeout=APP_TIMEOUT)
    joined = "\n".join(m.value for m in at.markdown)
    assert 'class="md-h1">Document<' in joined
    assert 'class="md-h1">Assistant<' not in joined
    assert 'class="md-h1">Recherche par image<' not in joined

    at.switch_page(IMAGE_SEARCH_PAGE).run(timeout=APP_TIMEOUT)
    joined = "\n".join(m.value for m in at.markdown)
    assert 'class="md-h1">Recherche par image<' in joined
    assert 'class="md-h1">Assistant<' not in joined
    assert 'class="md-h1">Document<' not in joined


def test_section0c_reglages_avances_expander_appears_exactly_once() -> None:
    """Bug fix (0c): "Réglages avancés" must never be duplicated, on any
    page (render_sidebar() is called exactly once per script run)."""
    for page_path in (None, DOCUMENT_PAGE, IMAGE_SEARCH_PAGE):
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=APP_TIMEOUT)
        if page_path is not None:
            at.switch_page(page_path).run(timeout=APP_TIMEOUT)
        labels = [e.label for e in at.sidebar.expander]
        assert labels.count("Réglages avancés") == 1, (page_path, labels)


def test_status_rows_never_contain_a_raw_material_shortcode() -> None:
    """Bug fix (0d): the "État des services" status rows are built as a raw
    HTML block (a coloured dot span, see ui/icons.py::status_dot) -- a
    Material shortcode embedded in that kind of block renders as literal
    text (observed on the real app: ":material_error: Base de données...").
    Native st.write/st.button/st.expander icon= calls elsewhere are a
    separate, legitimate use of the shortcode syntax (substituted by the
    browser, not visible to AppTest) and are out of scope for this check."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=APP_TIMEOUT)
    status_rows = [
        m.value for m in at.sidebar.markdown if "en ligne" in m.value or "GPU" in m.value
    ]
    assert status_rows, "no status rows found"
    joined = "\n".join(status_rows)
    assert ":material_" not in joined
    assert ":material/" not in joined
    assert "<span" in joined  # the coloured dot is real markup, not text


def test_section0h_logo_wordmark_rendered_above_page_links() -> None:
    """Bug fix (0h): the sidebar is built as logo+wordmark block, THEN the
    page_link navigation -- not Streamlit's own st.logo()/st.navigation(
    position="sidebar"), which put the wordmark below the nav links."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=APP_TIMEOUT)
    sidebar_markdown = [m.value for m in at.sidebar.markdown]
    header_index = next(
        i for i, v in enumerate(sidebar_markdown) if "Mainti" in v and "Doc" in v
    )
    assert "<svg" in sidebar_markdown[header_index]
    page_link_labels = [pl.label for pl in at.sidebar.page_link]
    assert page_link_labels == ["Assistant", "Recherche par image", "Document"]


# --- Section 6: real-screenshot feedback regressions ------------------------


def test_section6_1_header_lives_inside_the_shared_chat_content_container() -> None:
    """Bug fix (6.1): the h1/subtitle used to render OUTSIDE the
    `chat_content` container (left-aligned at the main area's edge) while
    everything else (mode selector, messages, chat input) was centered
    inside that container's own 820px-max box -- the two never lined up.
    The header now lives inside render_assistant_tab(), i.e. inside
    whatever container the caller (app_pages/assistant.py) wraps it in."""
    assistant_page_source = Path("ui/app_pages/assistant.py").read_text(encoding="utf-8")
    assert 'class="md-h1"' not in assistant_page_source

    chat_source = Path("ui/components/chat.py").read_text(encoding="utf-8")
    assert 'class="md-h1"' in chat_source


def test_section6_2_user_message_is_a_right_aligned_bubble_not_chat_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bug fix (6.2): a petrol, right-aligned bubble (reference style), not
    a full-width st.chat_message row with an avatar."""
    monkeypatch.setattr("app.config.get_settings", lambda: _settings())
    result = _answer()
    monkeypatch.setattr(chat_module, "get_answer", _fake_get_answer_rendering_text(result))

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=APP_TIMEOUT)
    _select_mode(at, "Simple")
    at.chat_input[0].set_value("Une question <script>").run(timeout=APP_TIMEOUT)

    assert not at.exception
    bubbles = [
        m.value
        for m in at.markdown
        if "md-user-bubble" in m.value and "justify-content:flex-end" in m.value
    ]
    assert len(bubbles) == 1
    assert "Une question &lt;script&gt;" in bubbles[0]  # escaped, never raw HTML


def test_section6_3_source_row_top_line_is_a_single_html_block() -> None:
    """Bug fix (6.3): icon + "p. N" + type label on ONE line via a single
    HTML block -- not split across a wide/narrow st.columns pair, which
    pushed the icon and page number far apart."""
    source = Path("ui/components/cards.py").read_text(encoding="utf-8")
    render_source_row_body = source.split("def render_source_row(")[1].split("\ndef ")[0]
    # Only the code after the docstring (which mentions "st.columns" in
    # prose) up to the snippet caption -- the actual top-line block.
    code_only = render_source_row_body.split('"""')[-1]
    top_line_block = code_only.split("st.caption(snippet")[0]
    assert "st.columns(" not in top_line_block
    assert "display:flex" in top_line_block
    assert "p.&nbsp;{page}" in top_line_block


def test_section6_4_footer_uses_a_short_model_label_with_full_id_as_tooltip() -> None:
    from ui.components.cards import _short_model_label

    assert _short_model_label("qwen2.5:7b-instruct-q3_K_M") == "Qwen 7B"
    source = Path("ui/components/cards.py").read_text(encoding="utf-8")
    assert "title=" in source  # native tooltip holds the full model id


def test_section6_5_inline_citations_render_as_chips_not_raw_brackets() -> None:
    """Bug fix (6.5): "[p. 5, tableau]" showed as raw text inside the
    answer. DISPLAY ONLY -- the underlying answer text used by the citation
    checker is untouched (this is a pure string->string render helper).
    Section 7.2 fix: the inline chip shows "p. N" only (no type hint) --
    the type still appears once, in the Source(s) list below."""
    from ui.components.cards import _inline_citations_as_chips_html

    html_out = _inline_citations_as_chips_html("Une info [p. 5, tableau] et une autre [p. 12].")
    assert "[p. 5, tableau]" not in html_out
    assert "[p. 12]" not in html_out
    assert '<span class="md-chip">p.&nbsp;5</span>' in html_out
    assert '<span class="md-chip">p.&nbsp;12</span>' in html_out
    assert "tableau" not in html_out


def test_section6_5_inline_citation_chips_escape_untrusted_answer_text() -> None:
    from ui.components.cards import _inline_citations_as_chips_html

    html_out = _inline_citations_as_chips_html("<script>alert(1)</script> [p. 1].")
    assert "<script>" not in html_out
    assert "&lt;script&gt;" in html_out


def test_section6_6_entirely_empty_table_column_shows_pictogramme_placeholder() -> None:
    """Bug fix (6.6): a column whose pictogram cells are images (never
    captured as table text) used to render as a confusing blank gap."""
    from ui.components.cards import (
        _fill_empty_columns_with_pictogram_placeholder,
        _parse_markdown_table,
    )

    table_md = (
        "| Symbole | Description |\n"
        "| --- | --- |\n"
        "|  | Risque électrique |\n"
        "|  | Risque mécanique |\n"
    )
    rows = _fill_empty_columns_with_pictogram_placeholder(_parse_markdown_table(table_md))
    assert all(row[0] == "(pictogramme)" for row in rows[1:])
    assert rows[1][1] == "Risque électrique"
    assert rows[2][1] == "Risque mécanique"


def test_section6_6_non_empty_columns_are_left_unchanged() -> None:
    from ui.components.cards import (
        _fill_empty_columns_with_pictogram_placeholder,
        _parse_markdown_table,
    )

    table_md = "| A | B |\n| --- | --- |\n| x | y |\n"
    rows = _parse_markdown_table(table_md)
    assert _fill_empty_columns_with_pictogram_placeholder(rows) == rows


# --- M7.4b section 3: table preview never breaks on a leading-empty cell ---
# --- or an escaped "|" inside a pictogram placeholder -----------------------


def test_split_table_line_never_collapses_a_leading_empty_cell() -> None:
    """A row whose FIRST cell is empty (stored with no padding spaces) must
    not have `.strip("|")`'s double-strip shift every column left by one."""
    from ui.components.cards import _split_table_line

    assert _split_table_line("||Risque d'atmosphère explosive|") == [
        "",
        "Risque d'atmosphère explosive",
    ]


def test_split_table_line_unescapes_a_pipe_inside_a_pictogram_cell() -> None:
    """A naive `.split("|")` would wrongly cut a cell containing an escaped
    "\\|" (an M7.4 pictogram placeholder can legitimately contain one) in
    two -- the column shift bug reported against the real pictogram
    question's table preview."""
    from ui.components.cards import _split_table_line

    line = r"|[pictogramme \| triangle jaune]|Risque d'atmosphère explosive|"
    assert _split_table_line(line) == [
        "[pictogramme | triangle jaune]",
        "Risque d'atmosphère explosive",
    ]


def test_split_table_line_tolerates_a_row_with_no_outer_pipes() -> None:
    """The standalone table_row chunk's own row line has no outer pipes
    (unlike the whole-table markdown's rows) -- both forms must parse to
    the same cells."""
    from ui.components.cards import _split_table_line

    assert _split_table_line("Symbole|Description") == ["Symbole", "Description"]


def test_parse_markdown_table_returns_none_for_non_table_text() -> None:
    from ui.components.cards import _parse_markdown_table

    assert _parse_markdown_table("Juste un passage de texte normal.") is None


def test_parse_markdown_table_returns_none_for_a_bare_table_row_snippet() -> None:
    """The standalone table_row chunk's own 2-line "header\\nrow" text (no
    outer pipes, no separator row) is NOT itself a parseable table -- in
    production _render_table_preview always resolves the full table via
    table_id first, falling back to plain text only if that lookup fails."""
    from ui.components.cards import _parse_markdown_table

    assert _parse_markdown_table("Symbole|Description\n[pictogramme]|Risque") is None


# --- Section 7: new user feedback after section 6 --------------------------


def test_section7_1_chat_input_called_before_entering_the_content_container() -> None:
    """Bug fix (7.1): st.chat_input must be called while NOT nested inside
    any st.container -- that's what makes Streamlit pin it to the real
    bottom of the page instead of rendering it in the document flow. The
    content container is re-entered twice (header row first, history/
    suggestions/new-turn after) with the chat_input call strictly between
    the two "with content_container:" blocks, not inside either."""
    source = Path("ui/components/chat.py").read_text(encoding="utf-8")
    entries = [
        i for i in range(len(source)) if source.startswith("with content_container:", i)
    ]
    assert len(entries) == 2, entries
    chat_input_indices = [
        i for i in range(len(source)) if source.startswith("st.chat_input(", i)
    ]
    assert chat_input_indices, "no st.chat_input call found"
    assert all(entries[0] < i < entries[1] for i in chat_input_indices)


def test_section7_1_suggestions_disappear_once_the_conversation_has_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.config.get_settings", lambda: _settings())
    result = _answer()
    monkeypatch.setattr(chat_module, "get_answer", _fake_get_answer_rendering_text(result))

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=APP_TIMEOUT)
    _select_mode(at, "Simple")

    empty_heading = "Posez une question sur le manuel"
    labels_before = [b.label for b in at.button]
    assert any(s in labels_before for s in chat_module.SUGGESTIONS)
    headings_before = [m.value for m in at.markdown if empty_heading in m.value]
    assert headings_before

    at.chat_input[0].set_value("Une question").run(timeout=APP_TIMEOUT)
    assert not at.exception

    labels_after = [b.label for b in at.button]
    assert not any(s in labels_after for s in chat_module.SUGGESTIONS)
    headings_after = [m.value for m in at.markdown if empty_heading in m.value]
    assert not headings_after


def test_section7_1_suggestions_disappear_on_the_same_run_a_suggestion_is_clicked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: a suggestion-button click (unlike typed chat_input) only
    reveals the clicked question AFTER render_empty_state() has already
    drawn the pills for this run -- they must still disappear on this same
    run, not linger above the new exchange until the next rerun."""
    monkeypatch.setattr("app.config.get_settings", lambda: _settings())
    result = _answer()
    monkeypatch.setattr(chat_module, "get_answer", _fake_get_answer_rendering_text(result))

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=APP_TIMEOUT)
    _select_mode(at, "Simple")

    empty_heading = "Posez une question sur le manuel"
    first_label = next(iter(chat_module.SUGGESTIONS))
    at.button(key=f"suggestion_{first_label}").click().run(timeout=APP_TIMEOUT)
    assert not at.exception

    labels_after = [b.label for b in at.button]
    assert not any(s in labels_after for s in chat_module.SUGGESTIONS)
    headings_after = [m.value for m in at.markdown if empty_heading in m.value]
    assert not headings_after


def test_section7_2_no_standalone_citation_chip_under_verified_badge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bug fix (7.2): the page used to be cited 3 times -- inline chip,
    standalone "p. N · type" chip row, and the Source(s) list. The
    standalone row is removed; inline chip + Source(s) list remain."""
    monkeypatch.setattr("app.config.get_settings", lambda: _settings())
    citation = Citation(page=4, content_type="table", node_id="n1", snippet="Niveaux de danger")
    result = _answer(citations=[citation])
    monkeypatch.setattr(chat_module, "get_answer", _fake_get_answer_rendering_text(result))

    at = AppTest.from_file(APP_PATH)
    at.run(timeout=APP_TIMEOUT)
    _select_mode(at, "Simple")
    at.chat_input[0].set_value("Quels sont les niveaux de danger ?").run(timeout=APP_TIMEOUT)

    assert not at.exception
    standalone_chips = [
        m.value
        for m in at.markdown
        if m.value.strip() == '<span class="md-chip">p.&nbsp;4 · tableau</span>'
    ]
    assert not standalone_chips


def test_section7_3_suggestions_are_in_a_single_row_container() -> None:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=APP_TIMEOUT)
    assert "suggestions_row" in [c.key for c in at.get("container")]


# --- M7.4c: unique widget keys across multiple messages in one conversation


@pytest.mark.integration
def test_three_questions_in_a_row_no_duplicate_key_crash() -> None:
    """Real regression (M7.4c): asking a 2nd question in the same
    conversation crashed with StreamlitDuplicateElementKey('answer_copy')
    -- render_answer_meta() always used its default key_prefix="answer" for
    every history entry. Simple, then Agent, then Simple: each answer must
    render with NO exception and its OWN Copier/👍/👎 buttons (3 of each,
    not deduplicated down to 1)."""
    try:
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=60)
    except Exception as exc:  # pragma: no cover - real stack unavailable
        pytest.skip(f"Real stack unavailable: {exc}")

    # Reusing the 3 SUGGESTIONS questions -- already verified elsewhere
    # (test_suggestions_are_verified_in_default_mode) to reliably answer,
    # so a failure here points at the key-collision bug, not at answer
    # quality/agent variance.
    questions = list(chat_module.SUGGESTIONS.values())

    _select_mode(at, "Simple")
    at.chat_input[0].set_value(questions[0]).run(timeout=90)
    assert not at.exception, f"Q1 (Simple) raised: {at.exception}"

    _select_mode(at, "Agent")
    at.chat_input[0].set_value(questions[1]).run(timeout=90)
    assert not at.exception, f"Q2 (Agent) raised: {at.exception}"

    _select_mode(at, "Simple")
    at.chat_input[0].set_value(questions[2]).run(timeout=90)
    assert not at.exception, f"Q3 (Simple) raised: {at.exception}"

    copy_buttons = [b for b in at.button if b.label == "Copier"]
    assert len(copy_buttons) == 3, f"expected 3 Copier buttons, found {len(copy_buttons)}"


@pytest.mark.integration
def test_suggestion_click_after_a_reset_no_duplicate_key_crash() -> None:
    """A conversation, reset via "Nouvelle conversation", then a suggestion
    click -- must not crash and must render that answer's own buttons."""
    try:
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=60)
    except Exception as exc:  # pragma: no cover - real stack unavailable
        pytest.skip(f"Real stack unavailable: {exc}")

    _select_mode(at, "Simple")
    at.chat_input[0].set_value("Quels sont les niveaux de danger ?").run(timeout=90)
    assert not at.exception

    reset_button = next(b for b in at.sidebar.button if b.label == "Nouvelle conversation")
    reset_button.click().run(timeout=APP_TIMEOUT)
    assert not at.exception

    first_label = next(iter(chat_module.SUGGESTIONS))
    at.button(key=f"suggestion_{first_label}").click().run(timeout=90)
    assert not at.exception, f"Post-reset suggestion click raised: {at.exception}"
    assert any(b.label == "Copier" for b in at.button)
