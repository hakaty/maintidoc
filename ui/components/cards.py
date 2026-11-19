"""Reusable card/chip renderers for the "Technicien" design system.

Safety rule: unsafe_allow_html is used ONLY for static, hardcoded wrapper
markup (div open/close tags, icons) that never contains interpolated model
output, document text, or user-supplied text. Any such text is always
rendered via a plain st.markdown/st.text/st.caption call (auto-escaped),
nested between the wrapper tags -- never formatted into the HTML string
itself. This is also the fix for the empty-yellow-pill bug (section 0c): the
old code opened a `.atelier-badge`/`.atelier-verified-badge` div, wrote the
label as a SEPARATE st.markdown call right after it, then closed the div --
the label ended up as a sibling of the (now genuinely empty) pill, not
inside it. Every chip below is rendered as ONE escaped string with the icon
+ label already inside it, passed to a single st.markdown call.
"""

import html
import re

import streamlit as st

from app.config import get_settings
from app.rag.retriever import get_full_table_text
from app.rag.service import CITATION_PATTERN, RAGAnswer
from ui.format import fr_seconds
from ui.icons import (
    ICON_CHECK_CIRCLE,
    ICON_ERROR_CIRCLE,
    ICON_INFO_CIRCLE,
    ICON_SEARCH,
    ICON_WARNING_TRIANGLE,
    SOURCE_ROW_ICONS,
)
from ui.pdf_preview import get_page_thumbnail_png
from ui.theme import MATERIAL_ICONS

# Short, demo-friendly model labels -- the full model id (e.g.
# "qwen2.5:7b-instruct-q3_K_M") is kept as a native browser tooltip via the
# HTML `title` attribute (section 6.4 of the UI feedback), never dropped.
_SHORT_MODEL_LABELS = {
    "qwen2.5:7b-instruct": "Qwen 7B",
}


def _short_model_label(model: str) -> str:
    for prefix, label in _SHORT_MODEL_LABELS.items():
        if model.startswith(prefix):
            return label
    return model.split(":")[0] if model else model


def render_verified_chip(n_sources: int) -> None:
    label = f"Réponse vérifiée · {n_sources} source{'s' if n_sources != 1 else ''}"
    st.markdown(
        f'<span class="md-chip-success md-fade-in">{ICON_CHECK_CIRCLE} {label}</span>',
        unsafe_allow_html=True,
    )


def render_no_source_chip() -> None:
    st.markdown(
        f'<span class="md-chip-neutral">{ICON_INFO_CIRCLE} Pas de source fiable</span>',
        unsafe_allow_html=True,
    )


def _badge_label(content_type: str | None) -> tuple[str, str]:
    labels = {
        "text": "passage",
        "table": "tableau",
        "table_row": "tableau",
        "image_description": "figure",
    }
    icon = SOURCE_ROW_ICONS.get(content_type or "text", SOURCE_ROW_ICONS["text"])
    return labels.get(content_type or "text", "passage"), icon


_VIEW_LABELS = {
    "table": "Voir le tableau",
    "table_row": "Voir le tableau",
    "image_description": "Voir la figure",
    "text": "Voir le passage",
}


def _split_table_line(line: str) -> list[str]:
    """One row of a pipe-delimited markdown table -> unescaped cell strings.
    M7.4b fix: NEVER `.strip("|")` or a naive `.split("|")` -- a row whose
    FIRST cell is empty (e.g. "||Risque...|") would have `.strip("|")`
    collapse that cell's own delimiter too, shifting every column left by
    one; and a cell containing an escaped "\\|" (an M7.4 pictogram
    placeholder can legitimately contain one) would be wrongly split in two
    by a plain `.split("|")`. Tolerates rows with or without the outer
    pipes: the standalone table_row chunk's own row line has none, the
    whole-table markdown's rows do."""
    stripped = line.strip()
    inner = stripped[1:-1] if stripped.startswith("|") and stripped.endswith("|") else stripped
    cells = re.split(r"(?<!\\)\|", inner)
    return [cell.strip().replace("\\|", "|") for cell in cells]


def _parse_markdown_table(markdown: str) -> list[list[str]] | None:
    """Header + data rows (separator row dropped), each a list of unescaped
    cell strings -- or None if `markdown` isn't a well-formed table, so the
    caller falls back to plain text instead of guessing at a broken parse."""
    lines = [line for line in markdown.splitlines() if line.strip().startswith("|")]
    if len(lines) < 3:
        return None
    rows = [_split_table_line(line) for line in lines]
    header, separator, *data_rows = rows
    if not data_rows or not all(cell.strip("-: ") == "" for cell in separator):
        return None
    return [header, *data_rows]


def _fill_empty_columns_with_pictogram_placeholder(rows: list[list[str]]) -> list[list[str]]:
    """Section 6.6 fix: a column whose cells are ALL empty (e.g. a "Symbole"
    column made of pictogram IMAGES the table extractor couldn't capture as
    text) renders as a blank, confusing gap -- show a placeholder instead."""
    header, *data_rows = rows
    if not data_rows:
        return rows
    n_cols = len(header)
    empty_cols = {
        col
        for col in range(n_cols)
        if all(not (row[col] if col < len(row) else "") for row in data_rows)
    }
    if not empty_cols:
        return rows
    placeholder = "(pictogramme)"
    new_data_rows = [
        [placeholder if col in empty_cols else cell for col, cell in enumerate(row)]
        for row in data_rows
    ]
    return [header, *new_data_rows]


def _render_table_html(rows: list[list[str]], highlight_row: list[str] | None) -> None:
    """Render parsed table rows as a real HTML table (never by re-splitting
    a markdown string a second time) so the cited row can get its own
    background -- `st.markdown` of raw markdown has no per-row styling hook.
    Every cell is model/document-derived (untrusted), hence escaped."""
    header, *data_rows = rows
    parts = ['<table style="width:100%;border-collapse:collapse;font-size:14px;">', "<thead><tr>"]
    for cell in header:
        parts.append(
            '<th style="text-align:left;padding:6px 8px;border-bottom:2px solid #D7DEE7;">'
            f"{html.escape(cell)}</th>"
        )
    parts.append("</tr></thead><tbody>")
    for row in data_rows:
        row_style = "background-color:#E5F0F4;" if row == highlight_row else ""
        parts.append(f'<tr style="{row_style}">')
        for cell in row:
            parts.append(
                '<td style="padding:6px 8px;border-bottom:1px solid #EEF1F4;">'
                f"{html.escape(cell)}</td>"
            )
        parts.append("</tr>")
    parts.append("</tbody></table>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def _render_table_preview(
    content_type: str | None,
    page: int | None,
    full_text: str | None,
    snippet: str,
    table_id: str | None,
) -> None:
    """"Voir le tableau": always the WHOLE table (M7.4b section 3), with the
    cited row highlighted when the citation pinned down one specific row."""
    full_table_text = get_full_table_text(table_id, get_settings()) if table_id else None
    source_text = full_table_text or full_text or snippet
    rows = _parse_markdown_table(source_text)
    if rows is None:
        st.text(full_text or snippet)
        return
    rows = _fill_empty_columns_with_pictogram_placeholder(rows)
    highlight_row = None
    if content_type == "table_row" and (full_text or snippet):
        row_line = (full_text or snippet).strip().splitlines()[-1]
        highlight_row = _split_table_line(row_line)
    _render_table_html(rows, highlight_row)


def render_source_row(
    page: int | None,
    content_type: str | None,
    snippet: str,
    full_text: str | None = None,
    image_path: str | None = None,
    pdf_path: str | None = None,
    table_id: str | None = None,
) -> None:
    """One compact source row (section 6.3 fix): icon + mono "p. N" chip +
    type label, all on ONE line via a single HTML block -- not split across
    a wide/narrow st.columns pair, which pushed the icon and page number far
    apart. The snippet/"Voir ..." expander (native widgets) follow below."""
    label, icon_svg = _badge_label(content_type)
    # Section 7.4 fix: "p. N" and the type label use the SAME size (14px)
    # and weight -- "p. N" was previously 16px/600 (bold), visually
    # mismatched against the 14px/400 type label right next to it.
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:8px;">{icon_svg}'
        f'<span class="md-mono" style="font-size:14px;">p.&nbsp;{page}</span>'
        f'<span style="font-size:14px;color:#334155;">{html.escape(label.capitalize())}</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.caption(snippet[:160])
    view_label = _VIEW_LABELS.get(content_type or "text", "Voir le passage")
    material_icon = MATERIAL_ICONS.get(content_type or "text", "description")
    with st.expander(view_label, icon=f":material/{material_icon}:"):
        if image_path:
            st.image(image_path, width="stretch", alt=f"Figure page {page}")
        if content_type in ("table", "table_row"):
            if pdf_path and page is not None:
                thumb_cols = st.columns([3, 1])
                with thumb_cols[0]:
                    _render_table_preview(content_type, page, full_text, snippet, table_id)
                with thumb_cols[1]:
                    thumbnail = get_page_thumbnail_png(pdf_path, page)
                    if thumbnail:
                        st.caption("Voir la page")
                        st.image(thumbnail, width="stretch")
            else:
                _render_table_preview(content_type, page, full_text, snippet, table_id)
        else:
            st.text(full_text or snippet)


def render_refusal_card(answer_text: str) -> None:
    """One single HTML block (bug fix, section 0e): the previous version
    opened a `<div>` in one st.markdown call, wrote the title/body in
    separate calls, then closed the div in a fourth call. Each st.markdown
    call is wrapped in its OWN element by Streamlit, so the opening/closing
    tags never actually nested around the content -- the card rendered
    empty, with its text appearing as plain siblings below/outside it. All
    dynamic text (answer_text is model/document-derived, hence untrusted)
    is escaped before being interpolated into the single HTML string."""
    st.markdown(
        f'<div class="md-neutral-card" style="display:flex;gap:12px;">{ICON_SEARCH}'
        '<span style="display:flex;flex-direction:column;gap:2px;">'
        '<span style="font-size:14px;font-weight:700;">Le manuel ne répond pas à cette '
        "question</span>"
        f'<span style="font-size:14px;line-height:21px;color:#334155;">'
        f"{html.escape(answer_text)}</span>"
        '<span style="font-size:13px;color:#5B6675;">Essayez de reformuler votre question, '
        "ou consultez la page Document pour vérifier que le bon manuel est actif.</span>"
        "</span></div>",
        unsafe_allow_html=True,
    )


def render_safety_card(safety_notice: str) -> None:
    """One single HTML block -- see render_refusal_card docstring. safety_notice
    is extracted from the LLM's answer text (untrusted), hence escaped."""
    st.markdown(
        f'<div class="md-safety-card" role="note" style="display:flex;gap:12px;">'
        f'{ICON_WARNING_TRIANGLE.format(stroke="#7A4100")}'
        '<span style="display:flex;flex-direction:column;gap:2px;">'
        '<span style="font-size:14px;font-weight:700;color:#7A4100;">Sécurité</span>'
        f'<span style="font-size:14px;line-height:21px;color:#5C3300;">'
        f"{html.escape(safety_notice)}</span></span></div>",
        unsafe_allow_html=True,
    )


def render_error_card(message: str, retry_key: str = "retry_error") -> bool:
    """One single HTML block -- see render_refusal_card docstring. Returns
    True if the user clicked "Réessayer" (a real st.button placed right
    after the card: raw HTML can't host an interactive, stateful control)."""
    st.markdown(
        f'<div class="md-error-card" role="alert" style="display:flex;gap:12px;">'
        f"{ICON_ERROR_CIRCLE}"
        '<span style="display:flex;flex-direction:column;gap:2px;">'
        '<span style="font-size:14px;font-weight:700;color:#912018;">Une erreur est '
        "survenue</span>"
        f'<span style="font-size:14px;color:#334155;">{html.escape(message)}</span>'
        "</span></div>",
        unsafe_allow_html=True,
    )
    return st.button(
        "Réessayer", icon=":material/refresh:", key=retry_key, width="content", type="secondary"
    )


def render_timings_caption(result: RAGAnswer) -> None:
    """Section 6.4/8.3 fix: ONE 12px muted line -- the previous version mixed
    mono (JetBrains Mono) timings with the full raw model id
    ("qwen2.5:7b-instruct-q3_K_M · ollama") and no per-segment nowrap, so it
    still wrapped onto 2 lines ("Qwen / 7B"). Per the explicit spec: IBM Plex
    Sans (the body font, no override needed) with `font-variant-numeric:
    tabular-nums` instead of switching font family, and `white-space:nowrap`
    on each segment so a segment never breaks mid-text -- only the GAP
    between segments can wrap. The short model label keeps the line short;
    the full id is still available as a native browser tooltip."""
    retrieval = fr_seconds(result.timings.get("retrieval_ms"))
    generation = fr_seconds(result.timings.get("generation_ms"))
    nowrap_nums = 'style="white-space:nowrap;font-variant-numeric:tabular-nums;"'
    parts = []
    if retrieval is not None:
        parts.append(f'<span {nowrap_nums}>Recherche {retrieval}</span>')
    if generation is not None:
        parts.append(f'<span {nowrap_nums}>Rédaction {generation}</span>')
    short_model = html.escape(_short_model_label(result.model))
    provider_label = {"ollama": "Ollama", "openrouter": "OpenRouter"}.get(
        result.provider, result.provider
    )
    full_model_title = html.escape(f"{result.model} · {result.provider}")
    parts.append(
        f'<span style="white-space:nowrap;" title="{full_model_title}">'
        f"{short_model} · {provider_label}</span>"
    )
    st.markdown(
        '<span style="font-size:12px;color:#5B6675;">' + " · ".join(parts) + "</span>",
        unsafe_allow_html=True,
    )


def render_feedback_buttons(key_prefix: str) -> None:
    """👍/👎 as icon buttons with aria-labels; feedback stored in session only."""
    cols = st.columns([1, 1, 10])
    with cols[0]:
        if st.button(
            "",
            icon=":material/thumb_up:",
            key=f"{key_prefix}_up",
            help="Réponse utile",
            width="content",
            type="tertiary",
        ):
            st.session_state[f"{key_prefix}_feedback"] = "up"
    with cols[1]:
        if st.button(
            "",
            icon=":material/thumb_down:",
            key=f"{key_prefix}_down",
            help="Réponse pas utile",
            width="content",
            type="tertiary",
        ):
            st.session_state[f"{key_prefix}_feedback"] = "down"


def _inline_citations_as_chips_html(answer_text: str) -> str:
    """Section 6.5 fix: "[p. 5, tableau]" showed as raw bracket text inside
    the answer. DISPLAY ONLY -- this builds an HTML string for rendering;
    `result.answer` itself (what the citation checker in app.rag.service
    validated) is never modified. The answer is escaped FIRST, then the
    citation pattern is substituted with a chip -- CITATION_PATTERN's chars
    ([, ], ., digits, comma, "figure"/"tableau") are untouched by
    html.escape, so matching after escaping is safe."""
    escaped = html.escape(answer_text).replace("\n", "<br>")

    def _replace(match) -> str:
        # Section 7.2 fix: "p. N" only -- the type hint (", tableau") is
        # dropped here (it's still shown in the Source(s) list below,
        # avoiding a 3rd repetition of the same information).
        page = match.group(1)
        return f'<span class="md-chip">p.&nbsp;{page}</span>'

    return CITATION_PATTERN.sub(_replace, escaped)


def render_answer_text(result: RAGAnswer) -> None:
    """The answer text itself (or the error/refusal card in its place)."""
    if result.error:
        render_error_card(
            f"{result.error}. Vérifiez que Ollama/OpenRouter et la base de données sont "
            "bien disponibles (voir docker compose up -d db && ollama serve), puis réessayez."
        )
    elif result.refused:
        render_refusal_card(result.answer)
    else:
        st.markdown(
            f'<div style="font-size:16px;line-height:26px;max-width:680px;">'
            f"{_inline_citations_as_chips_html(result.answer)}</div>",
            unsafe_allow_html=True,
        )


def render_user_bubble(question: str) -> None:
    """Section 6.2 fix: right-aligned petrol bubble, no avatar -- the
    reference style, replacing the previous full-width grey st.chat_message
    row. `question` is raw user input, hence escaped."""
    st.markdown(
        '<div style="display:flex;justify-content:flex-end;">'
        '<span class="md-user-bubble" style="max-width:70%;display:inline-block;">'
        f"{html.escape(question)}</span></div>",
        unsafe_allow_html=True,
    )


TOOL_LABELS = {
    "search_manual": "Recherche dans le texte",
    "search_tables": "Recherche dans les tableaux",
    "search_figures": "Recherche dans les figures",
    "get_page": "Lecture d'une page",
    "similar_figures": "Figures similaires à la photo",
    "condense_question": "Question reformulée",
}


def render_trace(result: RAGAnswer) -> None:
    """Agent trace: shown whenever at least one tool call happened (success,
    refusal after re-search, or fallback) -- never raw chain-of-thought,
    only tool name/args/result count/duration."""
    if not result.tool_calls:
        return
    with st.expander(
        f"Recherches effectuées ({len(result.tool_calls)})", icon=":material/travel_explore:"
    ):
        for call in result.tool_calls:
            label = TOOL_LABELS.get(call["tool"], call["tool"])
            query = call.get("args", {}).get("query") or call.get("args", {}).get("page", "")
            st.caption(
                f"{label} · « {query} » · {call['n_results']} résultat(s) · {call['seconds']}s"
            )


def render_answer_meta(result: RAGAnswer, key_prefix: str = "answer") -> None:
    """Safety notice, verified chip, source rows, timings/feedback footer --
    never shown for a refusal or an error."""
    if result.error or result.refused:
        return
    if result.safety_notice:
        render_safety_card(result.safety_notice)
    if result.citations:
        render_verified_chip(len(result.citations))
    else:
        render_no_source_chip()
    # Section 7.2 fix: the page used to be cited 3 times (inline chip in
    # the answer text, this standalone "p. N · type" chip row, and the
    # Source(s) list below) -- removed; the inline chips + Source(s) list
    # are enough.
    st.markdown("**Source(s)**")
    pdf_path = str(get_settings().pdf_path)
    for citation in result.citations:
        render_source_row(
            citation.page,
            citation.content_type,
            citation.snippet,
            full_text=citation.full_text or citation.snippet,
            image_path=citation.image_path,
            pdf_path=pdf_path,
            table_id=citation.table_id,
        )
    footer = st.columns([3, 1, 1])
    with footer[0]:
        render_timings_caption(result)
    with footer[1]:
        st.button(
            "Copier", icon=":material/content_copy:", key=f"{key_prefix}_copy", type="tertiary"
        )
    with footer[2]:
        render_feedback_buttons(key_prefix)


def render_answer(result: RAGAnswer, key_prefix: str = "answer") -> None:
    """Full render (text + metadata): used whenever the text hasn't already
    been streamed/rendered earlier in this script run."""
    render_answer_text(result)
    render_answer_meta(result, key_prefix=key_prefix)


def _set_pending_suggestion(question: str) -> None:
    # Section 7.1 fix: an on_click callback runs BEFORE the script body
    # re-executes on this rerun, so by the time render_assistant_tab()
    # checks "is there a question this run" (to decide whether to render
    # the empty state at all), a suggestion click is already indistinguish-
    # able from a typed chat_input submission -- no post-hoc placeholder
    # clearing needed, and the suggestions never render on the same run
    # a click resolves a question (st.empty() was tried first but AppTest
    # doesn't traverse into st.empty() placeholders for .markdown lookups,
    # and rendering them this run then clearing is observably a 2-frame
    # flash risk in a real browser too -- the callback avoids both).
    st.session_state["_pending_suggestion"] = question


def render_empty_state(suggestions: dict[str, str], recent: list[str]) -> None:
    """Renders the guided empty state. `suggestions` maps a short pill label
    (shown on the button, with the full question as its tooltip via
    `help=`) to the full question actually sent -- bug fix, section 0j: the
    reference uses short labels, not the full gold question, as the
    visible button text. The clicked question is read by the caller from
    `st.session_state["_pending_suggestion"]` (section 7.1 fix above),
    not from this function's return value."""
    st.markdown("##### Posez une question sur le manuel")
    # Section 7.3 fix: one wrapping horizontal row of pills (reference
    # style), not a vertical stack -- `.st-key-suggestions_row` in
    # ui/theme.py scopes the flex-wrap layout to just this container.
    with st.container(key="suggestions_row"):
        for short_label, full_question in suggestions.items():
            st.button(
                short_label,
                key=f"suggestion_{short_label}",
                icon=":material/chat_bubble_outline:",
                type="secondary",
                help=full_question,
                on_click=_set_pending_suggestion,
                args=(full_question,),
            )
    if recent:
        st.caption("Récentes")
        for question in recent[-3:]:
            # st.badge auto-escapes its label -- user-typed text is never
            # interpolated into an unsafe_allow_html string (see module
            # docstring: that was the cause of the empty-pill bug, section 0c).
            st.badge(question, icon=":material/history:")
