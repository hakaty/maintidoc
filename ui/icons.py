"""Inline SVG icon snippets, copied path-for-path from the validated design
reference (docs/design/reference/*.html).

Bug fix (section 0d/0e of the UI pixel-match task): Material Symbols icon
shortcodes (":material/xxx:") rendered as literal shortcode text when
embedded inside a block-level `<div>` string passed to
`st.markdown(unsafe_allow_html=True)` -- Streamlit's markdown parser treats
a multi-line HTML block as raw passthrough and does not substitute
shortcodes inside it (that substitution only happens for Streamlit's own
markdown text nodes). The fix used throughout ui/ is to never rely on that
substitution happening inside raw HTML we write ourselves: these functions
return real `<svg>` markup instead, matching the reference exactly, so the
icon renders deterministically regardless of font loading or markdown
quirks. Native Streamlit widgets (st.button icon=..., st.expander icon=...)
still use Material shortcodes, since Streamlit does not accept raw SVG for
those -- see the Known limitations section of the final report.
"""

ICON_CHECK_CIRCLE = (
    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="3" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<path d="M5 12l5 5 9-10"/></svg>'
)

ICON_INFO_CIRCLE = (
    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2.5" stroke-linecap="round" aria-hidden="true">'
    '<circle cx="12" cy="12" r="9"/><path d="M12 8v4M12 16h0"/></svg>'
)

ICON_WARNING_TRIANGLE = (
    '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="{stroke}" '
    'stroke-width="2" stroke-linejoin="round" aria-hidden="true" style="flex: none;">'
    '<path d="M12 3l10 18H2z"/><path d="M12 10v4M12 17h0" stroke-linecap="round"/></svg>'
)

ICON_SEARCH = (
    '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#334155" '
    'stroke-width="2" stroke-linecap="round" aria-hidden="true" style="flex: none;">'
    '<circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/></svg>'
)

ICON_ERROR_CIRCLE = (
    '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#B42318" '
    'stroke-width="2" stroke-linecap="round" aria-hidden="true" style="flex: none;">'
    '<circle cx="12" cy="12" r="9"/><path d="M12 8v5M12 16h0"/></svg>'
)


ICON_PDF = (
    '<svg width="32" height="40" viewBox="0 0 52 64" aria-hidden="true">'
    '<path d="M4 2h30l16 16v42a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2z" '
    'fill="#FFFFFF" stroke="#8A96A3" stroke-width="2"/>'
    '<path d="M34 2v16h16" fill="none" stroke="#8A96A3" stroke-width="2"/>'
    '<rect x="2" y="38" width="36" height="16" rx="3" fill="#B42318"/>'
    '<text x="20" y="50" text-anchor="middle" font-family="IBM Plex Sans, sans-serif" '
    'font-size="11" font-weight="700" fill="#FFFFFF">PDF</text></svg>'
)


ICON_TABLE_SM = (
    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 10h18M9 4v16"/></svg>'
)

ICON_FIGURE_SM = (
    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="9" cy="10" r="2"/>'
    '<path d="M21 16l-5-5-9 9"/></svg>'
)

ICON_TEXT_SM = (
    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<path d="M6 3h9l4 4v14H6z"/><path d="M14 3v5h5M9 13h6M9 17h4"/></svg>'
)

SOURCE_ROW_ICONS = {
    "table": ICON_TABLE_SM,
    "table_row": ICON_TABLE_SM,
    "image_description": ICON_FIGURE_SM,
    "text": ICON_TEXT_SM,
}


def status_dot(color: str) -> str:
    """One <span class="dot"> exactly like the reference's "État des
    services" rows -- a plain coloured dot, never a Material icon shortcode
    (see module docstring)."""
    return (
        f'<span style="width:8px;height:8px;border-radius:50%;display:inline-block;'
        f'flex:none;background:{color};" aria-hidden="true"></span>'
    )
