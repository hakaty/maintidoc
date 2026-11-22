import tomllib

from ui.theme import (
    COLORS,
    FONT_SIZES_PX,
    FONT_WEIGHTS,
    SPACING_PX,
    build_config_toml,
    build_css,
    contrast_ratio,
)

ALLOWED_SIZES = {12, 14, 16, 18, 24, 32}
ALLOWED_WEIGHTS = {400, 500, 600, 700}
ALLOWED_SPACING = {4, 8, 12, 16, 24, 32, 40}


def test_font_sizes_are_only_the_allowed_scale() -> None:
    assert set(FONT_SIZES_PX.values()) <= ALLOWED_SIZES
    assert set(FONT_SIZES_PX.values()) == ALLOWED_SIZES  # all 6 actually used


def test_font_weights_are_only_the_allowed_set() -> None:
    assert set(FONT_WEIGHTS.values()) == ALLOWED_WEIGHTS


def test_spacing_scale_is_multiples_of_four_from_the_allowed_set() -> None:
    assert set(SPACING_PX) == ALLOWED_SPACING
    assert all(value % 4 == 0 for value in SPACING_PX)


def test_primary_color_in_generated_config_is_petrol() -> None:
    config = build_config_toml()
    assert f'primaryColor = "{COLORS["primary"]}"' in config
    assert COLORS["primary"] == "#0B5C7A"


def test_generated_config_never_uses_streamlit_default_red() -> None:
    config = build_config_toml()
    assert "#FF4B4B" not in config


def test_css_never_hardcodes_default_red_gradient_or_universal_selector() -> None:
    css = build_css()
    assert "#FF4B4B" not in css
    assert "linear-gradient" not in css
    assert "radial-gradient" not in css
    assert "purple" not in css.lower()
    # No universal selector anywhere, in or outside a media query (section 0a
    # root cause: a wildcard font-family rule hit Material icon spans).
    import re

    assert not re.search(r"(^|[\s{};])\*(\s*\{|\s*,)", css)


def test_css_does_not_override_icon_font_globally() -> None:
    """Regression test for the icon-font bug (section 0a): the stylesheet
    must never set font-family on html/body/a class wildcard that would also
    match Streamlit's Material Symbols icon span."""
    css = build_css()
    assert "html, body" not in css
    assert '[class*="st-"]' not in css


def test_segmented_control_checked_option_always_highlighted() -> None:
    """M7.5 section 7.1 fix: the Simple/Agent mode selector was reported to
    sometimes look unselected -- force an unmistakable highlight directly
    off the DOM's own aria-checked state, independent of Streamlit's
    default styling/rendering timing."""
    css = build_css()
    assert '[data-testid="stSegmentedControl"] [role="radio"][aria-checked="true"]' in css
    assert COLORS["primary"] in css


def test_body_text_contrast_meets_wcag_aa_normal_text() -> None:
    assert contrast_ratio(COLORS["ink"], COLORS["page_bg"]) >= 4.5
    assert contrast_ratio(COLORS["ink"], COLORS["surface"]) >= 4.5
    assert contrast_ratio(COLORS["text_2"], COLORS["surface"]) >= 4.5
    assert contrast_ratio(COLORS["muted"], COLORS["surface"]) >= 4.5


def test_control_border_meets_wcag_aa_non_text_contrast() -> None:
    # Non-text (control border) contrast >= 3:1 on white, never lighter.
    assert contrast_ratio(COLORS["control_border"], COLORS["surface"]) >= 3.0


def test_primary_on_white_button_text_meets_wcag_aa() -> None:
    assert contrast_ratio(COLORS["surface"], COLORS["primary"]) >= 4.5


def test_safety_text_on_safety_bg_meets_wcag_aa() -> None:
    assert contrast_ratio(COLORS["safety_text"], COLORS["safety_bg"]) >= 4.5


def test_success_and_error_text_meet_wcag_aa_on_their_backgrounds() -> None:
    assert contrast_ratio(COLORS["success"], COLORS["success_bg"]) >= 4.5
    assert contrast_ratio(COLORS["error"], COLORS["error_bg"]) >= 4.5


def test_only_one_primary_hue_used_for_fills() -> None:
    primary_like = {k: v for k, v in COLORS.items() if k.startswith("primary")}
    assert primary_like["primary"] == "#0B5C7A"


def test_no_purple_anywhere_in_the_palette() -> None:
    assert all("purple" not in str(v).lower() for v in COLORS.values())


def test_config_declares_light_base_even_when_os_prefers_dark() -> None:
    config = build_config_toml()
    assert 'base = "light"' in config


def test_config_toml_parses_with_tomllib() -> None:
    parsed = tomllib.loads(build_config_toml())
    assert parsed["theme"]["base"] == "light"
    assert parsed["theme"]["primaryColor"] == COLORS["primary"]


def test_font_and_code_font_each_load_with_streamlits_own_parser() -> None:
    """Regression test (section 0a, round 2): the previous config.toml had
    `font` and `codeFont` pointing at the SAME Google Fonts URL, which
    requested two families in one query string
    (`family=IBM+Plex+Sans...&family=JetBrains+Mono...`). Streamlit's own
    theme parser rejects any font/codeFont source URL with more than one
    `family=` occurrence and raises StreamlitInvalidThemeOptionError while
    the server starts -- which silently dropped the whole [theme] section,
    not just the font, and made the app fall back to Streamlit's default
    (dark-if-OS-prefers-dark, red #FF4B4B) theme. Feeding the generated
    config through Streamlit's real parser here means a regression fails a
    test instead of failing silently at runtime again."""
    from streamlit.runtime.theme_util import _parse_font_config

    parsed = tomllib.loads(build_config_toml())
    theme = parsed["theme"]

    for property_name in ("font", "codeFont"):
        font_name, source_url = _parse_font_config(theme[property_name], property_name)
        assert font_name
        assert source_url is not None
        assert source_url.count("family=") == 1


def test_font_and_code_font_source_urls_are_distinct_single_family_urls() -> None:
    from ui.theme import CODE_FONT_SOURCE_URL, FONT_SOURCE_URL

    assert FONT_SOURCE_URL != CODE_FONT_SOURCE_URL
    assert FONT_SOURCE_URL.count("family=") == 1
    assert CODE_FONT_SOURCE_URL.count("family=") == 1
    assert "IBM+Plex+Sans" in FONT_SOURCE_URL
    assert "JetBrains+Mono" in CODE_FONT_SOURCE_URL
