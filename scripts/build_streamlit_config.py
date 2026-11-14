"""Regenerate .streamlit/config.toml from ui/theme.py's tokens.

Streamlit reads config.toml at process startup, before any app code runs,
so the theme can't be applied purely at runtime -- run this once after
changing a token in ui/theme.py (it's also safe to commit the output).
"""

from pathlib import Path

from ui.theme import build_config_toml

OUTPUT_PATH = Path(".streamlit/config.toml")


def main() -> int:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(build_config_toml(), encoding="utf-8")
    print(f"Written {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
