from pathlib import Path

from defusedxml import ElementTree as DefusedElementTree

ASSETS_DIR = Path(__file__).resolve().parent.parent / "ui" / "assets"


def test_logo_files_exist() -> None:
    assert (ASSETS_DIR / "logo.svg").is_file()
    assert (ASSETS_DIR / "logo_small.svg").is_file()
    assert (ASSETS_DIR / "user.svg").is_file()


def test_logo_svgs_are_valid_xml() -> None:
    for name in ("logo.svg", "logo_small.svg", "user.svg"):
        DefusedElementTree.parse(ASSETS_DIR / name)  # raises ParseError if invalid


def test_logo_uses_petrol_and_safety_amber_only() -> None:
    content = (ASSETS_DIR / "logo.svg").read_text(encoding="utf-8")
    assert "#0B5C7A" in content
    assert "#F5A524" in content
    assert "#FFFFFF" in content


def test_logo_small_has_no_folded_corner_or_circle() -> None:
    small = (ASSETS_DIR / "logo_small.svg").read_text(encoding="utf-8")
    full = (ASSETS_DIR / "logo.svg").read_text(encoding="utf-8")
    assert "circle" not in small
    assert "circle" in full
    assert small.count("<path") < full.count("<path")
