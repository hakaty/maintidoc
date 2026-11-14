from pathlib import Path

import pytest

from app.indexing import __main__ as cli
from app.indexing.stores import BuildResult, ImageSearchResult, SearchResult


@pytest.fixture(autouse=True)
def _patch_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: "fake-settings")


def test_build_command_calls_build_index_with_args(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    captured = {}

    def _fake_build_index(processed_dir, force=False, settings=None):
        captured["processed_dir"] = processed_dir
        captured["force"] = force
        return BuildResult(
            pdf_sha256="deadbeef",
            text_nodes_total=3,
            text_nodes_added=3,
            image_nodes_total=1,
            image_nodes_added=1,
            text_count_in_db=3,
            image_count_in_db=1,
        )

    monkeypatch.setattr(cli, "build_index", _fake_build_index)

    exit_code = cli.main(["build", str(tmp_path), "--force"])

    assert exit_code == 0
    assert captured["processed_dir"] == tmp_path
    assert captured["force"] is True
    out = capsys.readouterr().out
    assert "deadbeef" in out


def test_search_command_passes_type_and_mode(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    captured = {}

    def _fake_search_text(query, k=5, content_type=None, mode="vector", settings=None):
        captured.update(query=query, k=k, content_type=content_type, mode=mode)
        return [SearchResult(node_id="n1", page=4, content_type="table", score=0.9, snippet="x")]

    monkeypatch.setattr(cli, "search_text", _fake_search_text)

    exit_code = cli.main(
        ["search", "quelle est la pression ?", "--k", "3", "--type", "table", "--mode", "hybrid"]
    )

    assert exit_code == 0
    assert captured == {
        "query": "quelle est la pression ?",
        "k": 3,
        "content_type": "table",
        "mode": "hybrid",
    }
    out = capsys.readouterr().out
    assert "page=4" in out
    assert "type=table" in out


def test_search_command_with_no_results_prints_placeholder(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(cli, "search_text", lambda *a, **k: [])

    exit_code = cli.main(["search", "une question sans reponse"])

    assert exit_code == 0
    assert "(no results)" in capsys.readouterr().out


def test_search_image_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    captured = {}

    def _fake_search_image(image_path, k=5, settings=None):
        captured.update(image_path=image_path, k=k)
        return [
            ImageSearchResult(
                image_id="img1", image_path="x.png", page=1, caption="Figure 1", score=0.8
            )
        ]

    monkeypatch.setattr(cli, "search_image", _fake_search_image)
    image_path = tmp_path / "query.png"

    exit_code = cli.main(["search-image", str(image_path), "--k", "3"])

    assert exit_code == 0
    assert captured["image_path"] == image_path
    assert captured["k"] == 3
    out = capsys.readouterr().out
    assert "image_id=img1" in out


def test_search_image_text_command(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    captured = {}

    def _fake_search_image_text(text, k=5, settings=None):
        captured.update(text=text, k=k)
        return [
            ImageSearchResult(image_id="img2", image_path=None, page=2, caption=None, score=0.5)
        ]

    monkeypatch.setattr(cli, "search_image_text", _fake_search_image_text)

    exit_code = cli.main(["search-image-text", "a pump", "--k", "2"])

    assert exit_code == 0
    assert captured == {"text": "a pump", "k": 2}
    out = capsys.readouterr().out
    assert "image_id=img2" in out


def test_search_command_handles_non_cp1252_characters(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Document text is untrusted PDF content and may contain characters outside
    the Windows console's cp1252 codepage (e.g. U+2264 "<="): must not crash."""
    monkeypatch.setattr(
        cli,
        "search_text",
        lambda *a, **k: [
            SearchResult(
                node_id="n1",
                page=17,
                content_type="text",
                score=0.5,
                snippet="température ≤ 110°C",
            )
        ],
    )

    exit_code = cli.main(["search", "une question"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "page=17" in out


def test_missing_command_errors() -> None:
    with pytest.raises(SystemExit):
        cli.main([])
