import subprocess
import zipfile
from pathlib import Path

import psycopg
import pytest
from scripts import export_index as export_module
from scripts import import_index as import_module

from app.config import get_settings


def test_export_zip_excludes_env_and_models_and_contains_data(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    processed = tmp_path / "data_processed"
    processed.mkdir()
    (processed / "manifest.json").write_text("{}", encoding="utf-8")
    media = tmp_path / "data_media"
    media.mkdir()
    (media / "img.png").write_bytes(b"fake-png-bytes")

    monkeypatch.setattr(export_module, "DATA_DIRS", (processed, media))
    monkeypatch.setattr(export_module, "DEFAULT_OUTPUT_DIR", tmp_path / "exports")

    captured_cmd = {}

    def _fake_run(cmd, stdout=None, check=None):
        captured_cmd["cmd"] = cmd
        if stdout is not None:
            stdout.write(b"-- fake dump\n")

    monkeypatch.setattr(export_module.subprocess, "run", _fake_run)

    output_path = export_module.export_index()

    with zipfile.ZipFile(output_path) as zf:
        names = zf.namelist()

    assert export_module.DUMP_ARCNAME in names
    assert not any(".env" in n for n in names)
    assert not any("models" in n.split("/") for n in names)
    assert any(n.endswith("manifest.json") for n in names)
    assert any(n.endswith("img.png") for n in names)
    assert "pg_dump" in captured_cmd["cmd"]


def test_import_raises_on_text_count_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    zip_path = tmp_path / "export.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(import_module.DUMP_ARCNAME, "-- dump")
        zf.writestr("data/processed/x.json", "{}")

    monkeypatch.setattr(import_module.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(import_module, "_table_count", lambda url, table: 5)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(import_module.ImportMismatchError, match="text index"):
        import_module.import_index(zip_path, expect_text=140)


def test_import_extracts_data_files_relative_to_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    zip_path = tmp_path / "export.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(import_module.DUMP_ARCNAME, "-- dump")
        zf.writestr("data/processed/x.json", '{"ok": true}')

    monkeypatch.setattr(import_module.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(import_module, "_table_count", lambda url, table: 0)
    monkeypatch.chdir(tmp_path)

    import_module.import_index(zip_path)

    assert (tmp_path / "data" / "processed" / "x.json").is_file()


@pytest.mark.integration
def test_export_and_restore_round_trip_into_scratch_db(tmp_path: Path) -> None:
    """Real pg_dump + real psql restore (into a disposable scratch database,
    never the live `maintidoc` database) + real row-count check."""
    settings = get_settings()
    scratch_db = "maintidoc_import_test"

    server_db_url = settings.database_url.rsplit("/", 1)[0] + "/postgres"
    try:
        psycopg.connect(settings.database_url).close()
    except psycopg.OperationalError:
        pytest.skip("Database not reachable (run `docker compose up -d db` first)")

    try:
        output_path = export_module.export_index(tmp_path / "export.zip")
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        pytest.skip(f"docker compose exec pg_dump unavailable: {exc}")

    with zipfile.ZipFile(output_path) as zf:
        dump_text = zf.read(export_module.DUMP_ARCNAME).decode("utf-8", errors="replace")
    assert "maintidoc_text" in dump_text or "maintidoc_images" in dump_text

    with psycopg.connect(server_db_url, autocommit=True) as conn:
        conn.execute(f'DROP DATABASE IF EXISTS "{scratch_db}"')
        conn.execute(f'CREATE DATABASE "{scratch_db}"')
    try:
        subprocess.run(
            ["docker", "compose", "exec", "-T", "db", "psql", "-U", "maintidoc", "-d", scratch_db],
            input=dump_text.encode("utf-8"),
            check=True,
            capture_output=True,
        )
        scratch_url = settings.database_url.rsplit("/", 1)[0] + f"/{scratch_db}"
        with psycopg.connect(scratch_url) as conn:
            text_count = conn.execute(
                f"SELECT count(*) FROM data_{settings.pgvector_text_table}"
            ).fetchone()[0]
            image_count = conn.execute(
                f"SELECT count(*) FROM data_{settings.pgvector_image_table}"
            ).fetchone()[0]
        with psycopg.connect(settings.database_url) as conn:
            live_text_count = conn.execute(
                f"SELECT count(*) FROM data_{settings.pgvector_text_table}"
            ).fetchone()[0]
            live_image_count = conn.execute(
                f"SELECT count(*) FROM data_{settings.pgvector_image_table}"
            ).fetchone()[0]
        assert text_count == live_text_count
        assert image_count == live_image_count
    finally:
        with psycopg.connect(server_db_url, autocommit=True) as conn:
            conn.execute(f'DROP DATABASE IF EXISTS "{scratch_db}"')
