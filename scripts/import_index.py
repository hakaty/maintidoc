"""Restore a snapshot produced by scripts/export_index.py.

Restores the PGVector dump via a direct TCP `psql` and re-extracts
data/processed/ and data/media/, then reports the row counts of both
PGVector tables so the caller can confirm the restore matches what was
exported (e.g. 140 text nodes / 52 images for the reference document).

Usage: python -m scripts.import_index export.zip [--expect-text N] [--expect-images N]

M7.5: this is the command the no-GPU colleague runs INSIDE the app
container (`docker compose --profile app run --rm app python -m
scripts.import_index data/<export>.zip`) -- it must never shell out to the
`docker` CLI or `docker compose exec` (neither is available inside a
container), only connect to the `db` service over the network already
reachable via DATABASE_URL (the service name "db" on Docker's internal
network, or 127.0.0.1:5433 natively).
"""

import argparse
import os
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import psycopg

from app.config import get_settings
from scripts.export_index import DUMP_ARCNAME


class ImportMismatchError(RuntimeError):
    pass


@dataclass
class _DbConn:
    user: str
    database: str
    host: str
    port: int
    password: str | None


def _db_conn(database_url: str) -> _DbConn:
    parts = urlsplit(database_url)
    return _DbConn(
        user=parts.username or "maintidoc",
        database=parts.path.lstrip("/") or "maintidoc",
        host=parts.hostname or "127.0.0.1",
        port=parts.port or 5432,
        password=parts.password,
    )


def _run_psql_restore(conn: _DbConn, dump_path: Path) -> None:
    env = dict(os.environ)
    if conn.password:
        env["PGPASSWORD"] = conn.password
    with dump_path.open("rb") as fh:
        subprocess.run(
            [
                "psql",
                "-h", conn.host,
                "-p", str(conn.port),
                "-U", conn.user,
                "-d", conn.database,
                "-v", "ON_ERROR_STOP=1",
            ],
            stdin=fh,
            env=env,
            check=True,
        )


def _table_count(database_url: str, table_name: str) -> int:
    with psycopg.connect(database_url) as conn:
        row = conn.execute(f"SELECT count(*) FROM data_{table_name}").fetchone()
        return int(row[0]) if row else 0


def import_index(
    zip_path: Path,
    expect_text: int | None = None,
    expect_images: int | None = None,
) -> dict[str, int]:
    settings = get_settings()
    conn = _db_conn(settings.database_url)

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            zf.extract(DUMP_ARCNAME, tmp_path)
            _run_psql_restore(conn, tmp_path / DUMP_ARCNAME)

        data_members = [n for n in names if n != DUMP_ARCNAME]
        zf.extractall(path=".", members=data_members)

    counts = {
        "text": _table_count(settings.database_url, settings.pgvector_text_table),
        "images": _table_count(settings.database_url, settings.pgvector_image_table),
    }
    print(f"text index: {counts['text']} nodes")
    print(f"image index: {counts['images']} nodes")

    if expect_text is not None and counts["text"] != expect_text:
        raise ImportMismatchError(
            f"text index has {counts['text']} nodes, expected {expect_text}"
        )
    if expect_images is not None and counts["images"] != expect_images:
        raise ImportMismatchError(
            f"image index has {counts['images']} nodes, expected {expect_images}"
        )
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scripts.import_index")
    parser.add_argument("zip_path", type=Path)
    parser.add_argument("--expect-text", type=int, default=None)
    parser.add_argument("--expect-images", type=int, default=None)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    import_index(args.zip_path, args.expect_text, args.expect_images)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
