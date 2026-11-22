"""Export a portable snapshot: PGVector dump + data/processed/ + data/media/.

Lets a teammate without a GPU reuse already-computed extraction/VLM/indexing
results instead of re-running them. Never includes .env, model weights
(models/), or any other secret: only the DB dump and the two named data
directories go into the zip.

Usage: python -m scripts.export_index [output.zip]
"""

import os
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

from app.config import get_settings

DEFAULT_OUTPUT_DIR = Path("exports")
DUMP_ARCNAME = "db_dump.sql"
DATA_DIRS = (Path("data/processed"), Path("data/media"))


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


def _run_pg_dump(conn: _DbConn, dump_path: Path) -> None:
    # M7.5 fix: a direct TCP `pg_dump` (never `docker compose exec`) --
    # the latter only works when run on the HOST with the `docker` CLI and
    # compose project context available, never from inside a container
    # (scripts/import_index.py's restore counterpart must run there, per
    # the Docker colleague workflow -- same connection logic, so both must
    # work identically regardless of where they run).
    env = dict(os.environ)
    if conn.password:
        env["PGPASSWORD"] = conn.password
    with dump_path.open("wb") as fh:
        subprocess.run(
            [
                "pg_dump",
                "-h", conn.host,
                "-p", str(conn.port),
                "-U", conn.user,
                "-d", conn.database,
            ],
            stdout=fh,
            env=env,
            check=True,
        )


def _add_directory(zf: zipfile.ZipFile, directory: Path) -> int:
    if not directory.is_dir():
        return 0
    count = 0
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            zf.write(path, arcname=str(path))
            count += 1
    return count


def export_index(output_path: Path | None = None) -> Path:
    settings = get_settings()
    conn = _db_conn(settings.database_url)

    if output_path is None:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        output_path = DEFAULT_OUTPUT_DIR / f"maintidoc_export_{stamp}.zip"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dump_path = output_path.with_suffix(".sql.tmp")
    _run_pg_dump(conn, dump_path)

    try:
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(dump_path, arcname=DUMP_ARCNAME)
            file_counts = {str(d): _add_directory(zf, d) for d in DATA_DIRS}
    finally:
        dump_path.unlink(missing_ok=True)

    print(f"Exported to {output_path}")
    for directory, count in file_counts.items():
        print(f"  {directory}: {count} files")
    return output_path


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    output_path = Path(argv[0]) if argv else None
    export_index(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
