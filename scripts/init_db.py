"""Create the pgvector extension in the target database. Safe to run more than once."""

import logging

import psycopg

from app.config import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DB_CONNECT_TIMEOUT_SECONDS = 5


def ensure_vector_extension(database_url: str) -> bool:
    with psycopg.connect(
        database_url, autocommit=True, connect_timeout=DB_CONNECT_TIMEOUT_SECONDS
    ) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        row = conn.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'").fetchone()
    return row is not None


def main() -> int:
    database_url = get_settings().database_url
    if not ensure_vector_extension(database_url):
        logger.error("pgvector extension could not be verified")
        return 1
    logger.info("pgvector extension enabled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
