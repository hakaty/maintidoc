import psycopg
import pytest
from scripts.init_db import ensure_vector_extension

from app.config import get_settings


@pytest.mark.integration
def test_ensure_vector_extension_is_idempotent() -> None:
    database_url = get_settings().database_url
    try:
        first_run = ensure_vector_extension(database_url)
        second_run = ensure_vector_extension(database_url)
    except psycopg.OperationalError:
        pytest.skip("Database not reachable (run `docker compose up -d db` first)")

    assert first_run is True
    assert second_run is True
