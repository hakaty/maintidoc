"""M7.5: static checks on docker-compose.yml and .dockerignore -- catch a
port exposed to 0.0.0.0, a service missing its profile, an inlined secret,
or .env/the venv accidentally shipped in the build context, without needing
Docker itself to run these tests."""

import re
from pathlib import Path

import yaml

COMPOSE_PATH = Path("docker-compose.yml")
DOCKERIGNORE_PATH = Path(".dockerignore")


def _load_compose() -> dict:
    return yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))


def test_compose_file_is_valid_yaml_with_services() -> None:
    compose = _load_compose()
    assert "services" in compose
    assert "db" in compose["services"]


def test_every_port_binding_is_localhost_only() -> None:
    """Never 0.0.0.0 or a bare container port -- every host:container
    mapping must start with "127.0.0.1:"."""
    compose = _load_compose()
    for name, service in compose["services"].items():
        for port in service.get("ports", []):
            assert str(port).startswith("127.0.0.1:"), f"{name}: port {port!r} not localhost-only"


def test_app_service_exists_under_the_app_profile() -> None:
    compose = _load_compose()
    app = compose["services"].get("app")
    assert app is not None, "no 'app' service defined"
    assert app.get("profiles") == ["app"]


def test_app_service_uses_env_file_not_inlined_secrets() -> None:
    compose = _load_compose()
    app = compose["services"]["app"]
    assert ".env" in app.get("env_file", [])


def test_app_service_depends_on_db_healthy() -> None:
    compose = _load_compose()
    app = compose["services"]["app"]
    assert app.get("depends_on", {}).get("db", {}).get("condition") == "service_healthy"


def test_db_has_a_pg_isready_healthcheck() -> None:
    compose = _load_compose()
    db = compose["services"]["db"]
    test = db.get("healthcheck", {}).get("test", [])
    assert any("pg_isready" in str(t) for t in test)


def test_no_ollama_service_in_the_default_profile() -> None:
    """The GPU owner's Ollama is always NATIVE, never dockerised on their
    machine -- any Ollama container service must be behind its own,
    never-started-by-default profile, never in the default (no "profiles"
    key) service set."""
    compose = _load_compose()
    for name, service in compose["services"].items():
        if "ollama" in name.lower() or "ollama" in str(service.get("image", "")).lower():
            profiles = service.get("profiles")
            assert profiles, f"{name}: an Ollama service must be behind an explicit profile"
            assert "app" not in profiles and profiles != [], (
                f"{name}: must not be started by the default/app profile"
            )


def test_app_profile_never_pulls_in_an_ollama_service() -> None:
    """`docker compose --profile app up -d` must never start an Ollama
    container on the GPU machine."""
    compose = _load_compose()
    for name, service in compose["services"].items():
        if "ollama" in name.lower():
            assert "app" not in (service.get("profiles") or [])


_SECRET_LOOKING_PATTERN = re.compile(r"sk-[A-Za-z0-9]{10,}|AKIA[0-9A-Z]{16}")


def test_no_secrets_inlined_in_compose_file() -> None:
    text = COMPOSE_PATH.read_text(encoding="utf-8")
    assert not _SECRET_LOOKING_PATTERN.search(text)
    assert "OPENROUTER_API_KEY=" not in text.replace("OPENROUTER_API_KEY=,", "")  # no literal value
    # Credentials come from env var substitution with a dev-only fallback
    # default, never a hardcoded production-looking secret.
    assert "${POSTGRES_PASSWORD" in text


def test_app_service_binds_hf_cache_and_data_volumes() -> None:
    compose = _load_compose()
    app = compose["services"]["app"]
    volume_specs = " ".join(app.get("volumes", []))
    assert "/app/models" in volume_specs
    assert "/app/data" in volume_specs


def test_dockerignore_excludes_env_and_venv() -> None:
    text = DOCKERIGNORE_PATH.read_text(encoding="utf-8")
    lines = {line.strip() for line in text.splitlines()}
    assert ".env" in lines
    assert ".venv/" in lines


def test_dockerignore_excludes_git_and_caches() -> None:
    text = DOCKERIGNORE_PATH.read_text(encoding="utf-8")
    assert ".git/" in text
    assert "__pycache__/" in text
    assert "exports/" in text
    assert "data/test_photos/" in text
