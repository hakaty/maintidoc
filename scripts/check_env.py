"""Check that the DB/pgvector and the selected LLM provider are reachable.

Prints an OK / MISSING status table with a fix command for anything missing.
Never prints secret values (e.g. OPENROUTER_API_KEY).
"""

from dataclasses import dataclass, field

import httpx
import psycopg

from app.config import Settings, get_settings


@dataclass
class CheckResult:
    name: str
    ok: bool
    fix: str = field(default="")


def check_database(settings: Settings) -> CheckResult:
    try:
        with psycopg.connect(settings.database_url, connect_timeout=5) as conn:
            row = conn.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'").fetchone()
    except psycopg.OperationalError:
        return CheckResult("database (pgvector)", False, "run: docker compose up -d db")
    if row is None:
        return CheckResult("database (pgvector)", False, "run: python scripts/init_db.py")
    return CheckResult("database (pgvector)", True)


def check_ollama(settings: Settings) -> list[CheckResult]:
    try:
        response = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5.0)
        response.raise_for_status()
        available = {model["name"] for model in response.json().get("models", [])}
    except httpx.HTTPError:
        fix = "run: docker compose up -d ollama"
        return [
            CheckResult("ollama reachable", False, fix),
            CheckResult(f"ollama model {settings.ollama_llm_model}", False, fix),
            CheckResult(f"ollama model {settings.ollama_vlm_model}", False, fix),
        ]

    results = [CheckResult("ollama reachable", True)]
    for model in (settings.ollama_llm_model, settings.ollama_vlm_model):
        ok = model in available
        fix = "" if ok else f"run: ollama pull {model}"
        results.append(CheckResult(f"ollama model {model}", ok, fix))
    return results


def check_openrouter(settings: Settings) -> list[CheckResult]:
    try:
        response = httpx.get(
            f"{settings.openrouter_base_url}/models",
            headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
            timeout=10.0,
        )
        response.raise_for_status()
        available = {model["id"] for model in response.json().get("data", [])}
    except httpx.HTTPError:
        return [
            CheckResult(
                "openrouter reachable", False, "check OPENROUTER_API_KEY and network access"
            )
        ]

    results = [CheckResult("openrouter reachable", True)]
    for model in (settings.openrouter_llm_model, settings.openrouter_vlm_model):
        ok = model in available
        fix = "" if ok else "check the model id on https://openrouter.ai/models"
        results.append(CheckResult(f"openrouter model {model}", ok, fix))
    return results


def run_checks(settings: Settings) -> list[CheckResult]:
    results = [check_database(settings)]
    if settings.llm_provider == "ollama":
        results.extend(check_ollama(settings))
    else:
        results.extend(check_openrouter(settings))
    return results


def print_report(results: list[CheckResult]) -> None:
    width = max(len(result.name) for result in results)
    for result in results:
        status = "OK" if result.ok else "MISSING"
        line = f"[{status:7}] {result.name.ljust(width)}"
        if not result.ok and result.fix:
            line += f"  -> {result.fix}"
        print(line)


def main() -> int:
    results = run_checks(get_settings())
    print_report(results)
    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
