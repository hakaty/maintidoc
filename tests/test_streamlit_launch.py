"""Regression test for the real blocking bug: a file named ui/app.py shadows
the top-level app/ package once Streamlit puts its own script directory
first on sys.path, causing "ModuleNotFoundError: No module named 'app.config';
'app' is not a package" on the real launch command. AppTest did not catch
this (it doesn't reproduce Streamlit's real sys.path ordering), hence the
real-subprocess check below.
"""

import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STREAMLIT_APP = PROJECT_ROOT / "ui" / "streamlit_app.py"
STARTUP_TIMEOUT_SECONDS = 45


def test_no_file_named_app_py_under_ui() -> None:
    """The root cause: never name a Streamlit entrypoint (or any file) app.py
    under ui/, since it shadows the top-level app/ package on sys.path."""
    offenders = list((PROJECT_ROOT / "ui").rglob("app.py"))
    assert offenders == [], f"found shadowing file(s): {offenders}"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.integration
def test_real_streamlit_launch_has_no_traceback() -> None:
    """Starts the real launch command in a subprocess, waits for
    /_stcore/health, fetches the page, asserts no traceback/ModuleNotFoundError
    appeared, then stops the process. Skips cleanly if streamlit is missing
    or the port can't be bound (environment issue, not a code issue)."""
    port = _free_port()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(STREAMLIT_APP),
            "--server.headless",
            "true",
            "--server.port",
            str(port),
        ],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        health_url = f"http://127.0.0.1:{port}/_stcore/health"
        deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
        healthy = False
        while time.monotonic() < deadline:
            if process.poll() is not None:
                break  # process exited early: definitely not healthy
            try:
                response = httpx.get(health_url, timeout=2)
                if response.status_code == 200 and response.text.strip() == "ok":
                    healthy = True
                    break
            except httpx.HTTPError:
                pass
            time.sleep(1)

        if not healthy:
            process.terminate()
            try:
                output = process.communicate(timeout=10)[0]
            except subprocess.TimeoutExpired:
                process.kill()
                output = ""
            pytest.fail(f"Streamlit did not become healthy in time. Output:\n{output}")

        page = httpx.get(f"http://127.0.0.1:{port}/", timeout=5)
        assert page.status_code == 200
        assert "Traceback" not in page.text
        assert "ModuleNotFoundError" not in page.text
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
