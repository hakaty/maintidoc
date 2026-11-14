"""Real-browser screenshot capture + automated visual checks for the UI
pixel-match verification loop: launches the real Streamlit app, drives it
with Playwright, captures each page in both light and dark OS colour
schemes (the app must look identical -- light -- in both), captures the
reference HTML files, builds side-by-side compare images, and runs a table
of automated checks (section 5 of the task) that make the script exit
non-zero on any failure.

Usage: python -m scripts.ui_screenshots [--port 8799]
Requires: python -m pip install playwright && python -m playwright install chromium
"""

import argparse
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
from PIL import Image
from playwright.sync_api import sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "ui" / "streamlit_app.py"
REFERENCE_DIR = PROJECT_ROOT / "docs" / "design" / "reference"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "design" / "screenshots"
FIXTURE_IMAGE = PROJECT_ROOT / "tests" / "fixtures" / "crop_p1_i2_pump.png"
STARTUP_TIMEOUT_SECONDS = 60
VIEWPORT = {"width": 1440, "height": 900}
VIEWPORT_WIDE = {"width": 1920, "height": 1080}

STREAMLIT_RED = (255, 75, 75)
PAGE_BG = (244, 246, 248)
INK = (15, 27, 45)

CHECK_RESULTS: list[tuple[str, str, bool, str]] = []  # (capture, check, passed, detail)


def _record(capture: str, check: str, passed: bool, detail: str = "") -> None:
    CHECK_RESULTS.append((capture, check, passed, detail))


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _kill_leftover_playwright_chromium() -> None:
    """Kill any Chromium process left running from a PREVIOUS run of this
    script that crashed/timed out before reaching browser.close() -- ONLY
    processes whose executable path is under Playwright's own browser
    cache (ms-playwright), never by image name alone (that would also
    match the user's real, unrelated Chrome windows)."""
    try:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Process -Filter \"Name='chrome.exe'\" | "
                "Where-Object { $_.ExecutablePath -like '*ms-playwright*' } | "
                "Select-Object -ExpandProperty ProcessId",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return
    for line in (result.stdout or "").splitlines():
        pid = line.strip()
        if pid.isdigit():
            print(f"Killing leftover Playwright Chromium process: PID {pid}")
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", pid], capture_output=True, check=False
            )


def _kill_leftover_process_on_port(port: int) -> None:
    """Section-7-era bug (a stale Streamlit process serving pre-fix code
    silently answered health checks on a reused port) -- always kill
    whatever is listening on the target port first, and print the PID."""
    try:
        result = subprocess.run(
            "netstat -ano",
            shell=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except FileNotFoundError:
        return
    stdout = result.stdout or ""
    for line in stdout.splitlines():
        if f":{port} " in line and "LISTENING" in line:
            pid = line.split()[-1]
            print(f"Killing leftover process on port {port}: PID {pid}")
            subprocess.run(
                f"taskkill /F /T /PID {pid}", shell=True, capture_output=True, check=False
            )


def _wait_for_health(base_url: str) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    last_exc: Exception | None = None
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"{base_url}/_stcore/health", timeout=2)
            if response.text.strip() == "ok":
                return
        except httpx.HTTPError as exc:  # noqa: PERF203 -- polling loop, retry is the point
            last_exc = exc
        time.sleep(0.5)
    raise RuntimeError(f"Streamlit did not become healthy in time: {last_exc}")


def _capture_reference_html(page, name: str, out_name: str) -> None:
    page.goto(f"file:///{(REFERENCE_DIR / name).as_posix()}")
    page.wait_for_timeout(300)
    page.screenshot(path=str(OUTPUT_DIR / out_name), full_page=True)


# --- automated visual checks (section 5) ------------------------------------


def _pixel_within(a: tuple[int, int, int], b: tuple[int, int, int], tolerance: int) -> bool:
    return all(abs(a[i] - b[i]) <= tolerance for i in range(3))


def _check_no_streamlit_red(image_path: Path, capture: str) -> None:
    img = Image.open(image_path).convert("RGB")
    small = img.resize((max(1, img.width // 4), max(1, img.height // 4)))
    hit = any(
        _pixel_within(small.getpixel((x, y)), STREAMLIT_RED, 30)
        for x in range(small.width)
        for y in range(small.height)
    )
    _record(capture, "no_streamlit_red", not hit)


def _check_background_color(image_path: Path, capture: str) -> None:
    img = Image.open(image_path).convert("RGB")
    # Sample a point in the main content area (right of the 256px sidebar,
    # near the top), away from any card/text.
    x, y = min(img.width - 1, 900), 5
    pixel = img.getpixel((x, y))
    _record(
        capture,
        "background_f4f6f8",
        _pixel_within(pixel, PAGE_BG, 3),
        f"sampled={pixel}",
    )


def _check_icons_render_as_glyphs(page, capture: str) -> None:
    """The DOM text content of a Material Symbols icon element is normally
    the icon's semantic ligature name (e.g. "forum") -- that's the intended
    mechanism (the font's ligature substitution turns that text into a
    single glyph, and it also serves as the accessible name/fallback if the
    font fails to load). That text content being non-empty is NOT the bug
    (an earlier version of this check wrongly required it to be empty,
    flagging every correctly-rendered icon as a false positive -- confirmed
    by inspecting the actual screenshots, where icons render as real
    glyphs). The actual section-0a bug was the font-family itself being
    overridden to the UI font, which is the only thing checked here."""
    icons = page.query_selector_all('[data-testid="stIconMaterial"]')
    bad = []
    for el in icons:
        font = el.evaluate("e => getComputedStyle(e).fontFamily")
        if "Material Symbols" not in font:
            bad.append((el.inner_text(), font))
    _record(capture, "icons_render_as_glyphs", not bad, f"n={len(icons)} bad={bad[:3]}")


def _check_no_material_shortcode_text(page, capture: str) -> None:
    body_text = page.inner_text("body")
    hit = ":material" in body_text
    _record(capture, "no_material_shortcode_text", not hit)


def _check_single_h1_ink_colored(page, capture: str) -> None:
    h1s = page.query_selector_all(".md-h1")
    ok = len(h1s) == 1
    detail = f"n={len(h1s)}"
    if ok:
        color = h1s[0].evaluate("e => getComputedStyle(e).color")
        ok = color == f"rgb({INK[0]}, {INK[1]}, {INK[2]})"
        detail += f" color={color}"
    _record(capture, "single_h1_ink_colored", ok, detail)


def _check_body_font_family(page, capture: str) -> None:
    font = page.evaluate("() => getComputedStyle(document.body).fontFamily")
    _record(capture, "body_font_ibm_plex_sans", "IBM Plex Sans" in font, f"font={font}")


def _check_no_theme_error_in_log(log_text: str, capture: str) -> None:
    _record(
        capture,
        "no_theme_option_error",
        "StreamlitInvalidThemeOptionError" not in log_text,
    )


def _check_dark_matches_light(light_path: Path, dark_path: Path, capture: str) -> None:
    if not (light_path.is_file() and dark_path.is_file()):
        _record(capture, "osdark_matches_light", False, "missing capture")
        return
    light = Image.open(light_path).convert("RGB")
    dark = Image.open(dark_path).convert("RGB")
    size = (min(light.width, dark.width), min(light.height, dark.height))
    light = light.resize(size)
    dark = dark.resize(size)
    samples = [(x, y) for x in (10, size[0] // 2, size[0] - 10) for y in (10, size[1] // 2)]
    diffs = [
        max(abs(light.getpixel(p)[i] - dark.getpixel(p)[i]) for i in range(3)) for p in samples
    ]
    ok = all(d <= 20 for d in diffs)
    _record(capture, "osdark_matches_light", ok, f"max_diff={max(diffs)}")


def run_checks_for_capture(page, image_path: Path, capture: str, server_log: str) -> None:
    _check_no_streamlit_red(image_path, capture)
    _check_background_color(image_path, capture)
    _check_icons_render_as_glyphs(page, capture)
    _check_no_material_shortcode_text(page, capture)
    _check_single_h1_ink_colored(page, capture)
    _check_body_font_family(page, capture)
    _check_no_theme_error_in_log(server_log, capture)


def _print_checks_table() -> bool:
    print("\n=== Automated visual checks ===")
    width_capture = max((len(r[0]) for r in CHECK_RESULTS), default=10)
    width_check = max((len(r[1]) for r in CHECK_RESULTS), default=10)
    all_ok = True
    for capture, check, passed, detail in CHECK_RESULTS:
        status = "PASS" if passed else "FAIL"
        all_ok = all_ok and passed
        print(f"{capture:<{width_capture}}  {check:<{width_check}}  {status}  {detail}")
    return all_ok


def _build_compare_image(ref_path: Path, app_path: Path, out_path: Path) -> None:
    if not (ref_path.is_file() and app_path.is_file()):
        return
    ref = Image.open(ref_path).convert("RGB")
    app = Image.open(app_path).convert("RGB")
    height = max(ref.height, app.height)
    canvas = Image.new("RGB", (ref.width + app.width + 8, height), (255, 255, 255))
    canvas.paste(ref, (0, 0))
    canvas.paste(app, (ref.width + 8, 0))
    canvas.save(out_path)


# --- page capture helpers ----------------------------------------------------


def _capture_app_empty(page, base_url: str, out_prefix: str, server_log: str) -> Path:
    page.goto(base_url)
    page.wait_for_selector("text=MaintiDoc", timeout=90_000)
    # Section 5 fix: on the very first capture of a freshly started server,
    # "MaintiDoc" (sidebar) renders immediately but the main content area
    # (including the h1) doesn't appear until the one-time model warm-up
    # (st.status("Chargement des modèles…")) finishes -- a fixed 1500ms
    # wait isn't enough for that first-time load, causing a deterministic
    # "0 h1 found" check failure on exactly the first capture only (every
    # later capture reuses the already-warm, @st.cache_resource'd models).
    # Wait for the real h1 element instead of a fixed timeout.
    page.wait_for_selector(".md-h1", timeout=120_000)
    page.wait_for_timeout(300)
    out = OUTPUT_DIR / f"{out_prefix}.png"
    page.screenshot(path=str(out), full_page=False)
    run_checks_for_capture(page, out, out_prefix, server_log)
    return out


def _capture_assistant_answered(page, base_url: str, out_prefix: str, server_log: str) -> Path:
    page.goto(base_url)
    page.wait_for_selector("text=MaintiDoc", timeout=90_000)
    page.get_by_role("button", name="Classe de résistance du béton ?").click()
    page.wait_for_selector("text=Réponse vérifiée", timeout=90_000)
    page.wait_for_timeout(500)
    out = OUTPUT_DIR / f"{out_prefix}.png"
    page.screenshot(path=str(out), full_page=True)
    run_checks_for_capture(page, out, out_prefix, server_log)
    return out


def _capture_image_search_empty(page, base_url: str, out_prefix: str, server_log: str) -> Path:
    page.goto(base_url)
    page.wait_for_selector("text=MaintiDoc", timeout=90_000)
    page.get_by_role("link", name="Recherche par image").click()
    page.wait_for_selector("text=Votre image", timeout=30_000)
    page.wait_for_timeout(500)
    out = OUTPUT_DIR / f"{out_prefix}.png"
    page.screenshot(path=str(out), full_page=True)
    run_checks_for_capture(page, out, out_prefix, server_log)
    return out


def _capture_image_search_result(page, base_url: str, out_prefix: str, server_log: str) -> Path:
    page.set_input_files("input[type=file]", str(FIXTURE_IMAGE))
    # "Meilleure correspondance" (not "Analysée", which doesn't appear
    # anywhere in ui/components/image_search.py -- a stale selector from an
    # earlier design draft caused every run to time out at 300s even though
    # the real page already rendered correct results within ~5s). Up to
    # 300s, no fixed sleeps -- the first search in a cold process pays a
    # one-time CUDA/model warm-up cost (see scripts/_profile_image_search.py).
    page.wait_for_selector("text=Meilleure correspondance", timeout=300_000)
    page.wait_for_timeout(500)
    out = OUTPUT_DIR / f"{out_prefix}.png"
    page.screenshot(path=str(out), full_page=True)
    run_checks_for_capture(page, out, out_prefix, server_log)
    return out


def _capture_document(page, base_url: str, out_prefix: str, server_log: str) -> Path:
    page.goto(base_url)
    page.wait_for_selector("text=MaintiDoc", timeout=90_000)
    page.get_by_role("link", name="Document").click()
    page.wait_for_selector("text=Qualité des réponses", timeout=30_000)
    page.wait_for_timeout(500)
    out = OUTPUT_DIR / f"{out_prefix}.png"
    page.screenshot(path=str(out), full_page=True)
    run_checks_for_capture(page, out, out_prefix, server_log)
    return out


def _capture_all(pw, browser, args, base_url: str, port: int) -> str:
    """Runs every capture using the ONE browser instance passed in. Returns
    the Streamlit server's captured stdout/stderr log (for the theme-error
    check). The caller owns closing `browser` (in a finally block) --
    this function never closes it itself, so a mid-run failure here still
    leaves cleanup to the caller."""
    if not args.skip_ref:
        print("Capturing reference HTML files...")
        ref_page = browser.new_page(viewport=VIEWPORT)
        _capture_reference_html(ref_page, "1-assistant.html", "1-assistant-ref.png")
        _capture_reference_html(ref_page, "2-recherche-image.html", "2-recherche-image-ref.png")
        _capture_reference_html(ref_page, "3-document.html", "3-document-ref.png")
        _capture_reference_html(ref_page, "4-composants.html", "4-composants-ref.png")
        ref_page.close()

    if args.skip_app:
        return ""

    print(f"Launching real app on 127.0.0.1:{port}...")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(APP_PATH),
            "--server.address",
            "127.0.0.1",
            "--server.port",
            str(port),
            "--server.headless",
            "true",
        ],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        _wait_for_health(base_url)
        print("App healthy, capturing screenshots...")

        for scheme, suffix in (("light", "light"), ("dark", "osdark")):
            context = browser.new_context(viewport=VIEWPORT, color_scheme=scheme)
            page = context.new_page()
            try:
                _capture_app_empty(page, base_url, f"1-assistant-app-empty-{suffix}", "")
                if suffix == "light":
                    _capture_assistant_answered(
                        page, base_url, f"1-assistant-app-answered-{suffix}", ""
                    )
                _capture_image_search_empty(
                    page, base_url, f"2-recherche-image-app-empty-{suffix}", ""
                )
                _capture_image_search_result(
                    page, base_url, f"2-recherche-image-app-result-{suffix}", ""
                )
                _capture_document(page, base_url, f"3-document-app-{suffix}", "")
            finally:
                context.close()

        # 1920px answered state (light only, per the task's explicit ask).
        wide_context = browser.new_context(viewport=VIEWPORT_WIDE, color_scheme="light")
        try:
            wide_page = wide_context.new_page()
            _capture_assistant_answered(
                wide_page, base_url, "1-assistant-app-answered-1920-light", ""
            )
        finally:
            wide_context.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    return proc.stdout.read() if proc.stdout else ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scripts.ui_screenshots")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--skip-app", action="store_true", help="only capture the reference HTML")
    parser.add_argument("--skip-ref", action="store_true", help="only capture the real app")
    args = parser.parse_args(argv)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _kill_leftover_playwright_chromium()
    port = args.port or _free_port()
    _kill_leftover_process_on_port(port)
    base_url = f"http://127.0.0.1:{port}"

    server_log = ""
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            server_log = _capture_all(pw, browser, args, base_url, port)
        finally:
            browser.close()

    if args.skip_app:
        return 0

    if "StreamlitInvalidThemeOptionError" in server_log:
        for i, (capture, check, _passed, detail) in enumerate(CHECK_RESULTS):
            if check == "no_theme_option_error":
                CHECK_RESULTS[i] = (capture, check, False, detail)

    # Compare images (ref | app), one per page.
    _build_compare_image(
        OUTPUT_DIR / "1-assistant-ref.png",
        OUTPUT_DIR / "1-assistant-app-answered-light.png",
        OUTPUT_DIR / "1-assistant-compare.png",
    )
    _build_compare_image(
        OUTPUT_DIR / "2-recherche-image-ref.png",
        OUTPUT_DIR / "2-recherche-image-app-result-light.png",
        OUTPUT_DIR / "2-recherche-image-compare.png",
    )
    _build_compare_image(
        OUTPUT_DIR / "3-document-ref.png",
        OUTPUT_DIR / "3-document-app-light.png",
        OUTPUT_DIR / "3-document-compare.png",
    )

    all_ok = _print_checks_table()
    print(f"\nScreenshots written to {OUTPUT_DIR}")
    if not all_ok:
        print("\nFAILED: one or more automated visual checks did not pass.")
        return 1
    print("\nAll automated visual checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
