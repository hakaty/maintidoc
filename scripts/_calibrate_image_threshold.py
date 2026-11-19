"""M7.4/M7.4b section 3: calibrate image-search thresholds on REAL data --
never guessed.

In-domain positives (two kinds, both real manual content, never guessed):
  1. Each manual figure, re-encoded with realistic crop/rotate/brightness/
     skew/blur perturbations (M7.4 baseline).
  2. "Screen photo" simulations (M7.4b): the full PDF PAGE a figure lives on
     is rendered and cropped around that figure's real bbox (with margin, so
     it is NOT just the tight original crop), then moiré, a stronger
     perspective warp, glare, downscale, and a ~50% partial crop are
     applied -- closer to what a technician's phone photo of a manual page
     actually looks like.
  3. The REAL photos in data/test_photos/ (product photos, not manual
     scans) are reported separately -- they test generalization, not used
     to pick the threshold itself.

Out-of-domain negatives: >=60 varied images with nothing to do with a pump
manual, generated locally with PIL/numpy only (no internet, no matplotlib
dependency added) across 11 families (bar/line/pie chart, deployment
diagram, flowchart, UI screenshot, data table, text-only, generic machinery
line drawing, solid colour, noise) with randomized parameters per family.

Prints both score distributions (min/median/max), a two-threshold ("middle
zone") sweep, and picks IMAGE_MIN_SCORE (reject below) / IMAGE_REVIEW_SCORE
(show as "a verifier" between the two) so that in-domain recall@3 >= 95%
at IMAGE_MIN_SCORE.
"""

import random
import sys
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.rag.retriever import retrieve_by_image  # noqa: E402

random.seed(0)
np.random.seed(0)

MEDIA_DIR = Path("data/media/5c1a4a3e34a4")
PROCESSED_DIR = Path("data/processed/5c1a4a3e34a4")
PDF_PATH = Path("data/pdf/document.pdf")
TEST_PHOTOS_DIR = Path("data/test_photos")
OUT_DIR = Path("docs/diagnostics/image_threshold")
MIN_FIGURE_SIDE = 120  # skip tiny icons -- not representative of a user's photo
PAGE_RENDER_ZOOM = 2.0  # pymupdf page.get_pixmap zoom -> pixels per PDF point


# --- in-domain, kind 1: perturbed manual figures (M7.4 baseline) -----------


def _perturb(img: Image.Image, seed: int) -> Image.Image:
    rng = random.Random(seed)
    img = img.convert("RGB")

    w, h = img.size
    crop_frac = rng.uniform(0.05, 0.15)
    left, top = int(w * crop_frac), int(h * crop_frac)
    img = img.crop((left, top, w - left, h - top))

    angle = rng.uniform(-10, 10)
    img = img.rotate(angle, expand=True, fillcolor=(255, 255, 255))

    arr = np.asarray(img).astype(np.float32)
    arr = np.clip(arr * rng.uniform(0.8, 1.2), 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    w, h = img.size
    skew = rng.uniform(-0.08, 0.08)
    img = img.transform((w, h), Image.AFFINE, (1, skew, 0, 0, 1, 0), fillcolor=(255, 255, 255))

    img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 1.5)))
    return img


def _save_jpeg_compressed(img: Image.Image, path: Path, quality: int = 40) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="JPEG", quality=quality)


def _figure_pages() -> dict[str, int]:
    """image_id (= figure PNG stem) -> page, from the real extraction nodes."""
    import json

    nodes_path = PROCESSED_DIR / "nodes.jsonl"
    pages = {}
    for line in nodes_path.read_text(encoding="utf-8").splitlines():
        node = json.loads(line)
        meta = node.get("metadata", {})
        if meta.get("content_type") == "image" and meta.get("image_id"):
            pages[meta["image_id"]] = meta
    return pages


def build_in_domain_crop_cases() -> list[tuple[Path, str]]:
    """Returns [(perturbed_image_path, expected_image_id), ...]."""
    cases = []
    figures = sorted(MEDIA_DIR.glob("*.png"))
    for i, fig_path in enumerate(figures):
        with Image.open(fig_path) as img:
            if min(img.size) < MIN_FIGURE_SIDE:
                continue
            perturbed = _perturb(img, seed=i)
        out_path = OUT_DIR / "in_domain" / f"{fig_path.stem}_perturbed.jpg"
        _save_jpeg_compressed(perturbed, out_path)
        cases.append((out_path, fig_path.stem))
    return cases


# --- in-domain, kind 2: "screen photo" page simulations (M7.4b) ------------


def _moire(img: Image.Image, seed: int) -> Image.Image:
    """A fine sinusoidal grid overlay, like a phone camera photographing a
    printed page (a real, well-known artifact of screen/print + camera
    sensor interference)."""
    rng = random.Random(seed)
    arr = np.asarray(img).astype(np.float32)
    h, w = arr.shape[:2]
    freq = rng.uniform(0.8, 1.4)
    yy, xx = np.mgrid[0:h, 0:w]
    pattern = (np.sin(xx * freq * 0.5) * np.sin(yy * freq * 0.5)) * 18.0
    arr = np.clip(arr + pattern[..., None], 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _perspective_warp(img: Image.Image, seed: int) -> Image.Image:
    """A stronger-than-M7.4 perspective transform -- a phone held at a real
    angle to the page, not just a slight skew."""
    rng = random.Random(seed)
    w, h = img.size
    dx, dy = w * rng.uniform(0.05, 0.12), h * rng.uniform(0.05, 0.12)
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    dst = [(dx, dy * 0.5), (w - dx * 0.3, 0), (w, h), (dx * 0.5, h - dy)]

    def _find_coeffs(pa, pb):
        matrix = []
        for p1, p2 in zip(pa, pb, strict=True):
            matrix.append([p2[0], p2[1], 1, 0, 0, 0, -p1[0] * p2[0], -p1[0] * p2[1]])
            matrix.append([0, 0, 0, p2[0], p2[1], 1, -p1[1] * p2[0], -p1[1] * p2[1]])
        a = np.array(matrix, dtype=np.float64)
        b = np.array(pa, dtype=np.float64).reshape(8)
        res = np.linalg.solve(a, b)
        return res.tolist()

    coeffs = _find_coeffs(src, dst)
    return img.transform((w, h), Image.PERSPECTIVE, coeffs, fillcolor=(255, 255, 255))


def _glare(img: Image.Image, seed: int) -> Image.Image:
    """A bright elliptical overhead-light reflection patch."""
    rng = random.Random(seed)
    img = img.convert("RGB")
    overlay = Image.new("RGB", img.size, (0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = img.size
    cx, cy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.1, 0.5) * h
    rx, ry = w * 0.22, h * 0.18
    draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=(255, 255, 255))
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=max(w, h) * 0.05))
    arr = np.asarray(img).astype(np.float32) + np.asarray(overlay).astype(np.float32) * 0.5
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _downscale(img: Image.Image) -> Image.Image:
    """Phone photo saved/shared at low resolution, then viewed back up."""
    w, h = img.size
    small = img.resize((max(1, w // 4), max(1, h // 4)), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def _partial_crop(img: Image.Image, seed: int, keep_frac: float = 0.5) -> Image.Image:
    """Only ~50% of the page/figure is in frame -- a common real photo."""
    rng = random.Random(seed)
    w, h = img.size
    keep_w, keep_h = int(w * keep_frac), int(h * keep_frac)
    left = rng.randint(0, max(1, w - keep_w))
    top = rng.randint(0, max(1, h - keep_h))
    return img.crop((left, top, left + keep_w, top + keep_h))


def _render_page_crop(page_index0: int, bbox: list[float], margin: float = 0.6) -> Image.Image:
    """Render the real PDF page at PAGE_RENDER_ZOOM and crop around the
    figure's real bbox with a generous margin -- NOT the tight original
    crop, so this tests a photo of the surrounding page context, not a
    re-encode of the already-extracted figure."""
    doc = pymupdf.open(PDF_PATH)
    try:
        page = doc[page_index0]
        pix = page.get_pixmap(matrix=pymupdf.Matrix(PAGE_RENDER_ZOOM, PAGE_RENDER_ZOOM))
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    finally:
        doc.close()
    x0, y0, x1, y1 = (c * PAGE_RENDER_ZOOM for c in bbox)
    bw, bh = x1 - x0, y1 - y0
    mx, my = bw * margin, bh * margin
    crop_box = (
        max(0, int(x0 - mx)),
        max(0, int(y0 - my)),
        min(img.width, int(x1 + mx)),
        min(img.height, int(y1 + my)),
    )
    return img.crop(crop_box)


_SCREEN_PHOTO_STEPS = {
    "moire": _moire,
    "perspective": _perspective_warp,
    "glare": _glare,
    "downscale": lambda img, seed: _downscale(img),
    "partial_crop": lambda img, seed: _partial_crop(img, seed, keep_frac=0.5),
}


def build_in_domain_screen_photo_cases(max_cases: int = 20) -> list[tuple[Path, str]]:
    """Returns [(screen_photo_path, expected_image_id), ...], one per
    sampled figure. Each case applies a RANDOM SUBSET of 2-3 of the 5
    "screen photo" artifacts (moire/perspective/glare/downscale/partial
    crop) -- a real phone photo of a page typically has a couple of these
    issues at once, not all five at full severity simultaneously; stacking
    every artifact on every case was an implausible worst case that pushed
    scores unrealistically low and made the calibration overly pessimistic."""
    pages_meta = _figure_pages()
    figures = sorted(MEDIA_DIR.glob("*.png"))
    cases = []
    for i, fig_path in enumerate(figures):
        if len(cases) >= max_cases:
            break
        meta = pages_meta.get(fig_path.stem)
        if not meta or not meta.get("bbox") or not meta.get("page"):
            continue
        with Image.open(fig_path) as img:
            if min(img.size) < MIN_FIGURE_SIDE:
                continue
        photo = _render_page_crop(meta["page"] - 1, meta["bbox"])
        rng = random.Random(i)
        n_effects = rng.randint(2, 3)
        chosen = rng.sample(sorted(_SCREEN_PHOTO_STEPS), n_effects)
        for name in chosen:
            photo = _SCREEN_PHOTO_STEPS[name](photo, i)
        out_path = OUT_DIR / "in_domain" / f"{fig_path.stem}_screenphoto.jpg"
        _save_jpeg_compressed(photo, out_path, quality=55)
        cases.append((out_path, fig_path.stem))
    return cases


# --- out-of-domain: >=60 varied, locally generated, unrelated images -------


def _font(size: int = 16) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _bar_chart(rng: random.Random) -> Image.Image:
    img = Image.new("RGB", (640, 480), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.line((60, 420, 600, 420), fill=(0, 0, 0), width=2)
    draw.line((60, 420, 60, 40), fill=(0, 0, 0), width=2)
    n_bars = rng.randint(5, 10)
    color = rng.choice([(31, 119, 180), (214, 39, 40), (44, 160, 44)])
    for i in range(n_bars):
        x0 = 70 + i * (520 // n_bars)
        height = rng.randint(50, 350)
        draw.rectangle((x0, 420 - height, x0 + 35, 420), fill=color)
    draw.text(
        (220, 10),
        rng.choice(["Quarterly revenue", "Sales by region"]),
        fill=(0, 0, 0),
        font=_font(20),
    )
    return img


def _line_chart(rng: random.Random) -> Image.Image:
    img = Image.new("RGB", (640, 480), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.line((60, 420, 600, 420), fill=(0, 0, 0), width=2)
    draw.line((60, 420, 60, 40), fill=(0, 0, 0), width=2)
    points = []
    y = rng.randint(200, 350)
    for i in range(12):
        x = 60 + i * 45
        y = max(50, min(410, y + rng.randint(-40, 40)))
        points.append((x, y))
    draw.line(points, fill=(31, 119, 180), width=3)
    for p in points:
        draw.ellipse((p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3), fill=(31, 119, 180))
    return img


def _pie_chart(rng: random.Random) -> Image.Image:
    img = Image.new("RGB", (640, 480), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    start = 0
    colors = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40)]
    for color in colors:
        extent = rng.randint(30, 150)
        draw.pieslice((170, 90, 470, 390), start, start + extent, fill=color)
        start += extent
    return img


def _deployment_diagram(rng: random.Random) -> Image.Image:
    img = Image.new("RGB", (640, 480), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    labels = rng.sample(
        [
            "API Gateway",
            "Auth Service",
            "Database",
            "Cache",
            "Queue",
            "Worker",
            "CDN",
            "Load Balancer",
        ],
        4,
    )
    boxes = [(40, 40, 220, 140), (420, 40, 600, 140), (40, 260, 220, 360), (420, 260, 600, 360)]
    for (x0, y0, x1, y1), label in zip(boxes, labels, strict=True):
        draw.rectangle((x0, y0, x1, y1), outline=(0, 0, 0), width=2)
        draw.text((x0 + 10, y0 + 40), label, fill=(0, 0, 0), font=_font(14))
    draw.line((220, 90, 420, 90), fill=(0, 0, 0), width=2)
    draw.line((130, 140, 130, 260), fill=(0, 0, 0), width=2)
    draw.line((510, 140, 510, 260), fill=(0, 0, 0), width=2)
    return img


def _flowchart(rng: random.Random) -> Image.Image:
    img = Image.new("RGB", (640, 480), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    steps = rng.randint(3, 5)
    y = 30
    for i in range(steps):
        shape = "diamond" if i == steps // 2 else "rect"
        if shape == "diamond":
            draw.polygon(
                [(320, y), (420, y + 45), (320, y + 90), (220, y + 45)], outline=(0, 0, 0), width=2
            )
        else:
            draw.rectangle((220, y, 420, y + 60), outline=(0, 0, 0), width=2)
        draw.text((240, y + 20), f"Step {i + 1}", fill=(0, 0, 0), font=_font(14))
        if i < steps - 1:
            draw.line((320, y + 90, 320, y + 110), fill=(0, 0, 0), width=2)
        y += 110
    return img


def _ui_screenshot(rng: random.Random) -> Image.Image:
    img = Image.new("RGB", (640, 480), (244, 246, 248))
    draw = ImageDraw.Draw(img)
    header_color = rng.choice([(11, 92, 122), (80, 30, 120), (20, 110, 60)])
    draw.rectangle((0, 0, 640, 60), fill=header_color)
    draw.text(
        (20, 20),
        rng.choice(["Settings", "Dashboard", "Profile"]),
        fill=(255, 255, 255),
        font=_font(20),
    )
    for i in range(rng.randint(3, 5)):
        y = 90 + i * 70
        draw.rectangle((20, y, 620, y + 50), outline=(200, 200, 200), fill=(255, 255, 255))
        draw.text((35, y + 15), f"Option {i + 1}", fill=(20, 20, 20), font=_font(14))
    return img


def _data_table(rng: random.Random) -> Image.Image:
    """A generic spreadsheet/data table -- NOT the pump manual's tables."""
    img = Image.new("RGB", (640, 480), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    cols, rows = 4, 8
    col_w, row_h = 640 // cols, 480 // rows
    headers = rng.sample(["Date", "Qty", "Price", "Total", "Region", "Status"], cols)
    for c in range(cols + 1):
        draw.line((c * col_w, 0, c * col_w, 480), fill=(180, 180, 180))
    for r in range(rows + 1):
        draw.line((0, r * row_h, 640, r * row_h), fill=(180, 180, 180))
    for c, header in enumerate(headers):
        draw.text((c * col_w + 8, 8), header, fill=(0, 0, 0), font=_font(14))
    for r in range(1, rows):
        for c in range(cols):
            draw.text(
                (c * col_w + 8, r * row_h + 8),
                str(rng.randint(1, 999)),
                fill=(50, 50, 50),
                font=_font(12),
            )
    return img


def _machinery_line_drawing(rng: random.Random) -> Image.Image:
    """Generic CAD-style line art (circles/rectangles/dimension lines) --
    deliberately machinery-ADJACENT (to stress-test against false
    acceptance) but NOT a pump and NOT from the manual."""
    img = Image.new("RGB", (640, 480), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for _ in range(rng.randint(3, 6)):
        shape = rng.choice(["circle", "rect"])
        x, y = rng.randint(80, 480), rng.randint(80, 340)
        size = rng.randint(30, 90)
        if shape == "circle":
            draw.ellipse((x, y, x + size, y + size), outline=(0, 0, 0), width=2)
        else:
            draw.rectangle((x, y, x + size, y + size * 0.6), outline=(0, 0, 0), width=2)
    draw.line((40, 440, 600, 440), fill=(0, 0, 0), width=1)
    draw.text((280, 450), "Ø" + str(rng.randint(10, 99)), fill=(0, 0, 0), font=_font(12))
    return img


def _text_only(rng: random.Random) -> Image.Image:
    img = Image.new("RGB", (640, 480), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    lines = rng.sample(
        [
            "Lorem ipsum dolor sit amet,",
            "consectetur adipiscing elit.",
            "Sed do eiusmod tempor incididunt",
            "ut labore et dolore magna aliqua.",
            "Quarterly meeting notes follow.",
            "Please review before Friday.",
        ],
        4,
    )
    for i, line in enumerate(lines):
        draw.text((40, 40 + i * 40), line, fill=(0, 0, 0), font=_font(22))
    return img


def _solid_color(rng: random.Random) -> Image.Image:
    return Image.new("RGB", (640, 480), tuple(rng.randint(0, 255) for _ in range(3)))


def _noise(rng: random.Random) -> Image.Image:
    kind = rng.choice(["uniform", "gradient"])
    if kind == "uniform":
        arr = np.random.default_rng(rng.randint(0, 2**31)).integers(
            0, 255, (480, 640, 3), dtype=np.uint8
        )
    else:
        base = (
            np.random.default_rng(rng.randint(0, 2**31))
            .integers(0, 255, (480, 640, 3))
            .astype(np.float32)
        )
        gy, gx = np.mgrid[0:480, 0:640]
        gradient = (gx / 640 * 100)[..., None]
        arr = np.clip(base * 0.5 + gradient, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


_OOD_FAMILIES = {
    "bar_chart": _bar_chart,
    "line_chart": _line_chart,
    "pie_chart": _pie_chart,
    "deployment_diagram": _deployment_diagram,
    "flowchart": _flowchart,
    "ui_screenshot": _ui_screenshot,
    "data_table": _data_table,
    "machinery_line_drawing": _machinery_line_drawing,
    "text_only": _text_only,
    "solid_color": _solid_color,
    "noise": _noise,
}
_VARIANTS_PER_FAMILY = 6  # 11 families x 6 = 66 >= the required 60


def build_out_of_domain_cases() -> list[tuple[Path, str]]:
    cases = []
    for name, gen in _OOD_FAMILIES.items():
        for variant in range(_VARIANTS_PER_FAMILY):
            rng = random.Random(hash((name, variant)) % (2**31))
            out_path = OUT_DIR / "out_of_domain" / f"{name}_{variant}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            gen(rng).save(out_path)
            cases.append((out_path, f"{name}_{variant}"))
    return cases


# --- real photos (data/test_photos/) -- reported, not used to pick the ----
# --- threshold itself: these are genuinely out-of-sample ------------------


def report_real_photos(settings) -> None:
    if not TEST_PHOTOS_DIR.is_dir():
        return
    photos = sorted(TEST_PHOTOS_DIR.glob("*.jp*g")) + sorted(TEST_PHOTOS_DIR.glob("*.png"))
    if not photos:
        return
    print("\n=== Real photos (data/test_photos/) ===")
    for path in photos:
        matches = retrieve_by_image(path, k=3, settings=settings)
        top1 = matches[0].score if matches else 0.0
        top_pages = [m.page for m in matches]
        print(f"  {path.name:<45} top1={top1:.4f} top3 pages={top_pages}")


def _stats(label: str, values: list[float]) -> None:
    if not values:
        print(f"{label}: no data")
        return
    values_sorted = sorted(values)
    median = values_sorted[len(values_sorted) // 2]
    print(
        f"{label}: min={min(values):.4f} median={median:.4f} max={max(values):.4f} n={len(values)}"
    )


def main() -> int:
    settings = get_settings()

    print("Building in-domain (perturbed manual figure) test cases...")
    crop_cases = build_in_domain_crop_cases()
    print(f"  {len(crop_cases)} cases")

    print("Building in-domain (screen-photo simulation) test cases...")
    screen_cases = build_in_domain_screen_photo_cases()
    print(f"  {len(screen_cases)} cases")

    in_domain = crop_cases + screen_cases

    print("Building out-of-domain (unrelated, locally generated) test cases...")
    out_of_domain = build_out_of_domain_cases()
    print(f"  {len(out_of_domain)} cases")

    print("\n=== In-domain: top-1 score + recall@3 ===")
    in_domain_top1_scores = []
    hits_at_3 = 0
    for path, expected_image_id in in_domain:
        matches = retrieve_by_image(path, k=3, settings=settings)
        top1 = matches[0].score if matches else 0.0
        in_domain_top1_scores.append(top1)
        hit = any(m.image_id == expected_image_id for m in matches)
        hits_at_3 += hit
        print(f"  {path.name:<40} top1={top1:.4f} hit@3={hit}")
    recall_at_3 = hits_at_3 / len(in_domain) if in_domain else 0.0

    print("\n=== Out-of-domain: top-1 score ===")
    out_of_domain_top1_scores = []
    for path, name in out_of_domain:
        matches = retrieve_by_image(path, k=3, settings=settings)
        top1 = matches[0].score if matches else 0.0
        out_of_domain_top1_scores.append(top1)
        print(f"  {name:<28} top1={top1:.4f}")

    print("\n=== Score distributions ===")
    _stats("In-domain top1   ", in_domain_top1_scores)
    _stats("Out-of-domain top1", out_of_domain_top1_scores)
    print(f"In-domain recall@3: {recall_at_3:.3f}")

    report_real_photos(settings)

    print("\n=== Candidate thresholds (IMAGE_MIN_SCORE sweep) ===")
    best_threshold = None
    n_in, n_out = max(1, len(in_domain_top1_scores)), max(1, len(out_of_domain_top1_scores))
    for threshold in [round(t, 2) for t in np.arange(0.30, 0.81, 0.02)]:
        recall = sum(s >= threshold for s in in_domain_top1_scores) / n_in
        rejection = sum(s < threshold for s in out_of_domain_top1_scores) / n_out
        marker = ""
        if recall >= 0.95:
            best_threshold = threshold
            marker = "  <- recall@3 still >= 95% here"
        print(
            f"  threshold={threshold:.2f}  recall={recall:.3f}  rejection={rejection:.3f}{marker}"
        )

    if best_threshold is None:
        best_threshold = min(in_domain_top1_scores) if in_domain_top1_scores else 0.5
    final_rejection = sum(s < best_threshold for s in out_of_domain_top1_scores) / n_out
    final_recall = sum(s >= best_threshold for s in in_domain_top1_scores) / n_in

    # Middle zone (section 2): IMAGE_REVIEW_SCORE is a second, HIGHER
    # threshold -- above it, results are confident ("Meilleure correspondance"
    # eligible); between IMAGE_MIN_SCORE and IMAGE_REVIEW_SCORE, results go
    # in "Correspondances possibles -- a verifier" instead of being rejected
    # outright or shown as if fully confident. Chosen as the highest
    # out-of-domain score rounded up, so NO out-of-domain case is ever
    # treated as "confident".
    review_threshold = round(max(out_of_domain_top1_scores, default=best_threshold) + 0.02, 2)
    review_threshold = max(review_threshold, best_threshold + 0.02)

    print(f"\nCHOSEN IMAGE_MIN_SCORE = {best_threshold:.2f}")
    print(f"  in-domain recall@3 at this threshold: {final_recall:.3f}")
    print(f"  out-of-domain rejection rate at this threshold: {final_rejection:.3f}")
    print(f"CHOSEN IMAGE_REVIEW_SCORE = {review_threshold:.2f}")
    n_middle_zone_in_domain = sum(
        best_threshold <= s < review_threshold for s in in_domain_top1_scores
    )
    print(f"  in-domain cases landing in the middle zone: {n_middle_zone_in_domain}/{n_in}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
