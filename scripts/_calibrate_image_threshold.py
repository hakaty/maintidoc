"""M7.4 section 3: calibrate IMAGE_MIN_SCORE on REAL data -- never guessed.

In-domain positives: each manual figure, re-encoded with realistic
perturbations (crop, rotation, blur, JPEG compression, brightness, a
phone-photo-like perspective warp). Each perturbed copy must still find its
OWN page among the top-3 real results.

Out-of-domain negatives: images with nothing to do with a pump manual,
generated locally with PIL/numpy only (no internet, no matplotlib
dependency added) -- a bar chart, a UML/deployment-style box-and-arrow
diagram, a fake UI screenshot, a text-only image, a solid-colour image, a
noise image.

Prints both score distributions (min/median/max), the rejection rate at
several candidate thresholds, and the in-domain recall@3, then picks the
threshold that maximises correct out-of-domain rejection while keeping
in-domain recall@3 >= 95%.
"""

import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.rag.retriever import retrieve_by_image  # noqa: E402

random.seed(0)
np.random.seed(0)

MEDIA_DIR = Path("data/media/5c1a4a3e34a4")
OUT_DIR = Path("docs/diagnostics/image_threshold")
MIN_FIGURE_SIDE = 120  # skip tiny icons -- not representative of a user's photo


# --- in-domain: perturbed manual figures ------------------------------------


def _perturb(img: Image.Image, seed: int) -> Image.Image:
    rng = random.Random(seed)
    img = img.convert("RGB")

    # Crop 5-15% off each side.
    w, h = img.size
    crop_frac = rng.uniform(0.05, 0.15)
    left, top = int(w * crop_frac), int(h * crop_frac)
    img = img.crop((left, top, w - left, h - top))

    # Rotate +/-10 deg.
    angle = rng.uniform(-10, 10)
    img = img.rotate(angle, expand=True, fillcolor=(255, 255, 255))

    # Brightness.
    arr = np.asarray(img).astype(np.float32)
    arr = np.clip(arr * rng.uniform(0.8, 1.2), 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    # Slight perspective-like skew via an affine transform (phone-photo feel).
    w, h = img.size
    skew = rng.uniform(-0.08, 0.08)
    img = img.transform(
        (w, h), Image.AFFINE, (1, skew, 0, 0, 1, 0), fillcolor=(255, 255, 255)
    )

    # Blur.
    from PIL import ImageFilter

    img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 1.5)))

    return img


def _save_jpeg_compressed(img: Image.Image, path: Path, quality: int = 40) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="JPEG", quality=quality)


def build_in_domain_cases() -> list[tuple[Path, str]]:
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
        expected_image_id = fig_path.stem
        cases.append((out_path, expected_image_id))
    return cases


# --- out-of-domain: locally generated, unrelated images ---------------------


def _font(size: int = 16) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _bar_chart() -> Image.Image:
    img = Image.new("RGB", (640, 480), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.line((60, 420, 600, 420), fill=(0, 0, 0), width=2)
    draw.line((60, 420, 60, 40), fill=(0, 0, 0), width=2)
    rng = random.Random(1)
    for i in range(8):
        x0 = 80 + i * 65
        height = rng.randint(50, 350)
        draw.rectangle((x0, 420 - height, x0 + 40, 420), fill=(31, 119, 180))
    draw.text((250, 10), "Quarterly revenue", fill=(0, 0, 0), font=_font(20))
    return img


def _deployment_diagram() -> Image.Image:
    img = Image.new("RGB", (640, 480), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    boxes = [(40, 40, 220, 140, "API Gateway"), (420, 40, 600, 140, "Auth Service"),
             (40, 260, 220, 360, "Database"), (420, 260, 600, 360, "Cache")]
    for x0, y0, x1, y1, label in boxes:
        draw.rectangle((x0, y0, x1, y1), outline=(0, 0, 0), width=2)
        draw.text((x0 + 10, y0 + 40), label, fill=(0, 0, 0), font=_font(14))
    draw.line((220, 90, 420, 90), fill=(0, 0, 0), width=2)
    draw.line((130, 140, 130, 260), fill=(0, 0, 0), width=2)
    draw.line((510, 140, 510, 260), fill=(0, 0, 0), width=2)
    return img


def _ui_screenshot() -> Image.Image:
    img = Image.new("RGB", (640, 480), (244, 246, 248))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, 640, 60), fill=(11, 92, 122))
    draw.text((20, 20), "Settings", fill=(255, 255, 255), font=_font(20))
    for i in range(4):
        y = 90 + i * 70
        draw.rectangle((20, y, 620, y + 50), outline=(200, 200, 200), fill=(255, 255, 255))
        draw.text((35, y + 15), f"Option {i + 1}", fill=(20, 20, 20), font=_font(14))
    return img


def _text_only() -> Image.Image:
    img = Image.new("RGB", (640, 480), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    lines = [
        "Lorem ipsum dolor sit amet,",
        "consectetur adipiscing elit.",
        "Sed do eiusmod tempor incididunt",
        "ut labore et dolore magna aliqua.",
    ]
    for i, line in enumerate(lines):
        draw.text((40, 40 + i * 40), line, fill=(0, 0, 0), font=_font(22))
    return img


def _solid_color() -> Image.Image:
    return Image.new("RGB", (640, 480), (120, 180, 220))


def _noise() -> Image.Image:
    arr = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def build_out_of_domain_cases() -> list[tuple[Path, str]]:
    generators = {
        "bar_chart": _bar_chart,
        "deployment_diagram": _deployment_diagram,
        "ui_screenshot": _ui_screenshot,
        "text_only": _text_only,
        "solid_color": _solid_color,
        "noise": _noise,
    }
    cases = []
    for name, gen in generators.items():
        out_path = OUT_DIR / "out_of_domain" / f"{name}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        gen().save(out_path)
        cases.append((out_path, name))
    return cases


def main() -> int:
    settings = get_settings()

    print("Building in-domain (perturbed manual figure) test cases...")
    in_domain = build_in_domain_cases()
    print(f"  {len(in_domain)} cases")

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
        print(f"  {name:<25} top1={top1:.4f}")

    def _stats(label: str, values: list[float]) -> None:
        if not values:
            print(f"{label}: no data")
            return
        values_sorted = sorted(values)
        median = values_sorted[len(values_sorted) // 2]
        print(f"{label}: min={min(values):.4f} median={median:.4f} max={max(values):.4f}")

    print("\n=== Score distributions ===")
    _stats("In-domain top1   ", in_domain_top1_scores)
    _stats("Out-of-domain top1", out_of_domain_top1_scores)
    print(f"In-domain recall@3: {recall_at_3:.3f}")

    print("\n=== Candidate thresholds ===")
    # Rejection only improves as the threshold rises, recall only drops --
    # so the BEST threshold is the HIGHEST one that still keeps recall@3
    # >= 95% (the lowest such threshold would leave rejection on the table).
    best_threshold = None
    n_in, n_out = max(1, len(in_domain_top1_scores)), max(1, len(out_of_domain_top1_scores))
    for threshold in [round(t, 2) for t in np.arange(0.30, 0.81, 0.02)]:
        recall = sum(s >= threshold for s in in_domain_top1_scores) / n_in
        rejection = sum(s < threshold for s in out_of_domain_top1_scores) / n_out
        marker = ""
        if recall >= 0.95:
            best_threshold = threshold
            marker = "  <- recall@3 still >= 95% here"
        line = (
            f"  threshold={threshold:.2f}  recall={recall:.3f}"
            f"  rejection={rejection:.3f}{marker}"
        )
        print(line)

    if best_threshold is None:
        best_threshold = min(in_domain_top1_scores) if in_domain_top1_scores else 0.5
    final_rejection = sum(s < best_threshold for s in out_of_domain_top1_scores) / max(
        1, len(out_of_domain_top1_scores)
    )
    final_recall = sum(s >= best_threshold for s in in_domain_top1_scores) / max(
        1, len(in_domain_top1_scores)
    )
    print(f"\nCHOSEN IMAGE_MIN_SCORE = {best_threshold:.2f}")
    print(f"  in-domain recall@3 at this threshold: {final_recall:.3f}")
    print(f"  out-of-domain rejection rate at this threshold: {final_rejection:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
