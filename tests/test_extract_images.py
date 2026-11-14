from pathlib import Path

import pytest

from app.ingestion.extract_images import extract_images, image_to_node


@pytest.fixture(scope="module")
def extraction(sample_pdf: Path, sample_pdf_sha256: str, tmp_path_factory: pytest.TempPathFactory):
    media_dir = tmp_path_factory.mktemp("media")
    return extract_images(sample_pdf, sample_pdf_sha256, media_dir)


def test_embedded_images_kept(extraction) -> None:
    kept, _skipped = extraction

    assert len(kept) == 4
    assert [image.page for image in kept] == [2, 4, 5, 8]
    for image in kept:
        assert min(image.width, image.height) >= 80
        assert image.source == "embedded"


def test_repeated_logo_skipped_with_reason(extraction) -> None:
    _kept, skipped = extraction

    logo = next(s for s in skipped if s.reason == "repeated_on_most_pages")
    assert logo.pages == [1, 2, 3, 4, 5, 6, 7, 8]


def test_tiny_image_skipped_with_reason(extraction) -> None:
    _kept, skipped = extraction

    tiny = next(s for s in skipped if s.reason == "too_small")
    assert tiny.pages == [1]


def test_png_files_written(extraction) -> None:
    kept, _skipped = extraction

    for image in kept:
        assert image.image_path.is_file()
        assert image.image_path.suffix == ".png"


def test_figure_captions_found(extraction) -> None:
    kept, _skipped = extraction
    captions = {image.page: image.caption for image in kept}

    assert captions[2] == "Figure 1 : Schema du circuit hydraulique principal"
    assert captions[4] == "Figure 2 : Emplacement des capteurs de pression"
    assert captions[5] == "Figure 3 : Points de controle visuel quotidien"
    assert captions[8] == "Figure 5 : Pictogramme de securite generique"


def test_image_to_node_metadata(extraction, sample_pdf: Path, sample_pdf_sha256: str) -> None:
    kept, _skipped = extraction
    node = image_to_node(kept[0], sample_pdf_sha256, str(sample_pdf))

    assert node.metadata["content_type"] == "image"
    assert node.metadata["description"] == ""
    assert node.metadata["clip_embedding"] is None
    assert node.image_path == str(kept[0].image_path)
    assert node.metadata["source"] == "embedded"
    assert node.metadata["kind"] == "figure"
    assert node.metadata["sha256"] == kept[0].sha256
    assert len(node.metadata["sha256"]) == 64


def test_icon_sized_image_tagged_as_icon(extraction) -> None:
    kept, _skipped = extraction
    icon = next(image for image in kept if image.page == 8)
    diagram = next(image for image in kept if image.page == 2)

    assert icon.kind == "icon"
    assert icon.width < 150
    assert icon.height < 150
    assert diagram.kind == "figure"


def test_report_counts_by_kind(extraction) -> None:
    kept, _skipped = extraction
    counts: dict[str, int] = {}
    for image in kept:
        counts[image.kind] = counts.get(image.kind, 0) + 1

    assert counts == {"figure": 3, "icon": 1}
