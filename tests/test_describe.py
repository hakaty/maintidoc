import base64
import io
import json
from pathlib import Path

import httpx
import pytest
from PIL import Image as PILImage

from app.config import Settings
from app.images.describe import (
    MAX_CONTEXT_CHARS,
    MAX_DESCRIPTION_LENGTH,
    MAX_VLM_IMAGE_SIDE,
    PROMPT_VERSION,
    _build_repair_prompt,
    _encode_image_b64,
    _try_parse,
    build_prompt,
    describe_image,
)

VALID_JSON = json.dumps(
    {
        "description": "Un schéma technique représentant une pompe.",
        "visible_text": "illisible",
        "kind": "schéma",
        "safety_relevant": False,
        "confidence": "medium",
    }
)


@pytest.fixture
def tiny_image(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.png"
    PILImage.new("RGB", (8, 8), color=(10, 20, 30)).save(path)
    return path


@pytest.fixture
def large_image(tmp_path: Path) -> Path:
    path = tmp_path / "large.png"
    PILImage.new("RGB", (2054, 1489), color=(40, 60, 80)).save(path)
    return path


def test_encode_image_b64_leaves_small_images_untouched(tmp_path: Path) -> None:
    path = tmp_path / "small.png"
    PILImage.new("RGB", (300, 200), color=(10, 20, 30)).save(path)

    encoded = _encode_image_b64(path)

    assert encoded == base64.b64encode(path.read_bytes()).decode("ascii")


def test_encode_image_b64_downscales_large_images(large_image: Path) -> None:
    original_bytes = large_image.read_bytes()

    encoded = _encode_image_b64(large_image)
    decoded = base64.b64decode(encoded)

    assert decoded != original_bytes  # re-encoded, not the raw file
    assert large_image.read_bytes() == original_bytes  # original on disk untouched

    resized = PILImage.open(io.BytesIO(decoded))
    assert max(resized.size) <= MAX_VLM_IMAGE_SIDE
    # aspect ratio preserved (2054x1489 ~ 1.38)
    assert abs(resized.size[0] / resized.size[1] - 2054 / 1489) < 0.01


def test_build_prompt_includes_context_as_data() -> None:
    prompt = build_prompt("Figure 1 : exemple", 5, "Texte de la page environnante.")

    assert "Figure 1 : exemple" in prompt
    assert "Page : 5" in prompt
    assert "Texte de la page environnante." in prompt
    assert "DONNÉES" in prompt or "DONNÉE" in prompt or "données" in prompt


def test_build_prompt_truncates_context() -> None:
    long_context = "x" * (MAX_CONTEXT_CHARS + 500)
    prompt = build_prompt(None, 1, long_context)

    assert "x" * MAX_CONTEXT_CHARS in prompt
    assert "x" * (MAX_CONTEXT_CHARS + 1) not in prompt


def test_build_prompt_handles_missing_caption() -> None:
    prompt = build_prompt(None, 1, "")
    assert "(aucune)" in prompt


def test_try_parse_valid_json() -> None:
    parsed = _try_parse(VALID_JSON)

    assert parsed is not None
    assert parsed.kind == "schéma"
    assert parsed.safety_relevant is False
    assert parsed.confidence == "medium"


def test_try_parse_invalid_json_returns_none() -> None:
    assert _try_parse("not json at all") is None


def test_try_parse_missing_required_field_returns_none() -> None:
    data = json.loads(VALID_JSON)
    del data["confidence"]
    assert _try_parse(json.dumps(data)) is None


def test_try_parse_invalid_kind_returns_none() -> None:
    data = json.loads(VALID_JSON)
    data["kind"] = "not a real category"
    assert _try_parse(json.dumps(data)) is None


def test_try_parse_drops_visible_text_that_only_echoes_description() -> None:
    """Regression test for the p5_i5 case: a gloves pictogram with literally
    no text printed on it got visible_text="gants" because the description
    happened to say "des gants" -- the model echoed its own word choice
    instead of reporting that nothing is actually printed on the image."""
    data = {
        "description": (
            "L'image représente un symbole de sécurité indiquant "
            "l'importance de porter des gants."
        ),
        "visible_text": "gants",
        "kind": "pictogramme de sécurité",
        "safety_relevant": True,
        "confidence": "high",
    }

    parsed = _try_parse(json.dumps(data))

    assert parsed is not None
    assert parsed.visible_text == ""


def test_try_parse_keeps_visible_text_genuinely_read_on_pictogram() -> None:
    """The "EX" pictogram genuinely has "EX" printed on it -- real text
    actually read off the image must survive the post-check."""
    data = {
        "description": (
            "Un triangle jaune avec les lettres EX, indiquant un risque d'atmosphère explosive."
        ),
        "visible_text": "EX",
        "kind": "pictogramme de sécurité",
        "safety_relevant": True,
        "confidence": "high",
    }

    parsed = _try_parse(json.dumps(data))

    assert parsed is not None
    assert parsed.visible_text == "EX"


def test_try_parse_keeps_illisible_visible_text() -> None:
    data = json.loads(VALID_JSON)
    data["kind"] = "logo"
    data["visible_text"] = "illisible"

    parsed = _try_parse(json.dumps(data))

    assert parsed is not None
    assert parsed.visible_text == "illisible"


def test_try_parse_does_not_touch_visible_text_for_other_kinds() -> None:
    """The echo-check is scoped to pictograms/logos -- a schema diagram
    legitimately has multi-word labels that happen to also appear in its
    own description, and must not be stripped."""
    data = {
        "description": "Un schéma montrant le point A et le point B reliés par une ligne.",
        "visible_text": "A B",
        "kind": "schéma",
        "safety_relevant": False,
        "confidence": "medium",
    }

    parsed = _try_parse(json.dumps(data))

    assert parsed is not None
    assert parsed.visible_text == "A B"


def test_prompt_version_is_v2() -> None:
    assert PROMPT_VERSION == "figure_description_v2"


def test_v2_prompt_file_tells_model_not_to_echo_description_into_visible_text() -> None:
    prompt = build_prompt("caption", 1, "")
    assert "visible_text" in prompt
    assert "gants" in prompt  # the concrete counter-example from the real failure


@pytest.mark.parametrize(
    "alias",
    [
        "safety_reliable",
        "safety_reliance",
        "safety_related",
        "safety_restricted",
        "safety_restriction",
        "safety_remark",
        "safety_notes",  # any unseen safety_* variant must also be caught
    ],
)
def test_try_parse_normalizes_known_vlm_key_aliases(alias: str) -> None:
    """qwen2.5vl:3b reproducibly writes one of these instead of "safety_relevant"."""
    data = json.loads(VALID_JSON)
    data[alias] = data.pop("safety_relevant")

    parsed = _try_parse(json.dumps(data))

    assert parsed is not None
    assert parsed.safety_relevant is False


def test_try_parse_coerces_string_boolean_for_safety_relevant() -> None:
    """Confirmed on the real PDF: the model sometimes writes "safety_remark":
    "true" -- a JSON string, not a boolean -- under a wrong key name too."""
    data = json.loads(VALID_JSON)
    data["safety_remark"] = "true"
    del data["safety_relevant"]

    parsed = _try_parse(json.dumps(data))

    assert parsed is not None
    assert parsed.safety_relevant is True


@pytest.mark.parametrize("confidence_word", ["low", "medium", "high"])
def test_try_parse_coerces_confidence_bleed_through_as_truthy(confidence_word: str) -> None:
    """Confirmed on the real PDF (3/4 of one failure batch): the model
    sometimes writes its own "confidence" value into the mis-named safety
    field, e.g. "safety_reliance": "high". Treated as truthy -- ambiguous
    either way, and under-flagging real safety content is the worse error."""
    data = json.loads(VALID_JSON)
    data["safety_reliance"] = confidence_word
    del data["safety_relevant"]

    parsed = _try_parse(json.dumps(data))

    assert parsed is not None
    assert parsed.safety_relevant is True


def test_try_parse_prefers_existing_safety_relevant_over_alias() -> None:
    """If "safety_relevant" is already present, a stray safety_*-looking key
    must not clobber its value."""
    data = json.loads(VALID_JSON)
    data["safety_relevant"] = True
    data["safety_notes"] = False

    parsed = _try_parse(json.dumps(data))

    assert parsed is not None
    assert parsed.safety_relevant is True


def test_repair_prompt_names_the_exact_mismatch() -> None:
    data = json.loads(VALID_JSON)
    data["safety_reliance"] = data.pop("safety_relevant")
    repair = _build_repair_prompt("original prompt", json.dumps(data))

    assert "safety_relevant" in repair
    assert "safety_reliance" in repair


def test_describe_image_valid_first_try(
    monkeypatch: pytest.MonkeyPatch, tiny_image: Path
) -> None:
    calls = []

    def fake_call_vlm(image_b64: str, prompt: str, settings: Settings) -> str:
        calls.append(prompt)
        return VALID_JSON

    monkeypatch.setattr("app.images.describe.call_vlm", fake_call_vlm)

    result = describe_image(tiny_image, "caption", 1, "context", settings=Settings(_env_file=None))

    assert len(calls) == 1  # no repair call needed
    assert result.status == "ok"
    assert result.kind == "schéma"


def test_describe_image_repairs_then_succeeds(
    monkeypatch: pytest.MonkeyPatch, tiny_image: Path
) -> None:
    responses = iter(["not valid json", VALID_JSON])
    calls = []

    def fake_call_vlm(image_b64: str, prompt: str, settings: Settings) -> str:
        calls.append(prompt)
        return next(responses)

    monkeypatch.setattr("app.images.describe.call_vlm", fake_call_vlm)

    result = describe_image(tiny_image, "caption", 1, "context", settings=Settings(_env_file=None))

    assert len(calls) == 2
    assert "Ta réponse précédente" in calls[1]  # second call used the repair prompt
    assert result.status == "ok"


def test_describe_image_falls_back_after_failed_repair(
    monkeypatch: pytest.MonkeyPatch, tiny_image: Path
) -> None:
    monkeypatch.setattr("app.images.describe.call_vlm", lambda *a, **kw: "still not json")

    result = describe_image(
        tiny_image, "ma légende", 3, "context", fallback_kind="photo",
        settings=Settings(_env_file=None),
    )

    assert result.status == "fallback"
    assert result.description == "ma légende"
    assert result.kind == "photo"
    assert result.confidence == "low"
    assert result.safety_relevant is False


def test_describe_image_http_error_falls_back_without_crashing(
    monkeypatch: pytest.MonkeyPatch, tiny_image: Path
) -> None:
    """A transient HTTP failure (e.g. context-length 400, connection drop)
    must not raise out of describe_image -- one bad image can't crash a batch."""

    def raising_call_vlm(image_b64: str, prompt: str, settings: Settings) -> str:
        request = httpx.Request("POST", "http://127.0.0.1:11434/api/chat")
        response = httpx.Response(400, request=request, text="context overflow")
        raise httpx.HTTPStatusError("400", request=request, response=response)

    monkeypatch.setattr("app.images.describe.call_vlm", raising_call_vlm)

    result = describe_image(tiny_image, "ma légende", 1, "", settings=Settings(_env_file=None))

    assert result.status == "fallback"
    assert result.description == "ma légende"


def test_describe_image_fallback_uses_default_when_no_caption(
    monkeypatch: pytest.MonkeyPatch, tiny_image: Path
) -> None:
    monkeypatch.setattr("app.images.describe.call_vlm", lambda *a, **kw: "nope")

    result = describe_image(tiny_image, None, 1, "", settings=Settings(_env_file=None))

    assert result.description == "Image sans description"


def test_describe_image_caps_description_length(
    monkeypatch: pytest.MonkeyPatch, tiny_image: Path
) -> None:
    long_description = "a" * (MAX_DESCRIPTION_LENGTH + 500)
    data = json.loads(VALID_JSON)
    data["description"] = long_description
    monkeypatch.setattr("app.images.describe.call_vlm", lambda *a, **kw: json.dumps(data))

    result = describe_image(tiny_image, "caption", 1, "context", settings=Settings(_env_file=None))

    assert len(result.description) == MAX_DESCRIPTION_LENGTH


@pytest.mark.integration
def test_real_ollama_describes_an_image(tiny_image: Path) -> None:
    settings = Settings(_env_file=None)
    try:
        result = describe_image(tiny_image, "test caption", 1, "", settings=settings)
    except httpx.HTTPError as exc:
        pytest.skip(f"Ollama unreachable: {exc}")

    assert result.status in ("ok", "fallback")
    assert result.description
