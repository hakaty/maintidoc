import hashlib
from pathlib import Path

import pytest
from scripts.make_sample_pdf import DEFAULT_OUTPUT, make_pdf


@pytest.fixture(scope="session")
def sample_pdf() -> Path:
    return make_pdf(DEFAULT_OUTPUT)


@pytest.fixture(scope="session")
def sample_pdf_sha256(sample_pdf: Path) -> str:
    return hashlib.sha256(sample_pdf.read_bytes()).hexdigest()
