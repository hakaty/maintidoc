"""Per-page text extraction via LlamaIndex's PyMuPDFReader, excluding table areas.

The document content extracted here is UNTRUSTED data: it is stored and
displayed, never interpreted as instructions.
"""

import logging
import unicodedata
from functools import lru_cache
from pathlib import Path

import pymupdf
from llama_index.core.schema import Document as LlamaDocument
from llama_index.readers.file import PyMuPDFReader

from app.ingestion._captions import get_table_bboxes, page_text_blocks

logger = logging.getLogger(__name__)

_KEEP_CONTROL_CHARS = {"\n", "\t"}


def _strip_control_characters(text: str) -> str:
    return "".join(
        ch for ch in text if ch in _KEEP_CONTROL_CHARS or unicodedata.category(ch) != "Cc"
    )


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text)
    return _strip_control_characters(normalized).strip()


@lru_cache(maxsize=16)
def extract_text(pdf_path: Path, pdf_sha256: str) -> list[LlamaDocument]:
    """One LlamaIndex Document per page, with table-area text excluded.

    PyMuPDFReader loads the PDF page-by-page (used here per the project's
    LlamaIndex-based extraction convention); its raw per-page text is then
    replaced with a table-excluded, normalised version built directly from
    PyMuPDF text blocks.
    """
    # Loaded for total_pages/page bookkeeping, consistent with the reader's own convention.
    reader = PyMuPDFReader()
    base_documents = reader.load_data(str(pdf_path))
    total_pages = len(base_documents)

    documents: list[LlamaDocument] = []
    with pymupdf.open(pdf_path) as doc:
        for index, page in enumerate(doc):
            table_bboxes = get_table_bboxes(str(pdf_path), pdf_sha256, index + 1, page)
            blocks = page_text_blocks(page, exclude=table_bboxes)
            raw_text = "\n".join(block.text for block in blocks)
            text = normalize_text(raw_text)
            scanned = len(text) == 0

            if scanned:
                logger.warning(
                    "Page %d has no native text (scanned page?): %s", index + 1, pdf_path
                )

            documents.append(
                LlamaDocument(
                    text=text,
                    metadata={
                        "source_file": str(pdf_path),
                        "pdf_sha256": pdf_sha256,
                        "page": index + 1,
                        "total_pages": total_pages,
                        "scanned": scanned,
                    },
                )
            )
    return documents
