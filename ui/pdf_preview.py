"""On-demand PDF page thumbnail rendering for the "Voir la page" preview
next to a table source (section 6.6 of the UI feedback): a UI-layer-only
feature using the already-indexed PDF file directly via PyMuPDF -- it does
NOT touch the ingestion pipeline or its stored artifacts.
"""

from pathlib import Path

import pymupdf
import streamlit as st

THUMBNAIL_DPI = 100


@st.cache_data(show_spinner=False)
def get_page_thumbnail_png(pdf_path_str: str, page_number: int) -> bytes | None:
    """1-indexed `page_number` (matches citations' `page` field). Returns
    None on any failure (missing file, out-of-range page, corrupt PDF) --
    the caller just hides the thumbnail in that case, never crashes."""
    pdf_path = Path(pdf_path_str)
    if not pdf_path.is_file():
        return None
    try:
        with pymupdf.open(pdf_path) as doc:
            if not (1 <= page_number <= doc.page_count):
                return None
            page = doc[page_number - 1]
            pixmap = page.get_pixmap(dpi=THUMBNAIL_DPI)
            return pixmap.tobytes("png")
    except Exception:  # noqa: BLE001 -- best-effort preview, never fatal
        return None
