"""PDF parser using PyMuPDF (fitz) with automatic Tesseract OCR fallback.

Extracts text on a per-page basis, preserving page numbers for chunk metadata.
If a page contains sparse or no digital text (e.g. scanned slides, images),
it automatically falls back to Tesseract OCR when available.
"""

import io
import logging
import os
from pathlib import Path
import shutil
from typing import Any, List, Optional, Tuple, Union
from PIL import Image
import pymupdf

from rag_service.config import settings

logger = logging.getLogger(__name__)

# State tracker for OCR availability and logging
_TESSERACT_AVAILABLE: Optional[bool] = None
_LOGGED_OCR_STATUS: bool = False


def is_ocr_available() -> bool:
    """Check if Tesseract OCR is operational on the host system.

    Resolves Tesseract binary path via:
    1. settings.TESSERACT_CMD (configured via TESSERACT_CMD env var)
    2. System PATH via shutil.which('tesseract')
    3. Optional convenience fallback for standard Windows installer path

    Logs availability status once at INFO level.
    """
    global _TESSERACT_AVAILABLE, _LOGGED_OCR_STATUS
    if _TESSERACT_AVAILABLE is not None:
        if not _LOGGED_OCR_STATUS:
            logger.info(
                "Tesseract OCR fallback: %s",
                "enabled" if _TESSERACT_AVAILABLE else "disabled",
            )
            _LOGGED_OCR_STATUS = True
        return _TESSERACT_AVAILABLE

    try:
        import pytesseract

        # 1. Configured via TESSERACT_CMD setting / env var
        if settings.TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD
        # 2. In PATH
        elif shutil.which("tesseract"):
            pass
        # 3. Optional convenience fallback for default Windows install directory; never raises
        elif os.name == "nt":
            try:
                win_path = Path("C:/Program Files/Tesseract-OCR/tesseract.exe")
                if win_path.is_file():
                    pytesseract.pytesseract.tesseract_cmd = str(win_path)
            except Exception:
                pass

        pytesseract.get_tesseract_version()
        _TESSERACT_AVAILABLE = True
    except Exception:
        _TESSERACT_AVAILABLE = False

    if not _LOGGED_OCR_STATUS:
        logger.info(
            "Tesseract OCR fallback: %s",
            "enabled" if _TESSERACT_AVAILABLE else "disabled",
        )
        _LOGGED_OCR_STATUS = True

    return _TESSERACT_AVAILABLE


def clean_text(text: str) -> str:
    """Clean extracted page text for downstream chunking."""
    lines = [line.strip() for line in text.splitlines()]
    non_empty = [line for line in lines if line]
    return "\n".join(non_empty)


def _serialize_table(tab: Any) -> List[str]:
    """Serialize a PyMuPDF Table object into lines formatted as:
    Header1: val1 | Header2: val2 | ...
    """
    headers = [h.strip() if h and h.strip() else f"Col{i+1}" for i, h in enumerate(tab.header.names)]
    rows_text: List[str] = []

    try:
        from pymupdf.table import extract_cells

        cell_boxes = [[c for c in r.cells] for r in tab.rows]
        j_start = 0 if tab.header.external else 1
        for r_idx in range(j_start, len(cell_boxes)):
            row = cell_boxes[r_idx]
            row_parts = []
            for c_idx, cell in enumerate(row):
                h = headers[c_idx] if c_idx < len(headers) else f"Col{c_idx+1}"
                val = ""
                if cell is not None:
                    txt = extract_cells(tab.textpage, cell, markdown=False)
                    if txt:
                        val = " ".join(txt.split()).strip()
                if val:
                    row_parts.append(f"{h}: {val}")
            if row_parts:
                rows_text.append(" | ".join(row_parts))
    except Exception:
        try:
            raw_rows = tab.extract()
            j_start = 0 if tab.header.external else 1
            for row in raw_rows[j_start:]:
                row_parts = []
                for c_idx, cell in enumerate(row):
                    h = headers[c_idx] if c_idx < len(headers) else f"Col{c_idx+1}"
                    val = " ".join(str(cell).split()).strip() if cell else ""
                    if val:
                        row_parts.append(f"{h}: {val}")
                if row_parts:
                    rows_text.append(" | ".join(row_parts))
        except Exception:
            pass

    return rows_text


def _extract_page_text_with_tables(page: pymupdf.Page) -> str:
    """Extract page text preserving table row structure with column headers.

    Uses PyMuPDF's page.find_tables() to identify tabular regions and serializes
    each row as 'Header: Value | ...'. Non-table text blocks and tables are
    ordered vertically by their page layout positions. Falls back to standard
    get_text() if table detection yields no tables or errors.
    """
    try:
        tabs = page.find_tables()
        if not tabs.tables:
            return clean_text(page.get_text())

        table_lines_map = {}
        for t_idx, tab in enumerate(tabs.tables):
            lines = _serialize_table(tab)
            if lines:
                table_lines_map[t_idx] = "\n".join(lines)

        if not table_lines_map:
            return clean_text(page.get_text())

        items = []
        tables_inserted = set()

        for b in page.get_text("blocks"):
            # b: (x0, y0, x1, y1, text, block_no, block_type)
            if b[6] != 0:
                continue
            b_rect = pymupdf.Rect(b[:4])
            matched_table = None
            for t_idx, tab in enumerate(tabs.tables):
                if t_idx not in table_lines_map:
                    continue
                t_rect = pymupdf.Rect(tab.bbox)
                if b_rect.intersects(t_rect):
                    inter = b_rect & t_rect
                    if b_rect.get_area() > 0 and (inter.get_area() / b_rect.get_area()) > 0.5:
                        matched_table = t_idx
                        break

            if matched_table is None:
                text = b[4].strip()
                if text:
                    items.append((b[1], b[0], text))
            else:
                if matched_table not in tables_inserted:
                    tables_inserted.add(matched_table)
                    t_tab = tabs.tables[matched_table]
                    items.append((t_tab.bbox[1], t_tab.bbox[0], table_lines_map[matched_table]))

        # Include any tables that did not intersect a detected text block
        for t_idx, t_text in table_lines_map.items():
            if t_idx not in tables_inserted:
                t_tab = tabs.tables[t_idx]
                items.append((t_tab.bbox[1], t_tab.bbox[0], t_text))

        items.sort(key=lambda x: (x[0], x[1]))
        full_text = "\n\n".join(it[2] for it in items)
        return clean_text(full_text)
    except Exception as e:
        logger.debug("Table-aware extraction failed, falling back to get_text(): %s", e)
        return clean_text(page.get_text())


def extract_pages_from_pdf(file_path: Union[str, Path]) -> List[Tuple[int, str]]:
    """Extract text from a PDF, preserving 1-indexed page numbers.

    Returns:
        List of (page_number, page_text) tuples.

    Raises:
        ValueError: If PDF contains no extractable text (e.g. empty or scanned image).
    """
    doc = pymupdf.open(str(file_path))
    pages: List[Tuple[int, str]] = []
    total_characters = 0
    ocr_enabled = is_ocr_available()

    try:
        for idx, page in enumerate(doc):
            page_text = _extract_page_text_with_tables(page)

            # If page has sparse or no digital text (e.g. slide titles or scanned images)
            if (not page_text or len(page_text) < 60) and ocr_enabled:
                try:
                    import pytesseract

                    pix = page.get_pixmap(dpi=150)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    ocr_text = clean_text(pytesseract.image_to_string(img))
                    if ocr_text:
                        if page_text and ocr_text != page_text:
                            page_text = f"{page_text}\n{ocr_text}"
                        else:
                            page_text = ocr_text
                except Exception as e:
                    logger.debug("OCR extraction skipped on page %d: %s", idx + 1, e)

            if page_text:
                total_characters += len(page_text)
                pages.append((idx + 1, page_text))
    finally:
        doc.close()

    if total_characters == 0 or not pages:
        raise ValueError(
            "No extractable text found in PDF. The document appears empty or scanned, "
            "and OCR is unavailable or found no text (install Tesseract to enable OCR for scanned pages)."
        )

    return pages