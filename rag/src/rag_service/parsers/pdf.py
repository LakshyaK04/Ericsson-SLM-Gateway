"""PDF parser using PyMuPDF (fitz) with automatic Tesseract OCR fallback.

Extracts text on a per-page basis, preserving page numbers for chunk metadata.
If a page contains sparse or no digital text (e.g. scanned slides, images),
it automatically falls back to Tesseract OCR when available.
"""

import io
import logging
from pathlib import Path
import shutil
from typing import List, Tuple, Union
from PIL import Image
import pymupdf

logger = logging.getLogger(__name__)

# Check Tesseract availability
_TESSERACT_AVAILABLE = False
try:
    import pytesseract

    if not shutil.which("tesseract"):
        win_path = Path("C:/Program Files/Tesseract-OCR/tesseract.exe")
        if win_path.exists():
            pytesseract.pytesseract.tesseract_cmd = str(win_path)
    pytesseract.get_tesseract_version()
    _TESSERACT_AVAILABLE = True
except Exception:
    _TESSERACT_AVAILABLE = False


def clean_text(text: str) -> str:
    """Clean extracted page text for downstream chunking."""
    lines = [line.strip() for line in text.splitlines()]
    non_empty = [line for line in lines if line]
    return "\n".join(non_empty)


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

    try:
        for idx, page in enumerate(doc):
            page_text = clean_text(page.get_text())

            # If page has sparse or no digital text (e.g. slide titles or scanned images)
            if (not page_text or len(page_text) < 60) and _TESSERACT_AVAILABLE:
                try:
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
            "No extractable text found in PDF; document appears empty or contains only scanned images (OCR is not supported or yielded no text)."
        )

    return pages