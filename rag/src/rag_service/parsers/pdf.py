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
from typing import List, Optional, Tuple, Union
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
            page_text = clean_text(page.get_text())

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