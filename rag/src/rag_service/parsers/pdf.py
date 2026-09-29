"""PDF parser using PyMuPDF (fitz).

Extracts text on a per-page basis, preserving page numbers for chunk metadata,
and rejects empty or scanned image-only PDFs without OCR.
"""

from pathlib import Path
from typing import List, Tuple, Union
import pymupdf


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
            if page_text:
                total_characters += len(page_text)
                pages.append((idx + 1, page_text))
    finally:
        doc.close()

    if total_characters == 0 or not pages:
        raise ValueError(
            "No extractable text found in PDF; document appears empty or contains only scanned images (OCR is not supported)."
        )

    return pages