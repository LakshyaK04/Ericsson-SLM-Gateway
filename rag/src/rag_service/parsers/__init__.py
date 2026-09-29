"""Document parser module supporting PDF and DOCX formats."""

from pathlib import Path
from typing import List, Optional, Tuple, Union

from .docx import extract_pages_from_docx
from .pdf import extract_pages_from_pdf


def parse_document(
    file_path: Union[str, Path],
    filename: Optional[str] = None,
) -> List[Tuple[int, str]]:
    """Parse a document file into (page_number, text) segments.

    Supported extensions: .pdf, .docx

    Raises:
        ValueError: If file extension is unsupported or extraction fails.
    """
    path = Path(file_path)
    name = filename or path.name
    ext = Path(name).suffix.lower()

    if ext == ".pdf":
        return extract_pages_from_pdf(path)
    elif ext == ".docx":
        return extract_pages_from_docx(path)
    else:
        raise ValueError(
            f"Unsupported file format '{ext}'. Only .pdf and .docx documents are supported."
        )


__all__ = [
    "parse_document",
    "extract_pages_from_pdf",
    "extract_pages_from_docx",
]
