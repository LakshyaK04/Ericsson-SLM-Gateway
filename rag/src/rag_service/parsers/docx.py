"""DOCX document parser using python-docx.

Extracts text, paragraphs, and heading styles from Microsoft Word documents.
"""

from pathlib import Path
from typing import List, Tuple, Union
import docx


def extract_pages_from_docx(file_path: Union[str, Path]) -> List[Tuple[int, str]]:
    """Extract structured text from a DOCX document.

    Returns a list of (page_number, text) tuples. For Word documents without
    fixed pagination, returns content grouped under page 1.

    Raises:
        ValueError: If no extractable text is present in the document.
    """
    doc = docx.Document(str(file_path))
    content_blocks: List[str] = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            # Prefix headings to preserve structural cues
            style_name = para.style.name if para.style else ""
            if "Heading" in style_name or "Title" in style_name:
                content_blocks.append(f"\n## {text}\n")
            else:
                content_blocks.append(text)

    # Also extract any table contents
    for table in doc.tables:
        table_rows = []
        for row in table.rows:
            row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if row_cells:
                table_rows.append(" | ".join(row_cells))
        if table_rows:
            content_blocks.append("\n" + "\n".join(table_rows) + "\n")

    full_text = "\n\n".join(content_blocks).strip()
    if not full_text:
        raise ValueError("No extractable text found in DOCX document; document appears empty.")

    return [(1, full_text)]
