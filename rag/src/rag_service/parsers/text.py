"""Plain text and Markdown document parser."""

from pathlib import Path
from typing import List, Tuple, Union


def extract_pages_from_text(file_path: Union[str, Path]) -> List[Tuple[int, str]]:
    """Parse a .txt or .md file into (page_number, text) tuples.

    Treats form-feed characters (\\x0c) as page breaks if present,
    otherwise treats the entire file as a single page (page 1).

    Raises:
        ValueError: If file is empty or contains no extractable text.
    """
    path = Path(file_path)
    if not path.exists():
        raise ValueError(f"File not found: {path}")

    # Read with UTF-8 first, fallback to latin-1
    content = ""
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = path.read_text(encoding="latin-1")

    cleaned = content.strip()
    if not cleaned:
        raise ValueError(f"No extractable text found in file '{path.name}'.")

    # If form feed is used, split on pages
    if "\x0c" in content:
        raw_pages = content.split("\x0c")
        pages = []
        page_num = 1
        for p in raw_pages:
            t = p.strip()
            if t:
                pages.append((page_num, t))
                page_num += 1
        if not pages:
            raise ValueError(f"No extractable text found in file '{path.name}'.")
        return pages

    return [(1, cleaned)]
