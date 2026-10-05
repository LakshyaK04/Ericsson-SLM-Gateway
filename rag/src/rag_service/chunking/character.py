"""Character-based chunking strategy.

Splits text into fixed-size overlapping chunks, snapping boundaries
to whitespace so words are not cut in half, returning standardized Chunk objects.
"""

from typing import List

from .base import Chunk


def _is_table_line(line: str) -> bool:
    """Check if a line represents a serialized table row."""
    return " | " in line or (line.startswith("|") and line.endswith("|"))


def split_character_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[str]:
    """Split text into raw string chunks snapping to whitespace without cutting table rows."""
    if not text or not text.strip():
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size

        if end < text_len:
            # If end cuts across a table row, snap to before the row to avoid splitting it
            last_nl = text.rfind("\n", start, end)
            if last_nl > start:
                next_nl = text.find("\n", end)
                cutting_line = text[last_nl + 1 : next_nl if next_nl != -1 else text_len]
                if _is_table_line(cutting_line):
                    end = last_nl
                else:
                    snap = text.rfind(" ", start, end)
                    if snap > start:
                        end = snap
            else:
                # Look backwards from 'end' for the nearest whitespace
                snap = text.rfind(" ", start, end)
                if snap > start:
                    end = snap

        chunk_str = text[start:end].strip()
        if chunk_str:
            chunks.append(chunk_str)

        step = max(end - start - overlap, 1)
        start += step

    return chunks


def character_chunking(
    text: str,
    source: str = "document",
    page: int = 1,
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[Chunk]:
    """Produce standardized Chunk dataclasses using character-based splitting."""
    raw_chunks = split_character_text(text, chunk_size=chunk_size, overlap=overlap)
    return [
        Chunk(
            chunk_id=f"{source}_char_p{page}_{idx}",
            text=chunk_str,
            source=source,
            page=page,
            strategy="character",
        )
        for idx, chunk_str in enumerate(raw_chunks)
    ]
