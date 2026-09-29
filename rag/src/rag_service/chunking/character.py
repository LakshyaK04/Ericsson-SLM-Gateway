"""Character-based chunking strategy.

Splits text into fixed-size overlapping chunks, snapping boundaries
to whitespace so words are not cut in half, returning standardized Chunk objects.
"""

from typing import List
from .base import Chunk


def split_character_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[str]:
    """Split text into raw string chunks snapping to whitespace."""
    if not text or not text.strip():
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size

        if end < text_len:
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