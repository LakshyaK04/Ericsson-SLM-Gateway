"""Structure-based chunking strategy.

Splits text by headings and paragraph breaks, merges small pieces,
and splits oversized ones.
"""

import re


def structure_chunking(
    text: str,
    max_chunk_size: int = 1000,
) -> list[str]:
    """Split text using document structure (headings, paragraphs).

    Splits on lines that look like headings (all-caps, title-case,
    or numbered headings) and on blank-line paragraph breaks.
    Small consecutive pieces are merged up to max_chunk_size.
    Oversized pieces are split at sentence boundaries.
    """
    # Split on heading-like lines or double newlines
    # Matches: numbered headings (1. Foo), ALL CAPS lines,
    # title-case lines, or lines followed by blank lines
    pattern = r"\n(?=(?:\d+[\.\)]\s+)?[A-Z][A-Za-z0-9 &:,\-]{2,80}\n)"
    raw_sections = re.split(pattern, text)

    # Also split on double newlines (paragraph breaks)
    sections = []
    for section in raw_sections:
        parts = re.split(r"\n\s*\n", section)
        sections.extend(parts)

    # Filter empty sections and strip whitespace
    sections = [s.strip() for s in sections if s.strip()]

    # Merge small consecutive sections, split oversized ones
    chunks = []
    buffer = ""

    for section in sections:
        combined = (buffer + "\n\n" + section).strip() if buffer else section

        if len(combined) <= max_chunk_size:
            buffer = combined
        else:
            # Flush buffer if it has content
            if buffer:
                chunks.append(buffer)
            # If this section alone is oversized, split at sentences
            if len(section) > max_chunk_size:
                chunks.extend(
                    _split_long_section(section, max_chunk_size)
                )
                buffer = ""
            else:
                buffer = section

    if buffer:
        chunks.append(buffer)

    return chunks


def _split_long_section(
    text: str,
    max_size: int,
) -> list[str]:
    """Split an oversized section at sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = ""

    for sentence in sentences:
        candidate = (current + " " + sentence).strip() if current else sentence
        if len(candidate) <= max_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)

    return chunks
