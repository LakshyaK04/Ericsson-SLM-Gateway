"""Structure-based chunking strategy.

Splits text by headings and paragraph breaks, merges small pieces,
and splits oversized ones at sentence boundaries, returning standardized Chunk objects.
"""

import re
from typing import List

from .base import Chunk

# Robust heading pattern:
# 1. Markdown style: # Heading, ## Section
# 2. Numbered outline: 1. Introduction, 1.2 Architecture, Section 3
# 3. Capitalized titles: ALL CAPS or Title Case headers on a line by themselves
HEADING_REGEX = re.compile(
    r"(?m)^(?:"
    r"#{1,6}\s+.+|"  # Markdown headings
    r"(?:Section\s+)?(?:\d+\.)+\d*\s+[A-Z].+|"  # Numbered sections: 1. Introduction, 1.1 Scope
    r"[A-Z0-9\s_\-:,]{3,60}$"  # ALL CAPS or short headers
    r")"
)


def _split_long_section(text: str, max_size: int) -> List[str]:
    """Split an oversized text block at sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: List[str] = []
    current = ""

    for sentence in sentences:
        candidate = (current + " " + sentence).strip() if current else sentence
        if len(candidate) <= max_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(sentence) > max_size:
                # Fallback word-wrap split for massive single sentences
                words = sentence.split()
                sub_curr = ""
                for w in words:
                    if len(sub_curr) + len(w) + 1 <= max_size:
                        sub_curr = (sub_curr + " " + w).strip()
                    else:
                        if sub_curr:
                            chunks.append(sub_curr)
                        sub_curr = w
                if sub_curr:
                    chunks.append(sub_curr)
                current = ""
            else:
                current = sentence

    if current:
        chunks.append(current)

    return chunks


def split_structure_text(
    text: str,
    max_chunk_size: int = 1000,
) -> List[str]:
    """Split text along structural headings and paragraphs into string chunks."""
    if not text or not text.strip():
        return []

    # First split on double newlines to isolate paragraphs
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    # Further break down any paragraph containing embedded headings
    raw_sections: List[str] = []
    for para in paragraphs:
        lines = para.split("\n")
        current_block: List[str] = []
        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue
            if HEADING_REGEX.match(line_str) and current_block:
                raw_sections.append("\n".join(current_block))
                current_block = [line_str]
            else:
                current_block.append(line_str)
        if current_block:
            raw_sections.append("\n".join(current_block))

    # Merge small consecutive sections up to max_chunk_size
    chunks: List[str] = []
    buffer = ""

    for sec in raw_sections:
        combined = (buffer + "\n\n" + sec).strip() if buffer else sec

        if len(combined) <= max_chunk_size:
            buffer = combined
        else:
            if buffer:
                chunks.append(buffer)
            if len(sec) > max_chunk_size:
                chunks.extend(_split_long_section(sec, max_chunk_size))
                buffer = ""
            else:
                buffer = sec

    if buffer:
        chunks.append(buffer)

    return chunks


def structure_chunking(
    text: str,
    source: str = "document",
    page: int = 1,
    max_chunk_size: int = 1000,
) -> List[Chunk]:
    """Produce standardized Chunk dataclasses using structural boundaries."""
    raw_chunks = split_structure_text(text, max_chunk_size=max_chunk_size)
    return [
        Chunk(
            chunk_id=f"{source}_struct_p{page}_{idx}",
            text=chunk_str,
            source=source,
            page=page,
            strategy="structure",
        )
        for idx, chunk_str in enumerate(raw_chunks)
    ]
