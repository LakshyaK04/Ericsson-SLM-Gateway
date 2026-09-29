"""Character-based chunking strategy.

Splits text into fixed-size overlapping chunks, snapping boundaries
to whitespace so words are not cut in half.
"""


def character_chunking(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """Split text into fixed-size overlapping character chunks.

    Boundaries snap to the nearest whitespace to avoid cutting
    words in half.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Snap to whitespace so we don't split mid-word
        if end < len(text):
            # Look backwards from 'end' for the nearest space
            snap = text.rfind(" ", start, end)
            if snap > start:
                end = snap

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move forward by (end - overlap), but at least 1 char
        step = max(end - start - overlap, 1)
        start += step

    return chunks