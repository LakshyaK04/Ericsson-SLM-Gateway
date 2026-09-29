"""Chunking module — selectable text-splitting strategies.

Provides character, structure, and semantic chunking behind
a single `chunk_text` dispatcher function.
"""

from .character import character_chunking
from .structure import structure_chunking
from .semantic import semantic_chunking


def chunk_text(
    text: str,
    strategy: str = "character",
) -> list[str]:
    """Select and apply a chunking strategy.

    Args:
        text: The document text to chunk.
        strategy: One of 'character', 'structure', or 'semantic'.

    Returns:
        A list of text chunks.

    Raises:
        ValueError: If the strategy name is not recognised.
    """
    if strategy == "character":
        return character_chunking(text)

    if strategy == "structure":
        return structure_chunking(text)

    if strategy == "semantic":
        return semantic_chunking(text)

    raise ValueError(f"Unknown chunking strategy: {strategy}")


__all__ = [
    "chunk_text",
    "character_chunking",
    "structure_chunking",
    "semantic_chunking",
]
