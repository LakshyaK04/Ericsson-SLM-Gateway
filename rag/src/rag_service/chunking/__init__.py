"""Chunking module — selectable text-splitting strategies producing standardized Chunks."""

from typing import Any, Dict, List, Optional, Tuple

from .base import Chunk
from .character import character_chunking, split_character_text
from .structure import structure_chunking, split_structure_text
from .semantic import semantic_chunking, split_semantic_text


def chunk_document(
    pages: List[Tuple[int, str]],
    source: str,
    strategies: Optional[List[str]] = None,
    embedding_model: Any = None,
) -> Dict[str, List[Chunk]]:
    """Chunk a multi-page document under multiple strategies.

    Args:
        pages: List of (page_number, text) tuples.
        source: Name or identifier of the document.
        strategies: List of strategies to apply ('character', 'structure', 'semantic').
        embedding_model: Pre-loaded embedding model needed for 'semantic' chunking.

    Returns:
        Dict mapping strategy name to list of Chunk objects.
    """
    target_strategies = strategies or ["character", "structure", "semantic"]
    results: Dict[str, List[Chunk]] = {strat: [] for strat in target_strategies}

    for page_num, page_text in pages:
        if not page_text or not page_text.strip():
            continue

        if "character" in target_strategies:
            char_chunks = character_chunking(
                text=page_text,
                source=source,
                page=page_num,
            )
            results["character"].extend(char_chunks)

        if "structure" in target_strategies:
            struct_chunks = structure_chunking(
                text=page_text,
                source=source,
                page=page_num,
            )
            results["structure"].extend(struct_chunks)

        if "semantic" in target_strategies:
            if embedding_model is None:
                raise ValueError("Embedding model is required for semantic chunking.")
            sem_chunks = semantic_chunking(
                text=page_text,
                embedding_model=embedding_model,
                source=source,
                page=page_num,
            )
            results["semantic"].extend(sem_chunks)

    return results


__all__ = [
    "Chunk",
    "chunk_document",
    "character_chunking",
    "structure_chunking",
    "semantic_chunking",
    "split_character_text",
    "split_structure_text",
    "split_semantic_text",
]
