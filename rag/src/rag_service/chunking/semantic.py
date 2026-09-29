"""Semantic chunking strategy.

Splits text at sentence boundaries where the embedding similarity between
neighboring sentences drops below a threshold, subject to a minimum chunk size
to preserve context, using BAAI/bge-small-en-v1.5.
"""

import re
from typing import Any, List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .base import Chunk


def split_semantic_text(
    text: str,
    embedding_model: Any,
    threshold: float = 0.65,
    min_chunk_size: int = 300,
) -> List[str]:
    """Split text into string chunks at semantic similarity dips."""
    if not text or not text.strip():
        return []

    sentences = [
        s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()
    ]

    if not sentences:
        return []

    if len(sentences) == 1:
        return sentences

    # Embed all sentences with the shared embedding model
    if hasattr(embedding_model, "encode_documents"):
        embeddings = embedding_model.encode_documents(sentences)
    elif hasattr(embedding_model, "encode"):
        embeddings = embedding_model.encode(sentences, normalize_embeddings=True)
    elif hasattr(embedding_model, "model"):
        embeddings = embedding_model.model.encode(sentences, normalize_embeddings=True)
    else:
        raise ValueError("Invalid embedding model provided to semantic_chunking.")

    chunks: List[str] = []
    current_chunk: List[str] = [sentences[0]]

    for i in range(1, len(sentences)):
        # Compute cosine similarity between adjacent sentence embeddings
        sim = float(
            cosine_similarity(
                np.array([embeddings[i - 1]]),
                np.array([embeddings[i]]),
            )[0][0]
        )

        current_text = " ".join(current_chunk)

        # Split condition: similarity below threshold AND chunk has met minimum size
        if sim < threshold and len(current_text) >= min_chunk_size:
            chunks.append(current_text)
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def semantic_chunking(
    text: str,
    embedding_model: Any,
    source: str = "document",
    page: int = 1,
    threshold: float = 0.65,
    min_chunk_size: int = 300,
) -> List[Chunk]:
    """Produce standardized Chunk dataclasses using semantic similarity dips."""
    raw_chunks = split_semantic_text(
        text=text,
        embedding_model=embedding_model,
        threshold=threshold,
        min_chunk_size=min_chunk_size,
    )
    return [
        Chunk(
            chunk_id=f"{source}_sem_p{page}_{idx}",
            text=chunk_str,
            source=source,
            page=page,
            strategy="semantic",
        )
        for idx, chunk_str in enumerate(raw_chunks)
    ]
