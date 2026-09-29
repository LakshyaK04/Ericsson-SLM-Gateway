"""Semantic chunking strategy.

Splits text at sentence boundaries where the embedding similarity
between neighbouring sentences drops below a threshold, subject to
a minimum chunk size to preserve context.
"""

import re

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def semantic_chunking(
    text: str,
    threshold: float = 0.65,
    min_chunk_size: int = 300,
    model: SentenceTransformer | None = None,
) -> list[str]:
    """Split text at semantic boundaries.

    Args:
        text: Input text to chunk.
        threshold: Cosine similarity below which a split occurs.
        min_chunk_size: Minimum characters before allowing a split.
        model: Optional pre-loaded SentenceTransformer to avoid
            reloading the model on every call.
    """
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]

    if not sentences:
        return []

    if len(sentences) == 1:
        return sentences

    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model.encode(
        sentences,
        normalize_embeddings=True,
    )

    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        similarity = cosine_similarity(
            [embeddings[i - 1]],
            [embeddings[i]],
        )[0][0]

        current_text = " ".join(current_chunk)

        # Only split if:
        # 1. There is a semantic boundary (low similarity)
        # 2. Current chunk is already large enough
        if (
            similarity < threshold
            and len(current_text) >= min_chunk_size
        ):
            chunks.append(current_text)
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
