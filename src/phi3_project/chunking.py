import re

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def character_chunking(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
):
    """
    Split text into fixed-size overlapping character chunks.
    """
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])

        start += chunk_size - overlap

    return chunks


def structure_chunking(text: str):
    """
    Split text using document structure such as headings
    and paragraphs.
    """
    sections = re.split(
        r"\n(?=[A-Z][A-Za-z0-9 &-]{2,50}\n)",
        text,
    )

    chunks = [
        section.strip()
        for section in sections
        if section.strip()
    ]

    return chunks


def semantic_chunking(
    text: str,
    threshold: float = 0.65,
    min_chunk_size: int = 300,
):
    """
    Split text at semantic boundaries while keeping
    chunks large enough to preserve context.
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
        # 1. There is a semantic boundary
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


def chunk_text(
    text: str,
    strategy: str = "character",
):
    """
    Select a chunking strategy.
    """
    if strategy == "character":
        return character_chunking(text)

    if strategy == "structure":
        return structure_chunking(text)

    if strategy == "semantic":
        return semantic_chunking(text)

    raise ValueError(
        f"Unknown chunking strategy: {strategy}"
    )