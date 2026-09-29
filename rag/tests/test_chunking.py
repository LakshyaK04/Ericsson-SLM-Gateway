"""Unit tests for character, structure, and semantic chunking strategies."""

import pytest
from rag_service.chunking.base import Chunk
from rag_service.chunking.character import character_chunking, split_character_text
from rag_service.chunking.structure import structure_chunking, split_structure_text
from rag_service.chunking.semantic import semantic_chunking, split_semantic_text
from rag_service.chunking import chunk_document


FIXED_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Artificial intelligence is transforming telecommunications networks across the globe. "
    "5G standalone core networks introduce cloud-native microservices for user plane functions. "
    "Edge computing brings compute resources closer to enterprise radio access nodes. "
    "Zero-touch automation enables autonomous self-healing cellular clusters."
)

STRUCTURED_TEXT = (
    "# 1. Executive Summary\n"
    "This document outlines the cloud-native transition for telecommunications systems.\n\n"
    "## 2. Architecture Overview\n"
    "The 5G core utilizes service-based interfaces using HTTP/2 and JSON serialization. "
    "Network repository functions maintain service registrations.\n\n"
    "### 2.1 User Plane Function\n"
    "The UPF processes packet forwarding at line rate with hardware acceleration.\n\n"
    "3. SECURITY PROTOCOLS\n"
    "Mutual TLS authentication is enforced across all internal microservice communications."
)


def test_character_chunking_snaps_whitespace_and_overlaps():
    """Character chunking must snap to word boundaries and maintain overlap without empty chunks."""
    chunk_size = 80
    overlap = 20
    chunks = split_character_text(FIXED_TEXT, chunk_size=chunk_size, overlap=overlap)

    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= chunk_size + 10  # approximate boundary
        assert len(c) > 0  # no empty chunks
        # Boundary should not end with broken trailing partial words if space was found
        assert not c.startswith(" ")
        assert not c.endswith(" ")

    # Assert consecutive chunks share overlapping tokens
    chunk_objects = character_chunking(FIXED_TEXT, source="test.pdf", page=1, chunk_size=chunk_size, overlap=overlap)
    assert len(chunk_objects) == len(chunks)
    assert all(isinstance(c, Chunk) for c in chunk_objects)
    assert all(c.strategy == "character" for c in chunk_objects)
    assert all(c.page == 1 for c in chunk_objects)
    assert all(c.source == "test.pdf" for c in chunk_objects)


def test_structure_chunking_splits_on_headings_and_merges():
    """Structure chunking splits on Markdown, numbered, and ALL CAPS headings."""
    chunks = split_structure_text(STRUCTURED_TEXT, max_chunk_size=150)

    assert len(chunks) >= 3
    # Check that headings remain at the beginning of sections
    has_exec_summary = any("Executive Summary" in c for c in chunks)
    has_arch = any("Architecture Overview" in c for c in chunks)
    has_upf = any("User Plane Function" in c for c in chunks)
    has_sec = any("SECURITY PROTOCOLS" in c for c in chunks)

    assert has_exec_summary
    assert has_arch
    assert has_upf
    assert has_sec

    # No empty chunks
    assert all(len(c.strip()) > 0 for c in chunks)

    # Test Chunk dataclass output
    chunk_objects = structure_chunking(STRUCTURED_TEXT, source="arch.docx", page=2, max_chunk_size=150)
    assert len(chunk_objects) == len(chunks)
    assert all(c.strategy == "structure" for c in chunk_objects)
    assert all(c.page == 2 for c in chunk_objects)


def test_structure_chunking_splits_oversized_section():
    """When a single section exceeds max_chunk_size, it must split at sentence boundaries."""
    long_section = "Sentence one is informative. " * 30  # ~870 chars
    chunks = split_structure_text(long_section, max_chunk_size=300)

    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 350
        assert c.endswith(".")


class MockEmbeddingModel:
    """Fast mock embedding model for unit testing semantic chunking."""
    def encode_documents(self, sentences):
        # Deterministic 4-dim embeddings: first 2 sentences close, 3rd sentence distant
        import numpy as np
        emb = []
        for i, s in enumerate(sentences):
            if i < 2:
                emb.append([1.0, 0.0, 0.0, 0.0])
            else:
                emb.append([0.0, 1.0, 0.0, 0.0])
        return emb


def test_semantic_chunking_splits_at_boundary():
    """Semantic chunking splits when similarity drops below threshold and min size is reached."""
    text = (
        "This is the first sentence discussing cellular architecture. "
        "This is the second sentence continuing the cellular discussion. "
        "Suddenly this sentence discusses medieval history and castles. "
        "Castles were fortified structures built in Europe during the Middle Ages."
    )
    mock_model = MockEmbeddingModel()
    chunks = semantic_chunking(
        text=text,
        embedding_model=mock_model,
        source="doc1.pdf",
        page=1,
        threshold=0.5,
        min_chunk_size=50,
    )

    assert len(chunks) == 2
    assert "cellular" in chunks[0].text
    assert "castles" in chunks[1].text
    assert all(c.strategy == "semantic" for c in chunks)


def test_chunk_document_dispatcher():
    """Multi-page chunk_document dispatcher correctly produces all requested strategies."""
    pages = [
        (1, "Page 1 intro text. More content here."),
        (2, "Page 2 # Heading\nSecond page details."),
    ]
    mock_model = MockEmbeddingModel()
    results = chunk_document(
        pages=pages,
        source="manual.pdf",
        strategies=["character", "structure", "semantic"],
        embedding_model=mock_model,
    )

    assert "character" in results
    assert "structure" in results
    assert "semantic" in results
    assert len(results["character"]) > 0
    assert len(results["structure"]) > 0
    assert len(results["semantic"]) > 0
