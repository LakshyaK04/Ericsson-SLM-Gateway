"""Tests for the RAG retrieval pipeline."""

import os

import pytest

from rag_service.parsers import extract_text_from_pdf, clean_text
from rag_service.chunking import chunk_text
from rag_service.embeddings import EmbeddingModel
from rag_service.store import VectorStore


# Resolve the sample PDF path relative to the repo root
SAMPLE_PDF = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "eval", "docs", "ericsson_rag_sample.pdf",
)


@pytest.mark.slow
def test_retrieval():
    """End-to-end retrieval: ingest a PDF, embed, search, rerank."""
    text = extract_text_from_pdf(SAMPLE_PDF)
    text = clean_text(text)

    chunks = chunk_text(text)

    assert len(chunks) > 0

    embedding_model = EmbeddingModel()
    chunk_embeddings = embedding_model.encode(chunks)

    assert chunk_embeddings.shape[0] == len(chunks)
    assert chunk_embeddings.shape[1] == 384

    vector_store = VectorStore(
        dimension=chunk_embeddings.shape[1]
    )

    vector_store.add(chunk_embeddings, chunks)

    query = "How does RAG retrieve information from documents?"
    query_embedding = embedding_model.encode([query])

    results = vector_store.search(
        query_embedding,
        query,
        top_k=2,
    )

    assert len(results) == 2
    assert "text" in results[0]
    assert "faiss_score" in results[0]
    assert "reranker_score" in results[0]

    assert (
        "RAG" in results[0]["text"]
        or "retriev" in results[0]["text"].lower()
    )
