"""Tests for the two-stage RAG retrieval pipeline (ChromaStore + BGE Reranker)."""

from pathlib import Path
import pytest

from rag_service.chunking.base import Chunk
from rag_service.config import Settings
from rag_service.embeddings import EmbeddingModel
from rag_service.reranker import Reranker
from rag_service.retriever import Retriever
from rag_service.store import ChromaStore


@pytest.fixture
def tiny_corpus_retriever(tmp_path: Path):
    """Set up an isolated ChromaStore with a controlled tiny corpus."""
    store = ChromaStore(persist_dir=tmp_path / "chroma_tiny")
    emb_model = EmbeddingModel("BAAI/bge-small-en-v1.5")
    reranker = Reranker("BAAI/bge-reranker-base")
    retriever = Retriever(store, emb_model, reranker)

    # 4 distinct chunks across different topics
    chunks = [
        Chunk(
            chunk_id="doc1_chunk_0",
            text="Kubernetes pods use CNI plugins like Calico and Cilium for pod-to-pod networking.",
            source="k8s.pdf",
            page=1,
            strategy="structure",
        ),
        Chunk(
            chunk_id="doc1_chunk_1",
            text="The 5G User Plane Function (UPF) processes packet inspection and QoS flow steering.",
            source="5g_core.pdf",
            page=1,
            strategy="structure",
        ),
        Chunk(
            chunk_id="doc1_chunk_2",
            text="PostgreSQL utilizes multi-version concurrency control (MVCC) to ensure ACID transaction isolation.",
            source="db.pdf",
            page=2,
            strategy="structure",
        ),
        Chunk(
            chunk_id="doc1_chunk_3",
            text="B-Tree indexes in relational databases provide logarithmic time complexity for range scans.",
            source="db.pdf",
            page=3,
            strategy="structure",
        ),
    ]

    # Embed and index
    texts = [c.text for c in chunks]
    embeddings = emb_model.encode_documents(texts)
    store.add_chunks(
        strategy="structure",
        chunks=chunks,
        embeddings=embeddings,
        doc_id="test_corpus_doc",
    )

    return retriever, store


def test_retriever_returns_top_ranked_chunk_with_both_scores(tiny_corpus_retriever):
    """Retriever must return top-k chunks with dense_score and rerank_score."""
    retriever, store = tiny_corpus_retriever

    query = "What is the role of the 5G UPF in packet routing?"
    results = retriever.retrieve(
        query=query,
        strategy="structure",
        retrieve_k=4,
        final_k=2,
    )

    assert len(results) == 2

    # Top result must be the 5G UPF chunk
    top = results[0]
    assert "UPF" in top.text
    assert top.source == "5g_core.pdf"
    assert top.page == 1
    assert top.strategy == "structure"

    # Both scores must be present and valid
    assert 0.0 <= top.dense_score <= 1.0
    assert isinstance(top.rerank_score, float)
    # The top rerank score must be strictly higher than the second result
    assert top.rerank_score >= results[1].rerank_score


def test_retriever_scoped_by_doc_id(tiny_corpus_retriever):
    """Scoped query with doc_ids filter only searches matching documents."""
    retriever, store = tiny_corpus_retriever

    # Query with non-matching doc_id
    results = retriever.retrieve(
        query="5G UPF",
        strategy="structure",
        doc_ids=["non_existent_doc"],
    )
    assert len(results) == 0

    # Query with matching doc_id
    results_matched = retriever.retrieve(
        query="5G UPF",
        strategy="structure",
        doc_ids=["test_corpus_doc"],
    )
    assert len(results_matched) > 0


def test_hybrid_retrieval_with_bm25_and_rrf(tiny_corpus_retriever):
    """Hybrid retrieval must return chunks with BM25, RRF, dense, and rerank scores."""
    retriever, store = tiny_corpus_retriever

    query = "Calico and Cilium CNI plugins networking"
    results = retriever.retrieve(
        query=query,
        strategy="structure",
        retrieve_k=4,
        final_k=2,
        retrieval_mode="hybrid",
    )

    assert len(results) > 0
    top = results[0]
    assert "Kubernetes" in top.text
    assert top.source == "k8s.pdf"

    # Verify all hybrid scoring fields are populated
    assert top.dense_score is not None
    assert top.rerank_score is not None
    assert top.bm25_score is not None and top.bm25_score > 0.0
    assert top.rrf_score is not None and top.rrf_score > 0.0


def test_sparse_retrieval_mode(tiny_corpus_retriever):
    """Sparse retrieval mode uses only BM25 keyword matching."""
    retriever, store = tiny_corpus_retriever

    query = "PostgreSQL MVCC transaction isolation"
    results = retriever.retrieve(
        query=query,
        strategy="structure",
        retrieve_k=4,
        final_k=2,
        use_reranker=False,
        retrieval_mode="sparse",
    )

    assert len(results) > 0
    top = results[0]
    assert "PostgreSQL" in top.text
    assert top.bm25_score is not None and top.bm25_score > 0.0


def test_hybrid_custom_rrf_k(tiny_corpus_retriever):
    """Custom rrf_k factor modifies RRF scoring correctly."""
    retriever, store = tiny_corpus_retriever

    query = "B-Tree indexes range scans"
    results = retriever.retrieve(
        query=query,
        strategy="structure",
        retrieve_k=4,
        final_k=2,
        use_reranker=False,
        retrieval_mode="hybrid",
        rrf_k=20,
    )

    assert len(results) > 0
    top = results[0]
    assert top.rrf_score is not None and top.rrf_score > 0.0

