from .ingestion import (
    extract_text_from_pdf,
    clean_text,
)
from .chunking import chunk_text
from .embeddings import EmbeddingModel
from .vector_store import VectorStore


def test_retrieval():
    text = extract_text_from_pdf("data/ericsson_rag_sample.pdf")
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

    vector_store.add(
        chunk_embeddings,
        chunks,
    )

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
