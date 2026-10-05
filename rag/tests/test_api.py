"""API route and lifecycle tests for the RAG service."""

from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient
from rag_service.config import settings
from rag_service.embeddings import EmbeddingModel
from rag_service.main import app
from rag_service.reranker import Reranker
from rag_service.retriever import Retriever
from rag_service.store import ChromaStore


@pytest.fixture(scope="module")
def shared_models(tmp_path_factory):
    """Load lightweight models and isolated ChromaStore once for the test module."""
    test_dir = tmp_path_factory.mktemp("chroma_test")
    test_store = ChromaStore(persist_dir=test_dir)
    emb_model = EmbeddingModel("BAAI/bge-small-en-v1.5")
    reranker = Reranker("BAAI/bge-reranker-base")
    retriever = Retriever(test_store, emb_model, reranker, settings)

    import rag_service.main as main_mod
    import rag_service.retriever as ret_mod
    import rag_service.store as store_mod

    store_mod._store = test_store
    ret_mod._retriever = retriever
    main_mod.store = test_store
    main_mod.embedding_model = emb_model
    main_mod.reranker = reranker
    main_mod.retriever = retriever

    app.state.store = test_store
    app.state.embedding_model = emb_model
    app.state.reranker = reranker
    app.state.retriever = retriever

    yield {
        "store": test_store,
        "embedding_model": emb_model,
        "reranker": reranker,
        "retriever": retriever,
    }

    store_mod._store = None
    ret_mod._retriever = None
    main_mod.store = None
    main_mod.embedding_model = None
    main_mod.reranker = None
    main_mod.retriever = None


@pytest.fixture
def client(shared_models):
    return TestClient(app)


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    pdf_path = tmp_path / "telecom_overview.pdf"
    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_text(
        (50, 50),
        "5G Core Architecture: The User Plane Function (UPF) is responsible for packet routing "
        "and forwarding, policy enforcement, and lawful interception.",
    )
    p2 = doc.new_page()
    p2.insert_text(
        (50, 50),
        "Radio Access Network: Massive MIMO beamforming enhances cell-edge throughput "
        "by focusing radiated RF energy directly toward active user equipment.",
    )
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def empty_pdf(tmp_path: Path) -> Path:
    pdf_path = tmp_path / "blank.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


def test_health_endpoint(client):
    """GET /health returns healthy status and collection names."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["service"] == "rag-service"
    assert "collections" in data


def test_ready_endpoint(client):
    """GET /ready returns 200 when components are initialized."""
    resp = client.get("/ready")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ready"
    assert data["service"] == "rag-service"


def test_upload_invalid_extension(client, tmp_path: Path):
    """POST /documents with non-pdf/docx file returns 400."""
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("Hello world")

    with open(txt_file, "rb") as f:
        resp = client.post(
            "/documents",
            files={"file": ("test.txt", f, "text/plain")},
        )
    assert resp.status_code == 400
    assert "Unsupported file format" in resp.json()["detail"]


def test_upload_empty_pdf_returns_400(client, empty_pdf: Path):
    """POST /documents with an empty PDF returns 400 when no text is found."""
    with open(empty_pdf, "rb") as f:
        resp = client.post(
            "/documents",
            files={"file": ("blank.pdf", f, "application/pdf")},
        )
    assert resp.status_code == 400
    assert "No extractable text found in PDF" in resp.json()["detail"]
    assert "OCR is unavailable or found no text" in resp.json()["detail"]


def test_document_ingestion_query_and_deletion_lifecycle(client, sample_pdf: Path):
    """Full lifecycle: upload document, verify chunking, query top chunks, and delete."""
    # 1. Upload valid PDF
    with open(sample_pdf, "rb") as f:
        upload_resp = client.post(
            "/documents",
            files={"file": ("telecom_overview.pdf", f, "application/pdf")},
            data={"strategies": "character,structure,semantic"},
        )
    assert upload_resp.status_code == 200
    upload_data = upload_resp.json()
    doc_id = upload_data["doc_id"]
    assert doc_id.startswith("doc_")
    assert upload_data["filename"] == "telecom_overview.pdf"
    assert "character" in upload_data["chunk_counts"]
    assert "structure" in upload_data["chunk_counts"]
    assert "semantic" in upload_data["chunk_counts"]
    assert upload_data["total_chunks"] > 0

    # 2. List documents
    list_resp = client.get("/documents")
    assert list_resp.status_code == 200
    list_data = list_resp.json()
    assert list_data["total_documents"] >= 1
    found = any(d["doc_id"] == doc_id for d in list_data["documents"])
    assert found

    # 3. Query documents
    query_payload = {
        "query": "What are the responsibilities of the UPF in 5G core?",
        "strategy": "structure",
        "retrieve_k": 10,
        "final_k": 2,
    }
    query_resp = client.post("/query", json=query_payload)
    assert query_resp.status_code == 200
    query_data = query_resp.json()
    assert query_data["strategy"] == "structure"
    assert len(query_data["results"]) >= 1

    top_chunk = query_data["results"][0]
    assert "text" in top_chunk
    assert "dense_score" in top_chunk
    assert "rerank_score" in top_chunk
    assert "UPF" in top_chunk["text"] or "User Plane Function" in top_chunk["text"]

    # 4. Delete document
    delete_resp = client.delete(f"/documents/{doc_id}")
    assert delete_resp.status_code == 200
    del_data = delete_resp.json()
    assert del_data["status"] == "deleted"
    assert del_data["doc_id"] == doc_id
    assert del_data["chunks_removed"] > 0

    # 5. Verify deletion
    verify_resp = client.get("/documents")
    assert not any(d["doc_id"] == doc_id for d in verify_resp.json()["documents"])

    # 6. Deleting non-existent doc returns 404
    del_again = client.delete(f"/documents/{doc_id}")
    assert del_again.status_code == 404


def test_rag_api_key_protection_on_documents(client, monkeypatch):
    """When RAG_API_KEY is configured, /documents endpoints require valid Bearer token."""
    from rag_service.config import settings

    monkeypatch.setattr(settings, "RAG_API_KEY", "secret-rag-key-123")

    # 1. GET /documents without auth -> 401
    r_no_auth = client.get("/documents")
    assert r_no_auth.status_code == 401

    # 2. GET /documents with invalid key -> 401
    r_bad_auth = client.get("/documents", headers={"Authorization": "Bearer wrong-key"})
    assert r_bad_auth.status_code == 401

    # 3. GET /documents with valid key -> 200
    r_good = client.get("/documents", headers={"Authorization": "Bearer secret-rag-key-123"})
    assert r_good.status_code == 200

    # 4. DELETE without auth -> 401
    r_del = client.delete("/documents/doc_123")
    assert r_del.status_code == 401

    # 5. POST /documents without auth -> 401
    r_post = client.post("/documents")
    assert r_post.status_code == 401


def test_ingestion_pii_redaction(client, monkeypatch, tmp_path: Path):
    """When PII_REDACTION_ON_INGEST=True, uploaded documents have PII stripped prior to chunking."""
    from rag_service.config import settings

    monkeypatch.setattr(settings, "PII_REDACTION_ON_INGEST", True)

    pdf_path = tmp_path / "sensitive_memo.pdf"
    doc = pymupdf.open()
    p = doc.new_page()
    p.insert_text(
        (50, 50),
        "Confidential Project Titan: Contact lead engineer at alice.smith@ericsson.com or call 555-019-2834.",
    )
    doc.save(str(pdf_path))
    doc.close()

    with open(pdf_path, "rb") as f:
        resp = client.post(
            "/documents",
            files={"file": ("sensitive_memo.pdf", f, "application/pdf")},
            data={"strategies": "structure"},
        )
    assert resp.status_code == 200
    doc_id = resp.json()["doc_id"]

    try:
        # Query for the ingested content
        query_resp = client.post(
            "/query", json={"query": "Who is the lead engineer?", "strategy": "structure"}
        )
        assert query_resp.status_code == 200
        results = query_resp.json()["results"]
        assert len(results) > 0

        chunk_text = results[0]["text"]
        # Raw PII must NOT appear in chunk text
        assert "alice.smith@ericsson.com" not in chunk_text
        assert "555-019-2834" not in chunk_text
        assert "Titan" not in chunk_text

        # Redacted tokens must appear
        assert "<EMAIL_ADDRESS>" in chunk_text
        assert "<PHONE_NUMBER>" in chunk_text
        assert "<PROJECT_CODENAME>" in chunk_text
    finally:
        client.delete(f"/documents/{doc_id}")


def test_content_hash_deduplication(client, sample_pdf: Path):
    """Uploading the same document twice returns HTTP 409 Conflict with clear detail."""
    # First upload succeeds
    with open(sample_pdf, "rb") as f:
        resp1 = client.post(
            "/documents",
            files={"file": ("telecom_overview.pdf", f, "application/pdf")},
            data={"strategies": "structure"},
        )
    assert resp1.status_code == 200
    doc_id = resp1.json()["doc_id"]

    try:
        # Second upload with identical content returns 409
        with open(sample_pdf, "rb") as f:
            resp2 = client.post(
                "/documents",
                files={"file": ("telecom_overview.pdf", f, "application/pdf")},
                data={"strategies": "structure"},
            )
        assert resp2.status_code == 409
        assert "Document with identical content already indexed" in resp2.json()["detail"]
        assert doc_id in resp2.json()["detail"]
    finally:
        client.delete(f"/documents/{doc_id}")


def test_txt_and_md_ingestion_lifecycle(client, tmp_path: Path):
    """Verify .txt and .md files can be uploaded, indexed, and retrieved."""
    # 1. Upload .txt file
    txt_path = tmp_path / "cloud_native.txt"
    txt_path.write_text(
        "Cloud Native Infrastructure: Kubernetes coordinates container deployment across edge nodes."
    )
    with open(txt_path, "rb") as f:
        resp_txt = client.post(
            "/documents",
            files={"file": ("cloud_native.txt", f, "text/plain")},
            data={"strategies": "structure"},
        )
    assert resp_txt.status_code == 200
    txt_doc_id = resp_txt.json()["doc_id"]

    # 2. Upload .md file
    md_path = tmp_path / "observability.md"
    md_path.write_text(
        "# Observability Guide\nPrometheus scrapes OpenTelemetry metrics every 15 seconds."
    )
    with open(md_path, "rb") as f:
        resp_md = client.post(
            "/documents",
            files={"file": ("observability.md", f, "text/markdown")},
            data={"strategies": "structure"},
        )
    assert resp_md.status_code == 200
    md_doc_id = resp_md.json()["doc_id"]

    try:
        # Query for txt content
        q1 = client.post(
            "/query", json={"query": "Kubernetes container deployment", "strategy": "structure"}
        )
        assert q1.status_code == 200
        assert any("Kubernetes" in r["text"] for r in q1.json()["results"])

        # Query for md content
        q2 = client.post(
            "/query", json={"query": "Prometheus metric scraping interval", "strategy": "structure"}
        )
        assert q2.status_code == 200
        assert any("Prometheus" in r["text"] for r in q2.json()["results"])
    finally:
        client.delete(f"/documents/{txt_doc_id}")
        client.delete(f"/documents/{md_doc_id}")
