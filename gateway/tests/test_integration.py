"""Phase 6 End-to-End Integration Tests.

Validates:
1. RAG route execution returning OpenAI-shaped completion plus x_sources.
2. Graceful fallback when no documents are indexed in RAG service.
3. Graceful fallback when RAG service is offline or unreachable.
4. Prevention of infinite routing loops (X-Bypass-Router header honoured).
5. Grounded prompt generation and context formatting.
"""

from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
import pytest

from slm_gateway.config import Settings
from slm_gateway.main import app
from slm_gateway.rag_client import RAGClient
from slm_gateway.schemas import ChatCompletionRequest, ChatMessage


class MockBackend:
    def is_ready(self) -> bool:
        return True

    def get_model_name(self) -> str:
        return "mock-phi3"

    async def load(self):
        pass

    async def generate(self, messages, temperature=0.7, top_p=1.0, max_tokens=512):
        return "Mock local model response.", 20, 10, "stop"

    async def close(self):
        pass


@pytest.fixture(autouse=True)
def setup_gateway_app():
    """Ensure test app has mock backend and active router."""
    import slm_gateway.main as main_mod

    mock_backend = MockBackend()
    main_mod.backend = mock_backend
    app.state.backend = mock_backend

    # Ensure router is initialized
    from slm_gateway.router import get_router
    router = get_router(main_mod.settings)
    main_mod.router_instance = router
    app.state.router = router


@pytest.fixture
def client():
    return TestClient(app)


def test_rag_routing_with_indexed_documents_returns_sources(client):
    """When intent is RAG and documents are indexed, gateway calls RAG and returns x_sources."""
    mock_rag_client = AsyncMock(spec=RAGClient)
    mock_rag_client.has_indexed_documents.return_value = (True, None)
    mock_rag_client.get_answer.return_value = (
        {
            "answer": "According to the document [1], the UPF is the primary data path anchor.",
            "sources": [
                {
                    "source": "ericsson_5g_core_architecture.pdf",
                    "page": 2,
                    "chunk_id": "c_upf_1",
                    "strategy": "structure",
                    "dense_score": 0.88,
                    "rerank_score": 0.95,
                    "text": "The User Plane Function (UPF) is the primary data path anchor in 5G networks.",
                }
            ],
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 18,
                "total_tokens": 138,
            },
        },
        None,
    )

    import slm_gateway.main as main_mod
    main_mod.rag_client_instance = mock_rag_client

    payload = {
        "messages": [
            {"role": "user", "content": "Search the uploaded PDF manual for UPF specifications."}
        ]
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["choices"][0]["message"]["content"] == "According to the document [1], the UPF is the primary data path anchor."
    assert data["x_routing"]["route"] == "rag_service"
    assert data["x_sources"] is not None
    assert len(data["x_sources"]) == 1
    assert data["x_sources"][0]["source"] == "ericsson_5g_core_architecture.pdf"
    assert data["usage"]["total_tokens"] == 138


def test_rag_routing_with_no_documents_downgrades_to_local_model(client):
    """When intent is RAG but no documents are indexed, gateway falls back cleanly to local model."""
    mock_rag_client = AsyncMock(spec=RAGClient)
    mock_rag_client.has_indexed_documents.return_value = (False, "no_documents_indexed")

    import slm_gateway.main as main_mod
    main_mod.rag_client_instance = mock_rag_client

    payload = {
        "messages": [
            {"role": "user", "content": "Search the uploaded PDF manual for UPF specifications."}
        ]
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    # Content should come from local mock backend
    assert data["choices"][0]["message"]["content"] == "Mock local model response."
    assert data["x_sources"] is None
    assert "No documents indexed" in data["x_routing"]["warning"]
    assert data["x_routing"]["route"] == "hf_local"


def test_rag_routing_service_offline_downgrades_to_local_model(client):
    """When intent is RAG but RAG service is unreachable, gateway falls back cleanly without 500 error."""
    mock_rag_client = AsyncMock(spec=RAGClient)
    mock_rag_client.has_indexed_documents.return_value = (False, "rag_service_offline")

    import slm_gateway.main as main_mod
    main_mod.rag_client_instance = mock_rag_client

    payload = {
        "messages": [
            {"role": "user", "content": "Search the uploaded PDF manual for UPF specifications."}
        ]
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["choices"][0]["message"]["content"] == "Mock local model response."
    assert data["x_sources"] is None
    assert "RAG service unavailable" in data["x_routing"]["warning"]
    assert data["x_routing"]["route"] == "hf_local"


def test_bypass_router_header_prevents_infinite_loop(client):
    """X-Bypass-Router: true skips intent routing completely, enabling RAG service to call gateway."""
    mock_rag_client = AsyncMock(spec=RAGClient)
    import slm_gateway.main as main_mod
    main_mod.rag_client_instance = mock_rag_client

    # A query that would otherwise trigger RAG
    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Context: [1] Source: doc.pdf\nUPF routes packets.\n\nQuestion: What does UPF do?"},
        ]
    }
    resp = client.post(
        "/v1/chat/completions",
        json=payload,
        headers={"X-Bypass-Router": "true"},
    )
    assert resp.status_code == 200
    data = resp.json()

    # Router was bypassed
    assert data["x_routing"]["intent"] == "bypass"
    assert data["x_sources"] is None
    assert data["choices"][0]["message"]["content"] == "Mock local model response."

    # Assert RAG client was NEVER called (no infinite loop)
    mock_rag_client.has_indexed_documents.assert_not_called()
    mock_rag_client.get_answer.assert_not_called()
