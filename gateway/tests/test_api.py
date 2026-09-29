"""API route and authentication tests using a mock LLM backend for speed."""

import pytest
from fastapi.testclient import TestClient

from slm_gateway.config import settings
from slm_gateway.main import app
from slm_gateway.backends.base import LLMBackend


class MockBackend(LLMBackend):
    def __init__(self, ready: bool = True):
        self._ready = ready
        self._loaded = True

    async def load(self) -> None:
        self._loaded = True

    def is_ready(self) -> bool:
        return self._ready and self._loaded

    def get_model_name(self) -> str:
        return "mock-phi3-mini"

    async def generate(
        self,
        messages,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 512,
    ):
        return "Mock response from backend", 12, 6, "stop"

    async def close(self) -> None:
        self._loaded = False


class MockRouter:
    def __init__(self, ready: bool = True):
        self.ready = ready

    def classify(self, query: str):
        from slm_gateway.router import RoutingResult
        return RoutingResult(
            intent="general",
            confidence=0.95,
            route="hf_local",
            latency_ms=1.2,
            scores_by_intent={"general": 0.95, "technical": 0.4, "structured_json": 0.3, "rag": 0.2},
            fallback_applied=False,
        )


@pytest.fixture(autouse=True)
def setup_mock_backend(monkeypatch):
    """Ensure a fast mock backend and mock router are attached to the app for unit tests."""
    mock_b = MockBackend(ready=True)
    mock_r = MockRouter(ready=True)
    import slm_gateway.main as main_mod
    monkeypatch.setattr(main_mod, "backend", mock_b)
    monkeypatch.setattr(main_mod, "router_instance", mock_r)
    app.state.backend = mock_b
    app.state.router = mock_r
    yield mock_b


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


def test_health_endpoint(client):
    """GET /health must return 200 healthy."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "healthy",
        "service": "slm-gateway",
    }


def test_ready_endpoint_when_loaded(client):
    """GET /ready returns 200 when backend is ready."""
    resp = client.get("/ready")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ready"
    assert data["backend"] == "mock-phi3-mini"


def test_ready_endpoint_when_not_ready(client, monkeypatch):
    """GET /ready returns 503 when backend is not ready."""
    mock = MockBackend(ready=False)
    import slm_gateway.main as main_mod
    monkeypatch.setattr(main_mod, "backend", mock)
    app.state.backend = mock

    resp = client.get("/ready")
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == 503


def test_models_endpoint(client):
    """GET /v1/models returns OpenAI list format."""
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == "mock-phi3-mini"
    assert data["data"][0]["object"] == "model"


def test_chat_completions_success(client):
    """POST /v1/chat/completions returns expected OpenAI-compatible JSON."""
    payload = {
        "model": "mock-phi3-mini",
        "messages": [
            {"role": "user", "content": "Explain telemetry"}
        ],
        "temperature": 0.5,
        "max_tokens": 100,
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["id"].startswith("chatcmpl-")
    assert data["object"] == "chat.completion"
    assert isinstance(data["created"], int)
    assert data["model"] == "mock-phi3-mini"

    # Choices
    assert len(data["choices"]) == 1
    choice = data["choices"][0]
    assert choice["index"] == 0
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] == "Mock response from backend"
    assert choice["finish_reason"] == "stop"

    # Usage
    assert data["usage"]["prompt_tokens"] == 12
    assert data["usage"]["completion_tokens"] == 6
    assert data["usage"]["total_tokens"] == 18


def test_chat_completions_empty_messages_validation(client):
    """Empty messages array yields 422 with OpenAI error format."""
    payload = {
        "messages": [],
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 422
    data = resp.json()
    assert "error" in data
    assert data["error"]["code"] == 422
    assert data["error"]["type"] == "invalid_request_error"


def test_chat_completions_streaming_rejected(client):
    """stream=True must return 400 with a clear message per spec."""
    payload = {
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 400
    data = resp.json()
    assert "error" in data
    assert "streaming is not supported" in data["error"]["message"].lower()


def test_auth_disabled_by_default(client, monkeypatch):
    """When GATEWAY_API_KEY is unset, requests without Authorization succeed."""
    monkeypatch.setattr(settings, "GATEWAY_API_KEY", None)
    payload = {"messages": [{"role": "user", "content": "hello"}]}
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200


def test_auth_enforced_when_key_configured(client, monkeypatch):
    """When GATEWAY_API_KEY is set, valid Bearer token is strictly required."""
    monkeypatch.setattr(settings, "GATEWAY_API_KEY", "ericsson-secret-key-123")
    payload = {"messages": [{"role": "user", "content": "hello"}]}

    # 1. Missing header -> 401
    resp_no_auth = client.post("/v1/chat/completions", json=payload)
    assert resp_no_auth.status_code == 401
    assert resp_no_auth.json()["error"]["type"] == "authentication_error"

    # 2. Wrong token -> 401
    resp_wrong = client.post(
        "/v1/chat/completions",
        json=payload,
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert resp_wrong.status_code == 401

    # 3. Correct token -> 200
    resp_correct = client.post(
        "/v1/chat/completions",
        json=payload,
        headers={"Authorization": "Bearer ericsson-secret-key-123"},
    )
    assert resp_correct.status_code == 200


def test_chat_completions_includes_routing_metadata(client):
    """Chat completions response must include x_routing metadata."""
    payload = {"messages": [{"role": "user", "content": "What is Python?"}]}
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "x_routing" in data
    assert data["x_routing"]["intent"] == "general"
    assert "confidence" in data["x_routing"]
    assert "route" in data["x_routing"]
    assert "latency_ms" in data["x_routing"]


def test_chat_completions_bypass_router_header(client):
    """X-Bypass-Router header skips semantic routing."""
    payload = {"messages": [{"role": "user", "content": "What is Python?"}]}
    resp = client.post(
        "/v1/chat/completions",
        json=payload,
        headers={"X-Bypass-Router": "true"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["x_routing"] == {
        "intent": "bypass",
        "confidence": 1.0,
        "route": "hf_local",
        "latency_ms": 0.0,
    }


def test_structured_json_intent_injects_system_prompt(client, monkeypatch):
    """When router classifies as structured_json, a strict JSON system prompt is injected."""
    from unittest.mock import AsyncMock
    from slm_gateway.router import RoutingResult

    class JsonMockRouter:
        def classify(self, query: str):
            return RoutingResult(
                intent="structured_json",
                confidence=0.88,
                route="hf_local",
                latency_ms=2.0,
            )

    import slm_gateway.main as main_mod
    monkeypatch.setattr(main_mod, "router_instance", JsonMockRouter())

    # Spy on backend.generate to verify received payload
    received_messages = []
    original_generate = main_mod.backend.generate

    async def spy_generate(messages, **kwargs):
        received_messages.extend(messages)
        return "{}", 10, 5, "stop"

    monkeypatch.setattr(main_mod.backend, "generate", spy_generate)

    payload = {"messages": [{"role": "user", "content": "Return data as JSON"}]}
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    assert len(received_messages) == 2
    assert received_messages[0]["role"] == "system"
    assert "valid JSON" in received_messages[0]["content"]
    assert received_messages[1]["role"] == "user"

