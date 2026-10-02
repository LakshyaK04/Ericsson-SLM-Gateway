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

    async def generate_stream(
        self,
        messages,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 512,
    ):
        for token in ["Mock ", "streaming ", "response"]:
            yield token

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
            scores_by_intent={"general": 0.95, "technical": 0.4, "rag": 0.2},
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


def test_chat_completions_streaming_success(client):
    """stream=True returns Server-Sent Events (text/event-stream) with OpenAI chunk format."""
    payload = {
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    lines = [line.strip() for line in resp.text.split("\n") if line.strip()]
    assert len(lines) >= 3
    assert lines[-1] == "data: [DONE]"

    # Parse and validate chunks
    import json
    chunks = []
    for line in lines[:-1]:
        assert line.startswith("data: ")
        chunk = json.loads(line[6:])
        chunks.append(chunk)

    # First chunk contains metadata
    assert chunks[0]["object"] == "chat.completion.chunk"
    assert "x_routing" in chunks[0]
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"

    # Accumulated stream content matches backend tokens
    content = "".join([c["choices"][0]["delta"].get("content") or "" for c in chunks])
    assert "Mock streaming response" in content

    # Stop chunk
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


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


def test_gateway_api_key_enforcement(client, monkeypatch):
    """When GATEWAY_API_KEY is configured, enforce Bearer authentication."""
    monkeypatch.setattr(settings, "GATEWAY_API_KEY", "secret-test-key")

    payload = {"messages": [{"role": "user", "content": "Hello"}]}

    # 1. Missing header -> 401
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 401
    assert "error" in resp.json()
    assert "Expected 'Bearer <key>'" in resp.json()["error"]["message"]

    # 2. Invalid key -> 401
    resp = client.post(
        "/v1/chat/completions",
        json=payload,
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert resp.status_code == 401
    assert "Incorrect API key" in resp.json()["error"]["message"]

    # 3. Valid key -> 200
    resp = client.post(
        "/v1/chat/completions",
        json=payload,
        headers={"Authorization": "Bearer secret-test-key"},
    )
    assert resp.status_code == 200

    # 4. Check /v1/models route auth
    resp = client.get("/v1/models", headers={"Authorization": "Bearer secret-test-key"})
    assert resp.status_code == 200
    resp_no_auth = client.get("/v1/models")
    assert resp_no_auth.status_code == 401


def test_prometheus_metrics_endpoint(client):
    """GET /metrics returns valid OpenMetrics Prometheus text format."""
    # Send a request to generate metrics
    payload = {"messages": [{"role": "user", "content": "What is Python?"}]}
    client.post("/v1/chat/completions", json=payload)

    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
    text = resp.text

    assert "gateway_active_requests" in text
    assert "gateway_requests_total" in text
    assert "gateway_tokens_total" in text
    assert "gateway_request_duration_seconds" in text




