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


@pytest.fixture(autouse=True)
def setup_mock_backend(monkeypatch):
    """Ensure a fast mock backend is attached to the app for unit tests."""
    mock = MockBackend(ready=True)
    import slm_gateway.main as main_mod
    monkeypatch.setattr(main_mod, "backend", mock)
    app.state.backend = mock
    yield mock


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
