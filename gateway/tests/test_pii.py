"""Unit and integration tests for PII redaction engine and gateway integration."""

import pytest
from fastapi.testclient import TestClient

from slm_gateway.config import Settings, settings
from slm_gateway.main import app
from slm_gateway.pii import PIIRedactionError, PIIRedactor, redact_pii
from slm_gateway.backends.base import LLMBackend


class RecordingMockBackend(LLMBackend):
    """Mock backend that captures the exact messages received for inference."""

    def __init__(self):
        self.received_messages = []
        self._ready = True

    async def load(self) -> None:
        pass

    def is_ready(self) -> bool:
        return self._ready

    def get_model_name(self) -> str:
        return "recording-mock"

    async def generate(
        self,
        messages,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 512,
    ):
        self.received_messages = list(messages)
        return "Sanitized response", 10, 5, "stop"

    async def generate_stream(
        self,
        messages,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 512,
    ):
        self.received_messages = list(messages)
        for token in ["Sanitized ", "response"]:
            yield token

    async def close(self) -> None:
        pass


@pytest.fixture
def redactor():
    cfg = Settings(
        PII_FAIL_MODE="closed",
    )
    return PIIRedactor(cfg)


# ============================================================
# 1. Table-Driven Tests for Each Entity Type
# ============================================================

@pytest.mark.parametrize(
    "input_text,expected_placeholder,entity_type",
    [
        ("My email is alice@ericsson.com please reply.", "<EMAIL_ADDRESS>", "EMAIL_ADDRESS"),
        ("Call me at +1-555-123-4567 tomorrow.", "<PHONE_NUMBER>", "PHONE_NUMBER"),
        ("Pay with card 4532-0151-1283-0366 now.", "<CREDIT_CARD>", "CREDIT_CARD"),
        ("Connect to 192.168.1.100 port 22.", "<IP_ADDRESS>", "IP_ADDRESS"),
        ("Employee EMP-12345 reported for duty.", "<EMPLOYEE_ID>", "EMPLOYEE_ID"),
        ("Employee EMP-9876543 submitted the report.", "<EMPLOYEE_ID>", "EMPLOYEE_ID"),
        ("Project Phoenix is scheduled for delivery next quarter.", "<PROJECT_CODENAME>", "PROJECT_CODENAME"),
        ("The Titan initiative has been approved by the board.", "<PROJECT_CODENAME>", "PROJECT_CODENAME"),
    ],
)
def test_pii_entity_redaction(redactor, input_text, expected_placeholder, entity_type):
    """Verify that each target entity is detected and replaced with typed placeholder."""
    redacted_text, count = redactor.redact(input_text)
    assert count >= 1, f"Expected at least 1 redaction for {entity_type} in '{input_text}'"
    assert expected_placeholder in redacted_text, f"Expected {expected_placeholder} in '{redacted_text}'"


def test_person_redaction(redactor):
    """Verify person names are redacted."""
    text = "My name is John Doe and I work at Ericsson."
    redacted, count = redactor.redact(text)
    assert count >= 1
    assert "<PERSON>" in redacted
    assert "John Doe" not in redacted


# ============================================================
# 2. No-False-Positive Tests (Locations, Dates, Normal Text)
# ============================================================

@pytest.mark.parametrize(
    "query",
    [
        "What is the capital of Germany?",
        "Explain photosynthesis in plants.",
        "How do I sort a list in Python?",
        "The conference will take place on Monday at 3 PM in Stockholm.",
        "What does an HTTP 404 status code mean?",
    ],
)
def test_no_false_positives_on_normal_queries(redactor, query):
    """Ensure normal queries, locations, and dates are NOT redacted."""
    redacted, count = redactor.redact(query)
    assert count == 0, f"Expected 0 redactions for query '{query}', but got {count}: '{redacted}'"
    assert redacted == query, f"Expected unchanged text, got: {redacted}"


# ============================================================
# 3. Fail-Closed vs Fail-Open Behavior Tests
# ============================================================

def test_fail_closed_raises_error_when_redaction_fails(monkeypatch):
    """In fail-closed mode, errors during redaction must raise PIIRedactionError."""
    cfg = Settings(PII_FAIL_MODE="closed")
    failing_redactor = PIIRedactor(cfg)

    def mock_analyze_fail(*args, **kwargs):
        raise RuntimeError("Presidio internal crash")

    monkeypatch.setattr(failing_redactor.analyzer, "analyze", mock_analyze_fail)

    with pytest.raises(PIIRedactionError):
        failing_redactor.redact("Contact me at test@example.com")


def test_fail_open_passes_through_on_error(monkeypatch):
    """In fail-open mode, errors during redaction log warning and return raw text."""
    cfg = Settings(PII_FAIL_MODE="open")
    open_redactor = PIIRedactor(cfg)

    def mock_analyze_fail(*args, **kwargs):
        raise RuntimeError("Presidio internal crash")

    monkeypatch.setattr(open_redactor.analyzer, "analyze", mock_analyze_fail)

    text = "Contact me at test@example.com"
    redacted, count = open_redactor.redact(text)
    assert count == 0
    assert redacted == text


# ============================================================
# 4. End-to-End Test: Backend Never Receives Raw PII
# ============================================================

def test_end_to_end_gateway_pii_redaction(monkeypatch):
    """Verify gateway masks user messages before passing payload to LLM backend."""
    mock_backend = RecordingMockBackend()
    import slm_gateway.main as main_mod

    monkeypatch.setattr(main_mod, "backend", mock_backend)
    app.state.backend = mock_backend

    client = TestClient(app, raise_server_exceptions=False)

    sensitive_content = (
        "Hello, my email is john.doe@ericsson.com, "
        "my phone is +1-555-123-4567, and employee ID is EMP-12345."
    )

    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": sensitive_content},
        ]
    }

    response = client.post(
        "/v1/chat/completions",
        json=payload,
        headers={"X-Bypass-Router": "true"},
    )
    assert response.status_code == 200
    data = response.json()

    # Verify redaction count in namespaced x_pii extension
    assert "x_pii" in data
    assert data["x_pii"]["redactions"] >= 3

    # CRITICAL: Verify the backend NEVER saw the raw PII!
    assert len(mock_backend.received_messages) == 2
    user_msg = mock_backend.received_messages[1]
    assert user_msg["role"] == "user"
    received_text = user_msg["content"]

    # Raw secrets must NOT be present
    assert "john.doe@ericsson.com" not in received_text
    assert "+1-555-123-4567" not in received_text
    assert "EMP-12345" not in received_text

    # Typed placeholders must be present
    assert "<EMAIL_ADDRESS>" in received_text
    assert "<PHONE_NUMBER>" in received_text
    assert "<EMPLOYEE_ID>" in received_text



def test_gateway_fail_closed_returns_500(monkeypatch):
    """If PII redaction fails in fail-closed mode, gateway returns HTTP 500 error."""
    mock_backend = RecordingMockBackend()
    import slm_gateway.main as main_mod

    monkeypatch.setattr(main_mod, "backend", mock_backend)
    app.state.backend = mock_backend

    # Force redactor to throw
    class FailingRedactor:
        def redact(self, *args, **kwargs):
            raise PIIRedactionError("Simulated Presidio failure")

    failing_instance = FailingRedactor()
    monkeypatch.setattr(main_mod, "pii_redactor", failing_instance)
    app.state.pii_redactor = failing_instance

    client = TestClient(app, raise_server_exceptions=False)
    payload = {
        "messages": [
            {"role": "user", "content": "Call me at +1-555-123-4567"}
        ]
    }

    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 500
    data = response.json()
    assert "error" in data
    assert data["error"]["type"] == "pii_redaction_error"
    # Backend must NOT have received any messages
    assert len(mock_backend.received_messages) == 0
