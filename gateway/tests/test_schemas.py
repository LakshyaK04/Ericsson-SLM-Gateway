"""Tests for OpenAI-compatible Pydantic schemas in the SLM Gateway."""

import pytest
from pydantic import ValidationError

from slm_gateway.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    ChoiceMessage,
    ModelCard,
    ModelListResponse,
    Usage,
)


def test_valid_chat_completion_request():
    """Verify standard valid request succeeds with defaults."""
    req = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="Hello")]
    )
    assert req.temperature == 0.7
    assert req.top_p == 1.0
    assert req.max_tokens == 512
    assert req.stream is False
    assert len(req.messages) == 1
    assert req.messages[0].role == "user"
    assert req.messages[0].content == "Hello"


def test_empty_messages_raises_validation_error():
    """Empty messages list must fail schema validation (yielding 422)."""
    with pytest.raises(ValidationError) as exc_info:
        ChatCompletionRequest(messages=[])
    errors = exc_info.value.errors()
    assert any("messages" in error["loc"] for error in errors)


def test_invalid_temperature_bounds():
    """Temperature must be constrained between 0.0 and 2.0."""
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Hi")],
            temperature=-0.5,
        )

    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Hi")],
            temperature=2.5,
        )


def test_invalid_top_p_bounds():
    """top_p must be between 0.0 and 1.0."""
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Hi")],
            top_p=1.5,
        )


def test_chat_completion_response_shape():
    """Response must contain all standard OpenAI fields including usage."""
    resp = ChatCompletionResponse(
        model="microsoft/Phi-3-mini-4k-instruct",
        choices=[
            Choice(
                index=0,
                message=ChoiceMessage(role="assistant", content="Test reply"),
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=15,
            completion_tokens=5,
            total_tokens=20,
        ),
    )

    assert resp.id.startswith("chatcmpl-")
    assert resp.object == "chat.completion"
    assert isinstance(resp.created, int)
    assert resp.model == "microsoft/Phi-3-mini-4k-instruct"
    assert len(resp.choices) == 1
    assert resp.choices[0].message.role == "assistant"
    assert resp.choices[0].message.content == "Test reply"
    assert resp.choices[0].finish_reason == "stop"
    assert resp.usage.prompt_tokens == 15
    assert resp.usage.completion_tokens == 5
    assert resp.usage.total_tokens == 20


def test_model_list_response():
    """GET /v1/models response must match OpenAI list structure."""
    resp = ModelListResponse(
        data=[
            ModelCard(id="microsoft/Phi-3-mini-4k-instruct")
        ]
    )
    assert resp.object == "list"
    assert len(resp.data) == 1
    assert resp.data[0].id == "microsoft/Phi-3-mini-4k-instruct"
    assert resp.data[0].object == "model"
    assert resp.data[0].owned_by == "local"
