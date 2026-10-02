"""OpenAI-compatible request and response schemas for SLM Gateway."""

import time
import uuid
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message author (e.g. system, user, assistant)")
    content: str = Field(..., description="Contents of the message")


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = Field(default=None, description="ID of the model to use")
    messages: List[ChatMessage] = Field(..., description="List of messages comprising the conversation so far")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature between 0 and 2")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Nucleus sampling probability")
    max_tokens: int = Field(default=512, ge=1, description="Maximum number of tokens to generate")
    stream: bool = Field(default=False, description="Whether to stream back partial progress")

    @field_validator("messages")
    @classmethod
    def validate_messages_non_empty(cls, v: List[ChatMessage]) -> List[ChatMessage]:
        if not v:
            raise ValueError("messages list must not be empty")
        return v


class ChoiceMessage(BaseModel):
    role: str = "assistant"
    content: str


class Choice(BaseModel):
    index: int = 0
    message: ChoiceMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[Choice]
    usage: Usage

    # Namespaced extension fields per build plan
    x_routing: Optional[Dict[str, Any]] = None
    x_pii: Optional[Dict[str, Any]] = None
    x_sources: Optional[List[Dict[str, Any]]] = None


class ChatCompletionChunkDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChunkChoice]
    x_routing: Optional[Dict[str, Any]] = None
    x_pii: Optional[Dict[str, Any]] = None
    x_sources: Optional[List[Dict[str, Any]]] = None


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "local"


class ModelListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: List[ModelCard]


class OpenAIError(BaseModel):
    message: str
    type: str = "invalid_request_error"
    code: Optional[int] = None


class OpenAIErrorResponse(BaseModel):
    error: OpenAIError
