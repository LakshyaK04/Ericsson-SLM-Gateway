"""Configuration for the SLM Gateway.

All settings are loaded from environment variables or a .env file,
using pydantic-settings.
"""

from typing import Literal, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Service configuration
    SERVICE_NAME: str = "slm-gateway"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    GATEWAY_API_KEY: Optional[str] = None  # Optional Bearer key enforcement per spec
    CORS_ALLOW_ORIGINS: str = "http://localhost:8000,http://localhost:8001"

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.CORS_ALLOW_ORIGINS.split(",") if o.strip()]

    # LLM Backend
    BACKEND: Literal["hf_local", "openai_compatible"] = "hf_local"
    BACKEND_URL: str = "http://localhost:8000"
    BACKEND_TIMEOUT_SECONDS: float = 120.0

    # Local model configuration (hf_local)
    MODEL_ID: str = "microsoft/Phi-3-mini-4k-instruct"
    QUANTIZE: Literal["4bit", "none"] = "4bit"
    DEVICE: str = "auto"
    MAX_CONTEXT_LENGTH: int = 4096

    # Default generation parameters
    DEFAULT_MAX_TOKENS: int = 512
    DEFAULT_TEMPERATURE: float = 0.7
    DEFAULT_TOP_P: float = 1.0

    # Security & limits
    MAX_REQUEST_BODY_BYTES: int = 1_048_576  # 1MB max JSON body
    MAX_REQUEST_MESSAGES: int = 100         # Max conversation turns per request
    MAX_MESSAGE_CHARS: int = 32_768         # Max character count per message (~8k tokens)

    # In-memory rate limiting per client IP
    RATE_LIMIT_ENABLED: bool = False
    RATE_LIMIT_REQUESTS: int = 60
    RATE_LIMIT_WINDOW_SECONDS: int = 60

    # In-process inference queue limits
    INFERENCE_MAX_QUEUE_SIZE: int = 10
    INFERENCE_QUEUE_TIMEOUT_SECONDS: float = 30.0

    # Intent router & PII
    ROUTER_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    ROUTER_THRESHOLD: float = 0.55
    INTENTS_FILE: Optional[str] = None
    PII_FAIL_MODE: Literal["closed", "open"] = "closed"
    PROJECT_CODENAMES: Optional[str] = None  # Comma-separated deny-list, e.g. "Phoenix,Titan,Aurora"
    RAG_SERVICE_URL: str = "http://localhost:8001"
    RAG_API_KEY: Optional[str] = None        # Optional Bearer token when authenticating to RAG service
    RAG_TIMEOUT_SECONDS: float = 180.0
    RAG_DEFAULT_STRATEGY: str = "structure"


# Global settings singleton
settings = Settings()

