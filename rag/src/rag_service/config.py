"""Configuration for the RAG Service."""

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Service configuration
    SERVICE_NAME: str = "rag-service"
    HOST: str = "0.0.0.0"
    PORT: int = 8001
    DEBUG: bool = False
    RAG_API_KEY: Optional[str] = None  # Optional Bearer key for upload/delete/admin endpoints
    CORS_ALLOW_ORIGINS: str = "http://localhost:8000,http://localhost:8001"

    # Upload & request size limits
    MAX_UPLOAD_SIZE_BYTES: int = 26_214_400  # 25MB max file upload
    MAX_QUERY_LENGTH: int = 4096            # Max characters for retrieval query

    # In-memory rate limiting per client IP
    RATE_LIMIT_ENABLED: bool = False
    RATE_LIMIT_REQUESTS: int = 120
    RATE_LIMIT_WINDOW_SECONDS: int = 60

    # Ingestion PII redaction (optional)
    PII_REDACTION_ON_INGEST: bool = False
    PROJECT_CODENAMES: Optional[str] = None

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.CORS_ALLOW_ORIGINS.split(",") if o.strip()]

    # Storage paths
    CHROMA_PERSIST_DIR: str = "data/chroma"

    # Models per Section 5.5
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-base"

    # Retrieval parameters
    DEFAULT_STRATEGY: str = "structure"
    RETRIEVE_K: int = 20
    FINAL_K: int = 3

    # Gateway connection for Phase 6
    GATEWAY_URL: str = "http://localhost:8000"
    GATEWAY_MODEL: str = "microsoft/Phi-3-mini-4k-instruct"
    GATEWAY_TIMEOUT: float = 180.0

    # OCR configuration
    TESSERACT_CMD: Optional[str] = None


settings = Settings()
