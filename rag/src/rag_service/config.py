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


settings = Settings()
