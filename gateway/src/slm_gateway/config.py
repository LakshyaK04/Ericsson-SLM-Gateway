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

    # Security
    GATEWAY_API_KEY: Optional[str] = None

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

    # Intent router & PII
    ROUTER_THRESHOLD: float = 0.55
    PII_FAIL_MODE: Literal["closed", "open"] = "closed"
    PROJECT_CODENAMES: str = "Project-Titan,Project-Apollo,Project-Odin,Project-Thor,Project-Aegis"
    RAG_SERVICE_URL: str = "http://localhost:8001"

    @property
    def project_codenames_list(self) -> list[str]:
        return [c.strip() for c in self.PROJECT_CODENAMES.split(",") if c.strip()]


# Global settings singleton
settings = Settings()
