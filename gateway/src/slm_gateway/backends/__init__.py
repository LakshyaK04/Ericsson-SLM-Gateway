"""Backend factory and module exports."""

from ..config import Settings
from .base import LLMBackend
from .hf_local import HFLocalBackend, InferenceQueueFullError, InferenceTimeoutError
from .openai_compat import OpenAICompatibleBackend


def get_backend(config: Settings) -> LLMBackend:
    """Instantiate the configured LLM backend."""
    if config.BACKEND == "hf_local":
        return HFLocalBackend(config)
    elif config.BACKEND == "openai_compatible":
        return OpenAICompatibleBackend(config)
    else:
        raise ValueError(f"Unknown backend type: {config.BACKEND}")


__all__ = [
    "LLMBackend",
    "HFLocalBackend",
    "OpenAICompatibleBackend",
    "InferenceQueueFullError",
    "InferenceTimeoutError",
    "get_backend",
]
