"""Abstract base class for LLM backends."""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple


class LLMBackend(ABC):
    """Abstract interface for SLM Gateway model backends."""

    @abstractmethod
    async def load(self) -> None:
        """Initialize and load the model into memory / setup client."""
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        """Check if backend is ready to serve inference requests."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the loaded model identifier."""
        pass

    @abstractmethod
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 512,
    ) -> Tuple[str, int, int, str]:
        """Execute chat completion.

        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            temperature: Sampling temperature (0.0 to 2.0).
            top_p: Nucleus sampling probability.
            max_tokens: Maximum new tokens to generate.

        Returns:
            Tuple of (generated_text, prompt_tokens, completion_tokens, finish_reason).
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Release resources on application shutdown."""
        pass
