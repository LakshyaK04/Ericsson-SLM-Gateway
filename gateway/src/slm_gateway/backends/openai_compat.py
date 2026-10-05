"""OpenAI-compatible HTTP client backend."""

import json
import logging
from typing import AsyncIterator, Dict, List, Tuple

import httpx

from ..config import Settings
from .base import LLMBackend

logger = logging.getLogger(__name__)


class OpenAICompatibleBackend(LLMBackend):
    """Backend that forwards chat completions to an external OpenAI-compatible server."""

    def __init__(self, config: Settings):
        self.config = config
        base_url = config.BACKEND_URL.rstrip("/")
        if not base_url.endswith("/v1"):
            self.endpoint = f"{base_url}/v1/chat/completions"
        else:
            self.endpoint = f"{base_url}/chat/completions"

        self.timeout = config.BACKEND_TIMEOUT_SECONDS
        self.model_id = config.MODEL_ID
        self.client: httpx.AsyncClient | None = None
        self._ready = False

    async def load(self) -> None:
        """Initialize httpx client."""
        logger.info("Initializing OpenAICompatibleBackend with endpoint: %s", self.endpoint)
        self.client = httpx.AsyncClient(timeout=self.timeout)
        self._ready = True

    def is_ready(self) -> bool:
        return self._ready and self.client is not None

    def get_model_name(self) -> str:
        return self.model_id

    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 512,
    ) -> Tuple[str, int, int, str]:
        if not self.client:
            raise RuntimeError("Backend client is not initialized.")

        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": False,
        }

        try:
            response = await self.client.post(self.endpoint, json=payload)
            response.raise_for_status()
            data = response.json()

            choice = data["choices"][0]
            content = choice["message"]["content"]
            finish_reason = choice.get("finish_reason", "stop")

            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            return content, prompt_tokens, completion_tokens, finish_reason
        except httpx.HTTPStatusError as e:
            logger.error("Backend returned HTTP %d: %s", e.response.status_code, e.response.text)
            raise RuntimeError(
                f"Backend HTTP error {e.response.status_code}: {e.response.text}"
            ) from e
        except httpx.RequestError as e:
            logger.error("Failed to connect to backend endpoint %s: %s", self.endpoint, str(e))
            raise RuntimeError(f"Backend connection error: {str(e)}") from e

    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 512,
    ) -> AsyncIterator[str]:
        """Forward streaming chat completion to external OpenAI-compatible server."""
        if not self.client:
            raise RuntimeError("Backend client is not initialized.")

        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": True,
        }

        try:
            async with self.client.stream("POST", self.endpoint, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        choices = chunk.get("choices", [])
                        if choices:
                            delta_content = choices[0].get("delta", {}).get("content", "")
                            if delta_content:
                                yield delta_content
                    except json.JSONDecodeError:
                        continue
        except httpx.HTTPStatusError as e:
            logger.error(
                "Backend streaming returned HTTP %d: %s", e.response.status_code, e.response.text
            )
            raise RuntimeError(
                f"Backend HTTP error {e.response.status_code}: {e.response.text}"
            ) from e
        except httpx.RequestError as e:
            logger.error("Failed to connect to backend endpoint %s: %s", self.endpoint, str(e))
            raise RuntimeError(f"Backend connection error: {str(e)}") from e

    async def close(self) -> None:
        """Close httpx client."""
        if self.client:
            await self.client.aclose()
            self.client = None
        self._ready = False
