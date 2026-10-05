"""Async HTTP client to interact with the external RAG service on port 8001.

Handles document presence checks, answer requests, and graceful failure fallbacks
(service down, empty index, connection timeout) per Section 5.3.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import httpx

from .config import Settings, settings

logger = logging.getLogger(__name__)


class RAGClient:
    """Client for RAG microservice communication."""

    def __init__(self, config: Optional[Settings] = None, client: Optional[httpx.AsyncClient] = None):
        self.config = config or settings
        self.base_url = self.config.RAG_SERVICE_URL.rstrip("/")
        self.timeout = self.config.RAG_TIMEOUT_SECONDS
        self._external_client = client

    def _get_headers(self, request_id: Optional[str] = None) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if request_id:
            headers["X-Request-ID"] = request_id
        if getattr(self.config, "RAG_API_KEY", None):
            headers["Authorization"] = f"Bearer {self.config.RAG_API_KEY}"
        return headers

    async def _get_client(self) -> httpx.AsyncClient:
        return self._external_client or httpx.AsyncClient()

    async def has_indexed_documents(self, request_id: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """Check whether the RAG service is online and has indexed documents.

        Returns:
            Tuple of (has_docs: bool, reason_if_false: Optional[str])
        """
        endpoint = f"{self.base_url}/documents"
        headers = self._get_headers(request_id)
        try:
            if self._external_client:
                resp = await self._external_client.get(endpoint, headers=headers, timeout=self.timeout)
            else:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(endpoint, headers=headers, timeout=self.timeout)

            if resp.status_code != 200:
                logger.warning("RAG service /documents returned status %d", resp.status_code)
                return False, f"rag_status_{resp.status_code}"

            data = resp.json()
            total_docs = data.get("total_documents", 0)
            if total_docs == 0:
                return False, "no_documents_indexed"
            return True, None

        except httpx.ConnectError:
            logger.warning("Cannot connect to RAG service at %s (service down)", endpoint)
            return False, "rag_service_offline"
        except httpx.TimeoutException:
            logger.warning("RAG service connection timed out at %s", endpoint)
            return False, "rag_timeout"
        except Exception as e:
            logger.warning("Unexpected error querying RAG service: %s", str(e))
            return False, f"rag_exception_{type(e).__name__}"

    async def get_answer(
        self,
        query: str,
        strategy: Optional[str] = None,
        retrieve_k: int = 20,
        final_k: int = 3,
        request_id: Optional[str] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Request a grounded answer from the RAG service.

        Returns:
            Tuple of (answer_data: Optional[Dict], failure_reason: Optional[str])
        """
        endpoint = f"{self.base_url}/answer"
        payload = {
            "query": query,
            "strategy": strategy or self.config.RAG_DEFAULT_STRATEGY,
            "retrieve_k": retrieve_k,
            "final_k": final_k,
            "use_reranker": True,
        }
        headers = self._get_headers(request_id)

        try:
            if self._external_client:
                resp = await self._external_client.post(endpoint, json=payload, headers=headers, timeout=self.timeout)
            else:
                async with httpx.AsyncClient() as client:
                    resp = await client.post(endpoint, json=payload, headers=headers, timeout=self.timeout)

            if resp.status_code != 200:
                logger.warning("RAG service /answer returned status %d", resp.status_code)
                return None, f"rag_status_{resp.status_code}"

            data = resp.json()
            return data, None

        except httpx.ConnectError:
            logger.warning("RAG service unreachable during /answer request at %s", endpoint)
            return None, "rag_service_offline"
        except httpx.TimeoutException:
            logger.warning("RAG service timed out during /answer request at %s", endpoint)
            return None, "rag_timeout"
        except Exception as e:
            logger.warning("Unexpected error during RAG /answer request: %s", str(e))
            return None, f"rag_exception_{type(e).__name__}"
