"""SLM Gateway — main FastAPI application.

This is a transitional version: routing and PII are wired in,
but the model backend and full OpenAI response shaping will
be built in Phase 1.
"""

import logging

from fastapi import FastAPI, HTTPException
import httpx

from .pii import redact_pii
from .router import IntentRouter

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Ericsson SLM Gateway",
    description="OpenAI-compatible gateway with PII redaction and intent routing",
    version="0.1.0",
)

VLLM_URL = "http://localhost:8000"

router = IntentRouter()


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "slm-gateway",
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: dict):
    try:
        # 1. Redact PII from user messages
        for message in request.get("messages", []):
            if message.get("role") == "user":
                message["content"] = redact_pii(
                    message.get("content", "")
                )

        # 2. Get user's latest message for routing
        user_messages = [
            message.get("content", "")
            for message in request.get("messages", [])
            if message.get("role") == "user"
        ]

        user_query = user_messages[-1] if user_messages else ""

        # 3. Determine intent
        routing_result = router.classify(user_query)

        # 4. Forward to vLLM backend
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{VLLM_URL}/v1/chat/completions",
                json=request,
            )

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=response.text,
            )

        result = response.json()

        # 5. Add routing metadata
        result["routing"] = routing_result

        return result

    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Backend unavailable: {str(e)}",
        )