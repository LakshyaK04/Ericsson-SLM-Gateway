from fastapi import FastAPI, HTTPException
import httpx

from .pii import redact_pii


app = FastAPI(
    title="Ericsson SLM Gateway",
    description="FastAPI gateway for Phi-3 Mini served by vLLM",
    version="1.0.0",
)

VLLM_URL = "http://localhost:8000"


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "slm-gateway",
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: dict):
    try:
        # Redact PII from incoming user messages
        for message in request.get("messages", []):
            if message.get("role") == "user":
                message["content"] = redact_pii(message.get("content", ""))

        # Forward the sanitized request to vLLM
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

        return response.json()

    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail=f"vLLM server unavailable: {str(e)}",
        )