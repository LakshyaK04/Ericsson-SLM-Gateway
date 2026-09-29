from fastapi import FastAPI, HTTPException
import httpx

from .pii import redact_pii
from .router import IntentRouter
from .rag import RAGPipeline


app = FastAPI(
    title="Ericsson SLM Gateway",
    description="FastAPI gateway for Phi-3 Mini served by vLLM",
    version="1.0.0",
)

VLLM_URL = "http://localhost:8000"

router = IntentRouter()

rag_pipeline = RAGPipeline(
    "data/ericsson_rag_sample.pdf"
)


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "slm-gateway",
    }

@app.post("/v1/rag/query")
async def rag_query(request: dict):
    query = request.get("query", "")
    strategy = request.get(
        "chunking_strategy",
        "character",
    )

    if not query:
        raise HTTPException(
            status_code=400,
            detail="Query is required",
        )

    if strategy not in {
        "character",
        "structure",
        "semantic",
    }:
        raise HTTPException(
            status_code=400,
            detail="Invalid chunking strategy",
        )

    rag = RAGPipeline(
        "data/ericsson_rag_sample.pdf",
        strategy,
    )

    result = await rag.generate(query)

    return {
        "query": query,
        "chunking_strategy": strategy,
        "answer": result["answer"],
        "sources": result["sources"],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: dict):
    try:
        # 1. Redact PII
        for message in request.get("messages", []):
            if message.get("role") == "user":
                message["content"] = redact_pii(
                    message.get("content", "")
                )

        # 2. Get user's latest message
        user_messages = [
            message.get("content", "")
            for message in request.get("messages", [])
            if message.get("role") == "user"
        ]

        user_query = user_messages[-1] if user_messages else ""

        # 3. Determine intent
        routing_result = router.classify(user_query)

        # 4. RAG route
        if routing_result["intent"] == "rag":
            rag_result = await rag_pipeline.generate(
                user_query
            )

            return {
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": rag_result["answer"],
                        },
                    }
                ],
                "routing": routing_result,
                "sources": rag_result["sources"],
            }

        # 5. Normal vLLM route
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

        # 6. Add routing information
        result["routing"] = routing_result

        return result

    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail=f"vLLM server unavailable: {str(e)}",
        )