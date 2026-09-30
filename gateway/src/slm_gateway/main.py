"""SLM Gateway — OpenAI-compatible FastAPI application."""

from contextlib import asynccontextmanager
import logging
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .backends import LLMBackend, get_backend
from .config import settings
from .pii import PIIRedactionError, PIIRedactor, get_redactor
from .rag_client import RAGClient
from .router import IntentRouter, RoutingResult, get_router
from .schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
    ModelCard,
    ModelListResponse,
    OpenAIErrorResponse,
    Usage,
)

# Configure logging per rule 7: use logging, not print
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Backend, PII & Router instances
backend: Optional[LLMBackend] = None
pii_redactor: Optional[PIIRedactor] = None
router_instance: Optional[IntentRouter] = None
rag_client_instance: Optional[RAGClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model backend, PII redactor, and intent router once at startup per rule 5."""
    global backend, pii_redactor, router_instance, rag_client_instance
    if pii_redactor is None:
        logger.info("Initializing SLM Gateway PII redactor...")
        try:
            pii_redactor = get_redactor(settings)
        except Exception as e:
            logger.error("Failed to load PII redactor: %s", str(e), exc_info=True)
    app.state.pii_redactor = pii_redactor

    if router_instance is None:
        logger.info("Initializing SLM Gateway Intent router...")
        try:
            router_instance = get_router(settings)
        except Exception as e:
            logger.error("Failed to load Intent router: %s", str(e), exc_info=True)
    app.state.router = router_instance

    if rag_client_instance is None:
        rag_client_instance = RAGClient(settings)
    app.state.rag_client = rag_client_instance

    if backend is None:
        logger.info("Initializing SLM Gateway backend: %s...", settings.BACKEND)
        backend = get_backend(settings)
        try:
            await backend.load()
        except Exception as e:
            logger.error("Failed to load backend during startup: %s", str(e), exc_info=True)
    app.state.backend = backend
    yield
    logger.info("Shutting down SLM Gateway...")
    if backend:
        await backend.close()


app = FastAPI(
    title="Ericsson SLM Gateway",
    description="OpenAI-compatible SLM Gateway with PII redaction and semantic routing",
    version="0.1.0",
    lifespan=lifespan,
)


# ============================================================
# OpenAI-style Error Handlers
# ============================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Format Pydantic schema validation errors into OpenAI error format."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={
            "error": {
                "message": str(exc),
                "type": "invalid_request_error",
                "code": 422,
            }
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Ensure HTTP exceptions adhere to OpenAI error structure."""
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        content = exc.detail
    else:
        content = {
            "error": {
                "message": str(exc.detail),
                "type": "invalid_request_error",
                "code": exc.status_code,
            }
        }
    return JSONResponse(status_code=exc.status_code, content=content)


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Handle unexpected server exceptions."""
    logger.error("Internal server error: %s", str(exc), exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "message": str(exc),
                "type": "server_error",
                "code": 500,
            }
        },
    )


# ============================================================
# Health & Status Routes
# ============================================================

@app.get("/health")
async def health():
    """Liveness probe."""
    return {
        "status": "healthy",
        "service": settings.SERVICE_NAME,
    }


@app.get("/ready")
async def ready():
    """Readiness probe: 200 only when model backend and router are fully loaded per section 4."""
    global router_instance
    if router_instance is None:
        try:
            router_instance = get_router(settings)
        except Exception as e:
            logger.warning("Router not initialized yet: %s", e)

    backend_ready = backend is not None and backend.is_ready()
    router_ready = router_instance is not None

    if backend_ready and router_ready:
        return {
            "status": "ready",
            "backend": backend.get_model_name(),
            "router": settings.ROUTER_MODEL_NAME,
        }

    missing = []
    if not backend_ready:
        missing.append("model backend")
    if not router_ready:
        missing.append("intent router")

    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={
            "error": {
                "message": f"Service not ready. Missing: {', '.join(missing)}.",
                "type": "service_unavailable",
                "code": 503,
            }
        },
    )


# ============================================================
# OpenAI-Compatible API Routes
# ============================================================

@app.get("/v1/models", response_model=ModelListResponse)
async def list_models():
    """Return loaded model in OpenAI list format."""
    model_name = backend.get_model_name() if backend else settings.MODEL_ID
    return ModelListResponse(
        data=[
            ModelCard(
                id=model_name,
                object="model",
                owned_by="local",
            )
        ]
    )


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    x_bypass_router: Optional[str] = Header(None, alias="X-Bypass-Router"),
):
    """Generate chat completions conforming to the OpenAI API specification.

    WHY: This is the core 'front door' for all user requests. It ensures incoming
    prompts are scrubbed of PII before any model sees them, routes to RAG only when
    document context is required, and formats results in standard OpenAI shape.
    """
    if request.stream:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "message": "Streaming is not supported in this version. Set stream=false.",
                    "type": "invalid_request_error",
                    "code": 400,
                }
            },
        )

    if backend is None or not backend.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": {
                    "message": "Model backend is not ready to accept inference requests.",
                    "type": "service_unavailable",
                    "code": 503,
                }
            },
        )

    # 1. Redact PII from user messages
    total_redactions = 0
    sanitized_messages = []
    redactor = pii_redactor or get_redactor(settings)

    for message in request.messages:
        msg_dict = message.model_dump()
        if msg_dict.get("role") == "user":
            try:
                redacted_content, count = redactor.redact(msg_dict.get("content", ""))
                msg_dict["content"] = redacted_content
                total_redactions += count
            except PIIRedactionError as e:
                logger.error("PII redaction failed in fail-closed mode: %s", str(e))
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={
                        "error": {
                            "message": f"PII redaction failed: {str(e)}",
                            "type": "pii_redaction_error",
                            "code": 500,
                        }
                    },
                )
        sanitized_messages.append(msg_dict)

    # 2. Semantic Intent Routing
    x_routing: Optional[dict] = None
    bypass = x_bypass_router is not None and x_bypass_router.strip().lower() in ("true", "1")

    if bypass:
        x_routing = {
            "intent": "bypass",
            "confidence": 1.0,
            "route": "hf_local",
            "latency_ms": 0.0,
        }
    else:
        # Extract latest user message for classification
        user_queries = [
            m.get("content", "") for m in sanitized_messages if m.get("role") == "user"
        ]
        query_to_route = user_queries[-1] if user_queries else ""

        router = router_instance or get_router(settings)
        route_result = router.classify(query_to_route)
        x_routing = route_result.to_dict()

        # Apply intent effects: 'rag' calls the RAG service if documents are indexed
        if route_result.intent == "rag":
            rag_cli = rag_client_instance or RAGClient(settings)
            has_docs, check_err = await rag_cli.has_indexed_documents()
            if not has_docs:
                if check_err == "no_documents_indexed":
                    x_routing["warning"] = "No documents indexed in RAG service; fell back to local model."
                else:
                    x_routing["warning"] = f"RAG service unavailable ({check_err}); fell back to local model."
                x_routing["route"] = "hf_local"
            else:
                rag_resp, rag_err = await rag_cli.get_answer(query=query_to_route)
                if rag_err or not rag_resp:
                    x_routing["warning"] = f"RAG answer generation failed ({rag_err}); fell back to local model."
                    x_routing["route"] = "hf_local"
                else:
                    x_routing["route"] = "rag_service"
                    rag_content = rag_resp.get("answer", "")
                    rag_sources = rag_resp.get("sources", [])
                    raw_usage = rag_resp.get("usage", {})

                    model_name = request.model or "rag-pipeline"
                    return ChatCompletionResponse(
                        model=model_name,
                        choices=[
                            Choice(
                                index=0,
                                message=ChoiceMessage(role="assistant", content=rag_content),
                                finish_reason="stop",
                            )
                        ],
                        usage=Usage(
                            prompt_tokens=raw_usage.get("prompt_tokens", 0),
                            completion_tokens=raw_usage.get("completion_tokens", 0),
                            total_tokens=raw_usage.get("total_tokens", 0),
                        ),
                        x_routing=x_routing,
                        x_pii={"redactions": total_redactions},
                        x_sources=rag_sources,
                    )

    try:
        content, prompt_tokens, completion_tokens, finish_reason = await backend.generate(
            messages=sanitized_messages,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
        )
    except Exception as e:
        logger.error("Generation failed: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": f"Generation failed: {str(e)}",
                    "type": "generation_error",
                    "code": 500,
                }
            },
        )

    model_name = request.model or backend.get_model_name()

    return ChatCompletionResponse(
        model=model_name,
        choices=[
            Choice(
                index=0,
                message=ChoiceMessage(role="assistant", content=content),
                finish_reason=finish_reason,
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        x_routing=x_routing,
        x_pii={"redactions": total_redactions},
    )