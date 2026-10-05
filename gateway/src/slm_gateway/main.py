"""SLM Gateway — OpenAI-compatible FastAPI application."""

from contextlib import asynccontextmanager
import json
import logging
from pathlib import Path
import time
from typing import Optional
import uuid

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from .backends import LLMBackend, get_backend
from .config import settings
from .pii import PIIRedactionError, PIIRedactor, get_redactor
from .rag_client import RAGClient
from .router import IntentRouter, RoutingResult, get_router
from .schemas import (
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
    ModelCard,
    ModelListResponse,
    OpenAIErrorResponse,
    Usage,
)
from .telemetry import telemetry

# Configure logging per rule 7: use logging, not print
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Lifespan context manager: loads dependencies into app.state
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model backend, PII redactor, and intent router once at startup per rule 5."""
    if getattr(app.state, "pii_redactor", None) is None:
        logger.info("Initializing SLM Gateway PII redactor...")
        try:
            app.state.pii_redactor = get_redactor(settings)
        except Exception as e:
            logger.error("Failed to load PII redactor: %s", str(e), exc_info=True)
            if settings.PII_FAIL_MODE == "closed":
                raise RuntimeError(
                    f"SLM Gateway startup aborted: Presidio PII redactor failed to initialize with PII_FAIL_MODE=closed: {e}"
                ) from e
            logger.warning("Operating in PII fail-open mode; startup continuing without PII redaction.")
            app.state.pii_redactor = None

    if getattr(app.state, "router", None) is None:
        logger.info("Initializing SLM Gateway Intent router...")
        try:
            app.state.router = get_router(settings)
        except Exception as e:
            logger.error("Failed to load Intent router: %s", str(e), exc_info=True)
            app.state.router = None

    if getattr(app.state, "rag_client", None) is None:
        app.state.rag_client = RAGClient(settings)

    if getattr(app.state, "backend", None) is None:
        logger.info("Initializing SLM Gateway backend: %s...", settings.BACKEND)
        loaded_backend = get_backend(settings)
        try:
            await loaded_backend.load()
        except Exception as e:
            logger.error("Failed to load backend during startup: %s", str(e), exc_info=True)
        app.state.backend = loaded_backend

    yield
    logger.info("Shutting down SLM Gateway...")
    shutdown_backend = getattr(app.state, "backend", None)
    if shutdown_backend:
        await shutdown_backend.close()


app = FastAPI(
    title="Local SLM Gateway",
    description="OpenAI-compatible SLM Gateway with PII redaction, semantic routing, and streaming",
    version="0.1.0",
    lifespan=lifespan,
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


STATIC_DIR = Path(__file__).parent / "static"


# ============================================================
# Playground UI & Health Routes
# ============================================================

@app.get("/", response_class=HTMLResponse)
@app.get("/playground", response_class=HTMLResponse)
async def playground():
    """Interactive GenAI Stack Web Playground."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Playground template not found</h1>", status_code=404)


@app.get("/health")
async def health():
    """Liveness probe."""
    return {
        "status": "healthy",
        "service": settings.SERVICE_NAME,
    }


@app.get("/ready")
async def ready(request: Request):
    """Readiness probe: 200 only when model backend and router are fully loaded per section 4."""
    router = getattr(request.app.state, "router", None)
    if router is None:
        try:
            router = get_router(settings)
            request.app.state.router = router
        except Exception as e:
            logger.warning("Router not initialized yet: %s", e)

    backend = getattr(request.app.state, "backend", None)
    backend_ready = backend is not None and backend.is_ready()
    router_ready = router is not None

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


@app.get("/metrics")
async def metrics():
    """Prometheus telemetry scrape endpoint returning OpenMetrics plain-text."""
    return Response(
        content=telemetry.export_prometheus_text(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


# ============================================================
# OpenAI-Compatible API Routes
# ============================================================

def verify_api_key(authorization: Optional[str] = Header(None)) -> None:
    """Validate Bearer token if GATEWAY_API_KEY is configured per section 4."""
    if not settings.GATEWAY_API_KEY:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": {
                    "message": "Missing or invalid Authorization header. Expected 'Bearer <key>'.",
                    "type": "invalid_request_error",
                    "code": 401,
                }
            },
        )
    token = authorization.split("Bearer ", 1)[1].strip()
    if token != settings.GATEWAY_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": {
                    "message": "Incorrect API key provided.",
                    "type": "invalid_request_error",
                    "code": 401,
                }
            },
        )


@app.get("/v1/models", response_model=ModelListResponse)
async def list_models(raw_request: Request, _auth: None = Depends(verify_api_key)):
    """Return loaded model in OpenAI list format."""
    backend = getattr(raw_request.app.state, "backend", None)
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
    raw_request: Request,
    x_bypass_router: Optional[str] = Header(None, alias="X-Bypass-Router"),
    x_rag_strategy: Optional[str] = Header(None, alias="X-RAG-Strategy"),
    _auth: None = Depends(verify_api_key),
):
    """Generate chat completions conforming to the OpenAI API specification.

    WHY: This is the core 'front door' for all user requests. It ensures incoming
    prompts are scrubbed of PII before any model sees them, routes to RAG only when
    document context is required, and formats results in standard OpenAI shape.
    """
    backend = getattr(raw_request.app.state, "backend", None)
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

    # Validate optional X-RAG-Strategy header
    selected_strategy = None
    if x_rag_strategy is not None:
        strat = x_rag_strategy.strip().lower()
        if strat not in ("character", "structure", "semantic"):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail={
                    "error": {
                        "message": f"Invalid X-RAG-Strategy '{x_rag_strategy}'. Must be one of: character, structure, semantic.",
                        "type": "invalid_request_error",
                        "code": 422,
                    }
                },
            )
        selected_strategy = strat

    t0 = time.perf_counter()
    telemetry.inc_active_requests()

    # 1. Redact PII from user messages
    total_redactions = 0
    sanitized_messages = []
    redactor = getattr(raw_request.app.state, "pii_redactor", None) or get_redactor(settings)

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

        router = getattr(raw_request.app.state, "router", None) or get_router(settings)
        route_result = router.classify(query_to_route)
        x_routing = route_result.to_dict()

        # Apply intent effects: 'rag' calls the RAG service if documents are indexed
        if route_result.intent == "rag":
            rag_cli = getattr(raw_request.app.state, "rag_client", None) or RAGClient(settings)
            has_docs, check_err = await rag_cli.has_indexed_documents()
            if not has_docs:
                if check_err == "no_documents_indexed":
                    x_routing["warning"] = "No documents indexed in RAG service; fell back to local model."
                else:
                    x_routing["warning"] = f"RAG service unavailable ({check_err}); fell back to local model."
                x_routing["route"] = "hf_local"
            else:
                rag_resp, rag_err = await rag_cli.get_answer(
                    query=query_to_route,
                    strategy=selected_strategy,
                )
                if rag_err or not rag_resp:
                    x_routing["warning"] = f"RAG answer generation failed ({rag_err}); fell back to local model."
                    x_routing["route"] = "hf_local"
                else:
                    x_routing["route"] = "rag_service"
                    rag_content = rag_resp.get("answer", "")
                    rag_sources = rag_resp.get("sources", [])
                    raw_usage = rag_resp.get("usage", {})

                    model_name = request.model or "rag-pipeline"

                    if request.stream:
                        async def stream_rag_chunks():
                            chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
                            created = int(time.time())
                            try:
                                # Initial chunk with metadata
                                meta_chunk = ChatCompletionChunk(
                                    id=chunk_id,
                                    created=created,
                                    model=model_name,
                                    choices=[
                                        ChatCompletionChunkChoice(
                                            index=0,
                                            delta=ChatCompletionChunkDelta(role="assistant", content=""),
                                            finish_reason=None,
                                        )
                                    ],
                                    x_routing=x_routing,
                                    x_pii={"redactions": total_redactions},
                                    x_sources=rag_sources,
                                )
                                yield f"data: {meta_chunk.model_dump_json()}\n\n"

                                import re
                                tokens = re.findall(r"\S+|\s+", rag_content)
                                for t in tokens:
                                    chunk = ChatCompletionChunk(
                                        id=chunk_id,
                                        created=created,
                                        model=model_name,
                                        choices=[
                                            ChatCompletionChunkChoice(
                                                index=0,
                                                delta=ChatCompletionChunkDelta(content=t),
                                                finish_reason=None,
                                            )
                                        ],
                                    )
                                    yield f"data: {chunk.model_dump_json()}\n\n"

                                stop_chunk = ChatCompletionChunk(
                                    id=chunk_id,
                                    created=created,
                                    model=model_name,
                                    choices=[
                                        ChatCompletionChunkChoice(
                                            index=0,
                                            delta=ChatCompletionChunkDelta(),
                                            finish_reason="stop",
                                        )
                                    ],
                                )
                                yield f"data: {stop_chunk.model_dump_json()}\n\n"
                                yield "data: [DONE]\n\n"
                            finally:
                                telemetry.dec_active_requests()
                                telemetry.record_pii_redactions(total_redactions)
                                telemetry.record_request("rag", 200, time.perf_counter() - t0)

                        return StreamingResponse(stream_rag_chunks(), media_type="text/event-stream")

                    telemetry.dec_active_requests()
                    telemetry.record_pii_redactions(total_redactions)
                    telemetry.record_tokens(
                        raw_usage.get("prompt_tokens", 0),
                        raw_usage.get("completion_tokens", 0),
                    )
                    telemetry.record_request("rag", 200, time.perf_counter() - t0)

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

    model_name = request.model or backend.get_model_name()

    if request.stream:
        async def stream_local_chunks():
            chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            created = int(time.time())
            is_first = True
            try:
                async for token_text in backend.generate_stream(
                    messages=sanitized_messages,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_tokens=request.max_tokens,
                ):
                    if is_first:
                        chunk = ChatCompletionChunk(
                            id=chunk_id,
                            created=created,
                            model=model_name,
                            choices=[
                                ChatCompletionChunkChoice(
                                    index=0,
                                    delta=ChatCompletionChunkDelta(role="assistant", content=token_text),
                                    finish_reason=None,
                                )
                            ],
                            x_routing=x_routing,
                            x_pii={"redactions": total_redactions},
                        )
                        is_first = False
                    else:
                        chunk = ChatCompletionChunk(
                            id=chunk_id,
                            created=created,
                            model=model_name,
                            choices=[
                                ChatCompletionChunkChoice(
                                    index=0,
                                    delta=ChatCompletionChunkDelta(content=token_text),
                                    finish_reason=None,
                                )
                            ],
                        )
                    yield f"data: {chunk.model_dump_json()}\n\n"

                stop_chunk = ChatCompletionChunk(
                    id=chunk_id,
                    created=created,
                    model=model_name,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatCompletionChunkDelta(),
                            finish_reason="stop",
                        )
                    ],
                )
                yield f"data: {stop_chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error("Streaming generation failed: %s", str(e), exc_info=True)
                err_payload = json.dumps({"error": {"message": str(e), "type": "generation_error", "code": 500}})
                yield f"data: {err_payload}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                telemetry.dec_active_requests()
                telemetry.record_pii_redactions(total_redactions)
                telemetry.record_request(
                    x_routing.get("intent", "general") if x_routing else "general",
                    200,
                    time.perf_counter() - t0,
                )

        return StreamingResponse(stream_local_chunks(), media_type="text/event-stream")

    try:
        content, prompt_tokens, completion_tokens, finish_reason = await backend.generate(
            messages=sanitized_messages,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
        )
    except Exception as e:
        logger.error("Generation failed: %s", str(e), exc_info=True)
        telemetry.dec_active_requests()
        telemetry.record_request(
            x_routing.get("intent", "general") if x_routing else "general",
            500,
            time.perf_counter() - t0,
        )
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

    telemetry.dec_active_requests()
    telemetry.record_pii_redactions(total_redactions)
    telemetry.record_tokens(prompt_tokens, completion_tokens)
    telemetry.record_request(
        x_routing.get("intent", "general") if x_routing else "general",
        200,
        time.perf_counter() - t0,
    )

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


# ============================================================
# Compatibility Module Wrapper
# Ensures external references/tests accessing slm_gateway.main.backend
# or monkeypatching attributes seamlessly read/write app.state.
# ============================================================
import sys

class _GatewayModule(sys.modules[__name__].__class__):
    @property
    def backend(self):
        return getattr(app.state, "backend", None)

    @backend.setter
    def backend(self, value):
        app.state.backend = value

    @property
    def pii_redactor(self):
        return getattr(app.state, "pii_redactor", None)

    @pii_redactor.setter
    def pii_redactor(self, value):
        app.state.pii_redactor = value

    @property
    def router_instance(self):
        return getattr(app.state, "router", None)

    @router_instance.setter
    def router_instance(self, value):
        app.state.router = value

    @property
    def rag_client_instance(self):
        return getattr(app.state, "rag_client", None)

    @rag_client_instance.setter
    def rag_client_instance(self, value):
        app.state.rag_client = value

sys.modules[__name__].__class__ = _GatewayModule