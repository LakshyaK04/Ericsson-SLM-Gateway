"""SLM Gateway — OpenAI-compatible FastAPI application."""

from contextlib import asynccontextmanager
import logging
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .backends import LLMBackend, get_backend
from .config import settings
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
from .security import verify_api_key

# Configure logging per rule 7: use logging, not print
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Backend instance
backend: Optional[LLMBackend] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model backend once at startup per rule 5."""
    global backend
    logger.info("Initializing SLM Gateway backend: %s...", settings.BACKEND)
    backend = get_backend(settings)
    try:
        await backend.load()
    except Exception as e:
        logger.error("Failed to load backend during startup: %s", str(e), exc_info=True)
        # Service can still boot so /health works, but /ready will fail
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
    """Readiness probe: 200 only when model is fully loaded."""
    if backend is not None and backend.is_ready():
        return {
            "status": "ready",
            "backend": backend.get_model_name(),
        }
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={
            "error": {
                "message": "Model backend is not loaded or ready.",
                "type": "service_unavailable",
                "code": 503,
            }
        },
    )


# ============================================================
# OpenAI-Compatible API Routes
# ============================================================

@app.get("/v1/models", response_model=ModelListResponse)
async def list_models(
    _auth: None = Depends(verify_api_key),
):
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
    _auth: None = Depends(verify_api_key),
    x_bypass_router: Optional[str] = Header(None, alias="X-Bypass-Router"),
):
    """Generate chat completions conforming to the OpenAI API specification."""
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

    raw_messages = [m.model_dump() for m in request.messages]

    try:
        content, prompt_tokens, completion_tokens, finish_reason = await backend.generate(
            messages=raw_messages,
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
    )