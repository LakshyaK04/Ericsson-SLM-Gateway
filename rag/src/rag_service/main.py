"""RAG Service — Document Ingestion, Chunking, Retrieval, and Re-ranking API."""

from contextlib import asynccontextmanager
import hashlib
import logging
import os
from pathlib import Path
import tempfile
from typing import Optional
import uuid

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse, StreamingResponse

from .chunking import chunk_document
from .config import settings
from .embeddings import EmbeddingModel, get_embedding_model
from .parsers import parse_document
from .pii import redact_ingest_text
from .rate_limiter import InMemoryRateLimiter
from .reranker import Reranker, get_reranker
from .retriever import Retriever, get_retriever
from .generation import generate_grounded_answer, stream_grounded_answer
from .schemas import (
    AnswerResponse,
    DeleteDocumentResponse,
    DocumentInfo,
    DocumentListResponse,
    DocumentUploadResponse,
    QueryRequest,
    QueryResponse,
    QueryResultItem,
)
from .store import ChromaStore, get_chroma_store

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Global singletons
store: Optional[ChromaStore] = None
embedding_model: Optional[EmbeddingModel] = None
reranker: Optional[Reranker] = None
retriever: Optional[Retriever] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager: load heavy embedding & reranker models once at startup."""
    global store, embedding_model, reranker, retriever
    logger.info("Initializing RAG Service...")

    if store is None:
        store = get_chroma_store(settings)
    if embedding_model is None:
        embedding_model = get_embedding_model(settings)
    if reranker is None:
        reranker = get_reranker(settings)
    if retriever is None:
        retriever = get_retriever(store, embedding_model, reranker, settings)

    app.state.store = store
    app.state.embedding_model = embedding_model
    app.state.reranker = reranker
    app.state.retriever = retriever

    logger.info("RAG Service fully initialized and ready.")
    yield
    logger.info("Shutting down RAG Service...")


app = FastAPI(
    title="Local RAG Service",
    description="Multi-strategy document chunking, ChromaDB vector store, BM25 hybrid search, and neural cross-encoder re-ranking",
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

rate_limiter = InMemoryRateLimiter(
    requests_limit=settings.RATE_LIMIT_REQUESTS,
    window_seconds=settings.RATE_LIMIT_WINDOW_SECONDS,
    enabled=settings.RATE_LIMIT_ENABLED,
)


def verify_api_key(authorization: Optional[str] = Header(None)) -> None:
    """Validate Bearer token for protected RAG administration endpoints."""
    if not settings.RAG_API_KEY:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header. Expected 'Bearer <key>'.",
        )
    token = authorization.split("Bearer ", 1)[1].strip()
    if token != settings.RAG_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect API key provided.",
        )


@app.middleware("http")
async def rag_security_and_tracing_middleware(request: Request, call_next):
    """Enforce request-ID tracing and in-memory rate limiting."""
    req_id = request.headers.get("X-Request-ID")
    if not req_id:
        req_id = f"req-{uuid.uuid4().hex[:12]}"
    request.state.request_id = req_id

    if settings.RATE_LIMIT_ENABLED:
        client_ip = (
            request.headers.get("x-forwarded-for")
            or (request.client.host if request.client else "unknown")
        ).split(",")[0].strip()
        allowed, retry_after = rate_limiter.check(client_ip)
        if not allowed:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                headers={
                    "X-Request-ID": req_id,
                    "Retry-After": str(retry_after),
                },
                content={"detail": f"Rate limit exceeded. Try again in {retry_after} seconds."},
            )

    response = await call_next(request)
    response.headers["X-Request-ID"] = req_id
    return response


# ============================================================
# Health Probe
# ============================================================

@app.get("/health")
async def health():
    """Liveness probe."""
    return {
        "status": "healthy",
        "service": settings.SERVICE_NAME,
        "collections": list(store.collections.keys()) if store else [],
    }


# ============================================================
# Document Ingestion Routes
# ============================================================

@app.post("/documents", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    strategies: str = Form("character,structure,semantic"),
    _auth: None = Depends(verify_api_key),
):
    """Upload and index a document (.pdf or .docx) under specified chunking strategies.

    Args:
        file: Multipart uploaded document (.pdf or .docx).
        strategies: Comma-separated list of strategies ('character,structure,semantic').
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a valid filename.",
        )

    ext = Path(file.filename).suffix.lower()
    if ext not in (".pdf", ".docx", ".txt", ".md"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Only .pdf, .docx, .txt, and .md files are supported.",
        )

    # Parse requested strategies
    requested_strategies = [s.strip().lower() for s in strategies.split(",") if s.strip()]
    valid_strategies = {"character", "structure", "semantic"}
    invalid = set(requested_strategies) - valid_strategies
    if invalid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid chunking strategies: {invalid}. Supported: {valid_strategies}",
        )

    # Save to a temporary file for parsing
    temp_dir = Path(tempfile.gettempdir())
    temp_file = temp_dir / f"upload_{uuid.uuid4().hex}_{file.filename}"

    try:
        content = await file.read()
        if len(content) > settings.MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"Uploaded file size ({len(content)} bytes) exceeds maximum allowed size of {settings.MAX_UPLOAD_SIZE_BYTES} bytes.",
            )

        # Content-hash deduplication
        content_hash = hashlib.sha256(content).hexdigest()
        chroma_store = store or get_chroma_store(settings)
        existing = chroma_store.find_document_by_hash(content_hash)
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Document with identical content already indexed (doc_id: '{existing['doc_id']}', filename: '{existing['source']}').",
            )

        with open(temp_file, "wb") as f:
            f.write(content)

        # Parse document into pages
        try:
            pages = parse_document(temp_file, filename=file.filename)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            )

        # Optional PII redaction on ingest
        if settings.PII_REDACTION_ON_INGEST:
            codenames = (
                [c.strip() for c in settings.PROJECT_CODENAMES.split(",") if c.strip()]
                if settings.PROJECT_CODENAMES
                else None
            )
            redacted_pages = []
            for p_num, p_text in pages:
                sanitized, _ = redact_ingest_text(p_text, codenames=codenames)
                redacted_pages.append((p_num, sanitized))
            pages = redacted_pages

        # Generate unique document ID
        doc_id = f"doc_{uuid.uuid4().hex[:8]}"

        # Apply chunking strategies
        emb_model = embedding_model or get_embedding_model(settings)

        chunked_by_strategy = chunk_document(
            pages=pages,
            source=file.filename,
            strategies=requested_strategies,
            embedding_model=emb_model,
        )

        chunk_counts: dict[str, int] = {}
        total_chunks = 0

        for strat, chunks in chunked_by_strategy.items():
            if chunks:
                # Embed chunks without query prefix
                texts = [c.text for c in chunks]
                embeddings = emb_model.encode_documents(texts)
                # Store in ChromaDB
                chroma_store.add_chunks(
                    strategy=strat,
                    chunks=chunks,
                    embeddings=embeddings,
                    doc_id=doc_id,
                    content_hash=content_hash,
                )
                chunk_counts[strat] = len(chunks)
                total_chunks += len(chunks)
            else:
                chunk_counts[strat] = 0

        logger.info(
            "Document '%s' (ID: %s) indexed with %d total chunks: %s",
            file.filename,
            doc_id,
            total_chunks,
            chunk_counts,
        )

        if retriever:
            retriever.invalidate_bm25()

        return DocumentUploadResponse(
            doc_id=doc_id,
            filename=file.filename,
            chunk_counts=chunk_counts,
            total_chunks=total_chunks,
        )

    finally:
        if temp_file.exists():
            try:
                os.remove(temp_file)
            except OSError:
                pass


@app.get("/documents", response_model=DocumentListResponse)
async def list_documents(_auth: None = Depends(verify_api_key)):
    """List all indexed documents and their chunk distributions."""
    chroma_store = store or get_chroma_store(settings)
    raw_docs = chroma_store.list_documents()

    docs = [
        DocumentInfo(
            doc_id=d["doc_id"],
            source=d["source"],
            strategies=d["strategies"],
            total_chunks=d["total_chunks"],
        )
        for d in raw_docs
    ]

    return DocumentListResponse(
        documents=docs,
        total_documents=len(docs),
    )


@app.delete("/documents/{doc_id}", response_model=DeleteDocumentResponse)
async def delete_document(doc_id: str, _auth: None = Depends(verify_api_key)):
    """Delete an indexed document from all ChromaDB collections."""
    chroma_store = store or get_chroma_store(settings)
    deleted_count = chroma_store.delete_document(doc_id)

    if deleted_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with doc_id '{doc_id}' not found.",
        )

    if retriever:
        retriever.invalidate_bm25()

    logger.info("Deleted document '%s' (%d chunks removed).", doc_id, deleted_count)
    return DeleteDocumentResponse(
        status="deleted",
        doc_id=doc_id,
        chunks_removed=deleted_count,
    )


# ============================================================
# Retrieval & Query Route
# ============================================================

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Retrieve and cross-encoder re-rank document chunks via hybrid (BM25 + Dense) or dense search."""
    if len(request.query) > settings.MAX_QUERY_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Query length ({len(request.query)}) exceeds maximum limit of {settings.MAX_QUERY_LENGTH} characters.",
        )

    rag_retriever = retriever or get_retriever(cfg=settings)

    results = rag_retriever.retrieve(
        query=request.query,
        strategy=request.strategy,
        retrieve_k=request.retrieve_k,
        final_k=request.final_k,
        doc_ids=request.doc_ids,
        use_reranker=request.use_reranker,
        retrieval_mode=request.retrieval_mode,
        rrf_k=request.rrf_k,
    )

    return QueryResponse(
        query=request.query,
        strategy=request.strategy,
        results=results,
        total_retrieved=len(results),
    )


@app.post("/answer", response_model=AnswerResponse)
async def answer_question(request: QueryRequest, raw_request: Request):
    """Retrieve relevant document chunks and generate a grounded answer via Gateway."""
    if len(request.query) > settings.MAX_QUERY_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Query length ({len(request.query)}) exceeds maximum limit of {settings.MAX_QUERY_LENGTH} characters.",
        )

    rag_retriever = retriever or get_retriever(cfg=settings)

    results = rag_retriever.retrieve(
        query=request.query,
        strategy=request.strategy,
        retrieve_k=request.retrieve_k,
        final_k=request.final_k,
        doc_ids=request.doc_ids,
        use_reranker=request.use_reranker,
        retrieval_mode=request.retrieval_mode,
    )

    req_id = getattr(raw_request.state, "request_id", None)

    if request.stream:
        return StreamingResponse(
            stream_grounded_answer(
                query=request.query,
                chunks=results,
                config=settings,
                request_id=req_id,
            ),
            media_type="text/event-stream",
            headers={"X-Request-ID": req_id} if req_id else {},
        )

    call_kwargs = {}
    import inspect
    sig = inspect.signature(generate_grounded_answer)
    if "request_id" in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        call_kwargs["request_id"] = req_id

    answer_resp = await generate_grounded_answer(
        query=request.query,
        chunks=results,
        config=settings,
        **call_kwargs,
    )
    return answer_resp

