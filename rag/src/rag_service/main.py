"""RAG Service — Document Ingestion, Chunking, Retrieval, and Re-ranking API."""

from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
import tempfile
from typing import Optional
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from .chunking import chunk_document
from .config import settings
from .embeddings import EmbeddingModel, get_embedding_model
from .parsers import parse_document
from .reranker import Reranker, get_reranker
from .retriever import Retriever, get_retriever
from .schemas import (
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
    title="Ericsson Hybrid RAG Service",
    description="Multi-strategy document chunking, ChromaDB vector store, and neural cross-encoder re-ranking",
    version="0.1.0",
    lifespan=lifespan,
)


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
    if ext not in (".pdf", ".docx"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Only .pdf and .docx files are supported.",
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

        # Generate unique document ID
        doc_id = f"doc_{uuid.uuid4().hex[:8]}"

        # Apply chunking strategies
        emb_model = embedding_model or get_embedding_model(settings)
        chroma_store = store or get_chroma_store(settings)

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
async def list_documents():
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
async def delete_document(doc_id: str):
    """Delete an indexed document from all ChromaDB collections."""
    chroma_store = store or get_chroma_store(settings)
    deleted_count = chroma_store.delete_document(doc_id)

    if deleted_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with doc_id '{doc_id}' not found.",
        )

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
    """Retrieve and cross-encoder re-rank document chunks."""
    rag_retriever = retriever or get_retriever(cfg=settings)

    results = rag_retriever.retrieve(
        query=request.query,
        strategy=request.strategy,
        retrieve_k=request.retrieve_k,
        final_k=request.final_k,
        doc_ids=request.doc_ids,
        use_reranker=request.use_reranker,
    )

    return QueryResponse(
        query=request.query,
        strategy=request.strategy,
        results=results,
        total_retrieved=len(results),
    )
