"""Pydantic schemas for the RAG service API."""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    """Response returned upon successful document upload and chunking."""
    doc_id: str
    filename: str
    chunk_counts: Dict[str, int]
    total_chunks: int
    message: str = "Document successfully ingested and indexed."


class DocumentInfo(BaseModel):
    """Metadata for an indexed document in the RAG store."""
    doc_id: str
    source: str
    strategies: List[str]
    total_chunks: int


class DocumentListResponse(BaseModel):
    """Response returned by GET /documents."""
    documents: List[DocumentInfo]
    total_documents: int


class DeleteDocumentResponse(BaseModel):
    """Response returned by DELETE /documents/{doc_id}."""
    status: str = "deleted"
    doc_id: str
    chunks_removed: int


class QueryRequest(BaseModel):
    """Request body for POST /query."""
    query: str = Field(..., min_length=1, description="Search query string.")
    strategy: Literal["character", "structure", "semantic"] = "structure"
    retrieve_k: int = Field(default=20, ge=1, le=100)
    final_k: int = Field(default=3, ge=1, le=50)
    doc_ids: Optional[List[str]] = None
    use_reranker: bool = Field(default=True, description="Whether to apply neural cross-encoder re-ranking.")
    retrieval_mode: Literal["dense", "sparse", "hybrid"] = Field(
        default="hybrid",
        description="Retrieval mode: 'dense' (BGE vector search), 'sparse' (BM25 search), or 'hybrid' (BM25 + Dense RRF fusion).",
    )
    rrf_k: int = Field(default=60, ge=1, le=1000, description="Reciprocal Rank Fusion smoothing parameter.")
    stream: bool = Field(default=False, description="Whether to stream answer tokens as Server-Sent Events.")


class QueryResultItem(BaseModel):
    """Individual retrieved and re-ranked chunk with dense, BM25, RRF, and reranker scores."""
    chunk_id: str
    text: str
    source: str
    page: int
    strategy: str
    dense_score: float
    rerank_score: float
    bm25_score: Optional[float] = None
    rrf_score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    """Response body returned by POST /query."""
    query: str
    strategy: str
    results: List[QueryResultItem]
    total_retrieved: int


class SourceItem(BaseModel):
    """Source reference information included with answers."""
    chunk_id: str
    source: str
    page: int
    strategy: str
    dense_score: float
    rerank_score: float
    bm25_score: Optional[float] = None
    rrf_score: Optional[float] = None
    text: str


class UsageInfo(BaseModel):
    """Token usage report from generation."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class AnswerResponse(BaseModel):
    """Response body returned by POST /answer."""
    answer: str
    sources: List[SourceItem]
    usage: UsageInfo

