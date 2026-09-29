"""Two-stage hybrid retriever: dense vector search followed by neural cross-encoder re-ranking.

Per Section 5.5:
1. Dense Retrieval: Retrieve top retrieve_k (default 20) chunks from ChromaDB using BGE embeddings.
2. Cross-Encoder Re-Ranking: Re-rank the candidates using BAAI/bge-reranker-base to return top final_k (default 3).
"""

import logging
from typing import Any, Dict, List, Optional

from .config import Settings, settings
from .embeddings import EmbeddingModel, get_embedding_model
from .reranker import Reranker, get_reranker
from .schemas import QueryResultItem
from .store import ChromaStore, get_chroma_store

logger = logging.getLogger(__name__)


class Retriever:
    """Two-stage retriever combining dense search with cross-encoder re-ranking."""

    def __init__(
        self,
        store: Optional[ChromaStore] = None,
        embedding_model: Optional[EmbeddingModel] = None,
        reranker: Optional[Reranker] = None,
        config: Optional[Settings] = None,
    ):
        self.config = config or settings
        self.store = store or get_chroma_store(self.config)
        self.embedding_model = embedding_model or get_embedding_model(self.config)
        self.reranker = reranker or get_reranker(self.config)

    def retrieve(
        self,
        query: str,
        strategy: str = "structure",
        retrieve_k: int = 20,
        final_k: int = 3,
        doc_ids: Optional[List[str]] = None,
    ) -> List[QueryResultItem]:
        """Execute two-stage retrieval and return re-ranked chunks with both scores.

        Args:
            query: User search query.
            strategy: Chunking strategy to query ('character', 'structure', 'semantic').
            retrieve_k: Number of initial candidates to pull from ChromaDB (default 20).
            final_k: Number of top re-ranked chunks to return (default 3).
            doc_ids: Optional list of document IDs to scope search.

        Returns:
            List of QueryResultItem instances sorted by rerank_score descending.
        """
        if not query or not query.strip():
            return []

        # Stage 1: Dense Retrieval
        # Encode query with BGE instruction prefix
        query_embedding = self.embedding_model.encode_query(query)

        # Pull top retrieve_k candidates from Chroma
        candidates = self.store.query(
            strategy=strategy,
            query_embedding=query_embedding,
            n_results=retrieve_k,
            doc_ids=doc_ids,
        )

        if not candidates:
            logger.info("No candidate chunks retrieved for query '%s' under strategy '%s'.", query, strategy)
            return []

        # Stage 2: Cross-Encoder Re-Ranking
        reranked = self.reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=final_k,
        )

        # Convert to Pydantic items
        items = [
            QueryResultItem(
                chunk_id=c["chunk_id"],
                text=c["text"],
                source=c.get("source", ""),
                page=c.get("page", 1),
                strategy=c.get("strategy", strategy),
                dense_score=c.get("dense_score", 0.0),
                rerank_score=c.get("rerank_score", 0.0),
            )
            for c in reranked
        ]

        return items


_retriever: Optional[Retriever] = None


def get_retriever(
    store: Optional[ChromaStore] = None,
    embedding_model: Optional[EmbeddingModel] = None,
    reranker: Optional[Reranker] = None,
    cfg: Optional[Settings] = None,
) -> Retriever:
    """Return or initialize the singleton Retriever."""
    global _retriever
    if _retriever is None or store is not None:
        _retriever = Retriever(
            store=store,
            embedding_model=embedding_model,
            reranker=reranker,
            config=cfg or settings,
        )
    return _retriever