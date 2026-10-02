"""Two-stage retriever: dense vector search followed by neural cross-encoder re-ranking.

Per Section 5.5:
1. Dense Retrieval: Retrieve top retrieve_k (default 20) chunks from ChromaDB using BGE embeddings.
2. Cross-Encoder Re-Ranking: Re-rank the candidates using BAAI/bge-reranker-base to return top final_k (default 3).
"""

import logging
from typing import Any, Dict, List, Optional

from .bm25 import BM25Index, reciprocal_rank_fusion
from .config import Settings, settings
from .embeddings import EmbeddingModel, get_embedding_model
from .reranker import Reranker, get_reranker
from .schemas import QueryResultItem
from .store import ChromaStore, get_chroma_store

logger = logging.getLogger(__name__)


class Retriever:
    """Two-stage hybrid retriever combining dense vector search and BM25 with cross-encoder re-ranking."""

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
        self.bm25_indices: Dict[str, BM25Index] = {}

    def _get_bm25_index(self, strategy: str) -> BM25Index:
        """Get or lazily construct the BM25 index for a chunking strategy from ChromaDB."""
        if strategy not in self.bm25_indices:
            idx = BM25Index()
            chunks = self.store.get_all_chunks(strategy)
            idx.index_chunks(chunks)
            self.bm25_indices[strategy] = idx
        return self.bm25_indices[strategy]

    def invalidate_bm25(self, strategy: Optional[str] = None) -> None:
        """Invalidate cached BM25 index after document ingestion or deletion."""
        if strategy:
            self.bm25_indices.pop(strategy, None)
        else:
            self.bm25_indices.clear()

    def retrieve(
        self,
        query: str,
        strategy: str = "structure",
        retrieve_k: int = 20,
        final_k: int = 3,
        doc_ids: Optional[List[str]] = None,
        use_reranker: bool = True,
        retrieval_mode: str = "hybrid",
    ) -> List[QueryResultItem]:
        """Execute hybrid two-stage retrieval and return re-ranked chunks with all scores.

        WHY: Dense bi-encoder search (BGE-small) excels at conceptual semantic similarity,
        while sparse lexical search (Okapi BM25) excels at precise keyword, acronym, and
        identifier matching (e.g. EMP-12345, error codes, telecom standards). Fusing both
        via Reciprocal Rank Fusion (RRF) before the neural cross-encoder yields state-of-the-art
        retrieval robustness.

        Args:
            query: User search query.
            strategy: Chunking strategy to query ('character', 'structure', 'semantic').
            retrieve_k: Number of initial candidates to pull per retriever (default 20).
            final_k: Number of top re-ranked chunks to return (default 3).
            doc_ids: Optional list of document IDs to scope search.
            use_reranker: Whether to apply neural cross-encoder re-ranking (default True).
            retrieval_mode: 'hybrid' (BM25 + Dense RRF) or 'dense' (vector only).

        Returns:
            List of QueryResultItem instances.
        """
        if not query or not query.strip():
            return []

        # Stage 1A: Dense Bi-Encoder Retrieval
        query_embedding = self.embedding_model.encode_query(query)
        dense_candidates = self.store.query(
            strategy=strategy,
            query_embedding=query_embedding,
            n_results=retrieve_k,
            doc_ids=doc_ids,
        )

        if retrieval_mode == "hybrid":
            # Stage 1B: Sparse Lexical Retrieval (BM25)
            bm25_idx = self._get_bm25_index(strategy)
            lexical_candidates = bm25_idx.search(
                query=query,
                top_k=retrieve_k,
                doc_ids=doc_ids,
            )

            # Stage 1C: Reciprocal Rank Fusion
            candidates = reciprocal_rank_fusion(
                dense_results=dense_candidates,
                lexical_results=lexical_candidates,
                rrf_k=60,
                top_k=retrieve_k,
            )
        else:
            candidates = dense_candidates

        if not candidates:
            logger.info("No candidate chunks retrieved for query '%s' under strategy '%s'.", query, strategy)
            return []

        # Stage 2: Cross-Encoder Re-Ranking (optional)
        if use_reranker:
            reranked = self.reranker.rerank(
                query=query,
                candidates=candidates,
                top_k=final_k,
            )
        else:
            sort_key = "rrf_score" if retrieval_mode == "hybrid" else "dense_score"
            reranked = sorted(candidates, key=lambda c: c.get(sort_key, 0.0), reverse=True)[:final_k]

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
                bm25_score=c.get("bm25_score"),
                rrf_score=c.get("rrf_score"),
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