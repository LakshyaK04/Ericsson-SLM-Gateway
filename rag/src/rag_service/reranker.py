"""Cross-encoder re-ranking module using BAAI/bge-reranker-base per Section 5.5."""

import logging
from typing import Any, Dict, List, Optional
import numpy as np
from sentence_transformers import CrossEncoder

from .config import Settings, settings

logger = logging.getLogger(__name__)


class Reranker:
    """Neural cross-encoder reranker scoring (query, chunk_text) pairs."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.RERANKER_MODEL_NAME
        logger.info("Initializing CrossEncoder Reranker with %s...", self.model_name)
        self.model = CrossEncoder(self.model_name)
        logger.info("CrossEncoder Reranker %s loaded successfully.", self.model_name)

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Score candidate chunks against query and return top_k sorted by rerank_score."""
        if not candidates:
            return []

        if len(candidates) <= 1:
            if candidates:
                candidates[0]["rerank_score"] = 1.0
            return candidates[:top_k]

        # Prepare sentence pairs
        pairs = [[query, c["text"]] for c in candidates]

        # Compute cross-encoder relevance scores
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Attach scores to candidate dicts
        scored_candidates = []
        for c, score in zip(candidates, scores):
            item = dict(c)
            # Convert numpy float to Python float rounded to 4 decimals
            item["rerank_score"] = round(float(score), 4)
            scored_candidates.append(item)

        # Sort descending by rerank score
        scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored_candidates[:top_k]


_reranker: Optional[Reranker] = None


def get_reranker(cfg: Optional[Settings] = None) -> Reranker:
    """Return or initialize the singleton CrossEncoder Reranker."""
    global _reranker
    if _reranker is None:
        model_name = cfg.RERANKER_MODEL_NAME if cfg else settings.RERANKER_MODEL_NAME
        _reranker = Reranker(model_name)
    return _reranker
