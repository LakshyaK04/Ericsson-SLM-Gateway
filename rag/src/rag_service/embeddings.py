"""Dense embedding model using BAAI/bge-small-en-v1.5 per Section 5.5."""

import logging
from typing import List, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import Settings, settings

logger = logging.getLogger(__name__)

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class EmbeddingModel:
    """Singleton-ready dense embedding wrapper with query instruction prefixing."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        logger.info("Initializing EmbeddingModel with %s...", self.model_name)
        self.model = SentenceTransformer(self.model_name)
        logger.info("EmbeddingModel %s loaded.", self.model_name)

    def encode_documents(self, texts: List[str]) -> List[List[float]]:
        """Encode document passages without query prefix, with L2 normalization."""
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def encode_queries(self, queries: List[str]) -> List[List[float]]:
        """Encode queries with BGE instruction prefix, with L2 normalization."""
        if not queries:
            return []
        prefixed = [f"{BGE_QUERY_PREFIX}{q}" for q in queries]
        embeddings = self.model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def encode_query(self, query: str) -> List[float]:
        """Encode a single query with prefix and L2 normalization."""
        return self.encode_queries([query])[0]

    def encode(self, texts: List[str], normalize_embeddings: bool = True) -> np.ndarray:
        """Generic encode method for compatibility with semantic chunking."""
        return self.model.encode(
            texts,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=False,
        )


_embedding_model: Optional[EmbeddingModel] = None


def get_embedding_model(cfg: Optional[Settings] = None) -> EmbeddingModel:
    """Return or initialize the singleton EmbeddingModel."""
    global _embedding_model
    if _embedding_model is None:
        model_name = cfg.EMBEDDING_MODEL_NAME if cfg else settings.EMBEDDING_MODEL_NAME
        _embedding_model = EmbeddingModel(model_name)
    return _embedding_model