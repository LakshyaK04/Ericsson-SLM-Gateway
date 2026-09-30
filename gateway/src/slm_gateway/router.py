"""Semantic intent router using BAAI/bge-small-en-v1.5 and top-3 similarity scoring.

Scores inbound queries against exemplars in intents.yaml and determines the
target route, system prompt modifications, and metadata.
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import yaml

from .config import Settings, settings

logger = logging.getLogger(__name__)

DEFAULT_INTENTS_PATH = Path(__file__).parent / "intents.yaml"


@dataclass
class RoutingResult:
    """Detailed result of an intent routing classification."""
    intent: str
    confidence: float
    route: str
    latency_ms: float
    scores_by_intent: Dict[str, float] = field(default_factory=dict)
    fallback_applied: bool = False
    warning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary for OpenAI response metadata."""
        data = {
            "intent": self.intent,
            "confidence": round(self.confidence, 4),
            "route": self.route,
            "latency_ms": round(self.latency_ms, 2),
        }
        if self.warning:
            data["warning"] = self.warning
        return data


class IntentRouter:
    """Semantic intent router based on sentence embeddings and exemplar banks."""

    def __init__(
        self,
        config: Optional[Settings] = None,
        intents_path: Optional[Union[str, Path]] = None,
        model: Optional[SentenceTransformer] = None,
    ):
        self.config = config or settings
        self.threshold = float(self.config.ROUTER_THRESHOLD)
        
        if intents_path:
            self.intents_path = Path(intents_path)
        elif self.config.INTENTS_FILE:
            self.intents_path = Path(self.config.INTENTS_FILE)
        else:
            self.intents_path = DEFAULT_INTENTS_PATH

        self.model_name = getattr(self.config, "ROUTER_MODEL_NAME", "BAAI/bge-small-en-v1.5")
        
        # Load exemplar bank
        self.intent_examples: Dict[str, List[str]] = self._load_intents()
        
        # Load or attach SentenceTransformer model
        if model is not None:
            self.model = model
        else:
            logger.info("Initializing IntentRouter with model %s...", self.model_name)
            self.model = SentenceTransformer(self.model_name)

        # Precompute and cache exemplar embeddings
        self._precompute_embeddings()

    def _load_intents(self) -> Dict[str, List[str]]:
        """Load intent exemplars from YAML file."""
        if not self.intents_path.exists():
            raise FileNotFoundError(f"Intents file not found: {self.intents_path}")
        with open(self.intents_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Invalid intents format in {self.intents_path}")
        return data

    def _precompute_embeddings(self) -> None:
        """Embed and normalize all exemplars."""
        self.example_texts: List[str] = []
        self.example_intents: List[str] = []

        for intent, examples in self.intent_examples.items():
            for ex in examples:
                self.example_texts.append(ex)
                self.example_intents.append(intent)

        logger.info(
            "Precomputing embeddings for %d exemplars across %d intents (%s)...",
            len(self.example_texts),
            len(self.intent_examples),
            ", ".join(self.intent_examples.keys()),
        )
        self.example_embeddings = self.model.encode(
            self.example_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def classify(
        self,
        query: str,
        threshold: Optional[float] = None,
        top_k_per_intent: int = 3,
    ) -> RoutingResult:
        """Classify a user query using top-k cosine similarity aggregation.

        WHY: Instead of hardcoded keywords or an expensive LLM call for routing,
        we compute embedding cosine similarity against known exemplars. The mean of the
        top-3 matches identifies whether the user wants general chat, technical discussion,
        or document retrieval (RAG). If confidence is below threshold, it falls back to 'general'.
        """
        start_time = time.perf_counter()
        effective_threshold = threshold if threshold is not None else self.threshold

        if not query or not query.strip():
            latency = (time.perf_counter() - start_time) * 1000
            return RoutingResult(
                intent="general",
                confidence=0.0,
                route="hf_local",
                latency_ms=latency,
                scores_by_intent={i: 0.0 for i in self.intent_examples},
                fallback_applied=True,
            )

        # Embed query (normalized for cosine similarity)
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # Compute cosine similarities to all exemplars
        sims = cosine_similarity(query_embedding, self.example_embeddings)[0]

        # Aggregate mean of top-k similarities per intent
        scores_by_intent: Dict[str, float] = {}
        for intent in self.intent_examples.keys():
            intent_sims = [
                sims[idx]
                for idx, ex_intent in enumerate(self.example_intents)
                if ex_intent == intent
            ]
            if not intent_sims:
                scores_by_intent[intent] = 0.0
                continue

            k = min(top_k_per_intent, len(intent_sims))
            top_sims = sorted(intent_sims, reverse=True)[:k]
            scores_by_intent[intent] = float(np.mean(top_sims))

        # Select highest-scoring intent
        best_intent = max(scores_by_intent.items(), key=lambda x: x[1])[0]
        best_score = scores_by_intent[best_intent]

        # Fallback check
        fallback_applied = False
        if best_score < effective_threshold:
            final_intent = "general"
            fallback_applied = True
        else:
            final_intent = best_intent

        # Determine target route
        if final_intent == "rag":
            route = "rag"
        else:
            route = "hf_local"

        latency = (time.perf_counter() - start_time) * 1000

        return RoutingResult(
            intent=final_intent,
            confidence=best_score,
            route=route,
            latency_ms=latency,
            scores_by_intent=scores_by_intent,
            fallback_applied=fallback_applied,
        )


# Global singleton instance
_router: Optional[IntentRouter] = None


def get_router(
    cfg: Optional[Settings] = None,
    intents_path: Optional[Union[str, Path]] = None,
) -> IntentRouter:
    """Return or initialize the singleton IntentRouter."""
    global _router
    if _router is None or cfg is not None or intents_path is not None:
        _router = IntentRouter(cfg or settings, intents_path=intents_path)
    return _router