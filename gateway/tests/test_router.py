"""Unit tests for IntentRouter (BAAI/bge-small-en-v1.5 and exemplar banks)."""

import numpy as np
import pytest
from slm_gateway.router import DEFAULT_INTENTS_PATH, RoutingResult, get_router


@pytest.fixture(scope="module")
def router():
    """Shared router fixture across test module to avoid reloading weights repeatedly."""
    return get_router()


def test_intents_yaml_contains_all_intents_with_minimum_exemplars():
    """Verify intents.yaml contains at least 25 exemplars per intent."""
    import yaml

    with open(DEFAULT_INTENTS_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    expected_intents = {"general", "technical", "rag"}
    assert set(data.keys()) == expected_intents, (
        f"Expected intents {expected_intents}, got {set(data.keys())}"
    )

    for intent, examples in data.items():
        assert len(examples) >= 25, (
            f"Intent {intent} has only {len(examples)} examples (required >= 25)"
        )


def test_router_initialization(router):
    """Verify router loads model and precomputes normalized embeddings."""
    assert router.model is not None
    assert len(router.example_texts) >= 75  # 3 * 25 minimum
    assert router.example_embeddings.shape[0] == len(router.example_texts)
    assert router.example_embeddings.shape[1] == 384  # bge-small dimension

    # Check normalization: L2 norm of embeddings must be ~1.0
    norms = np.linalg.norm(router.example_embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


@pytest.mark.parametrize(
    "query,expected_intent",
    [
        ("Who was the first president of the United States?", "general"),
        ("Tell me an amusing joke about computers.", "general"),
        ("How do I implement a red-black tree in C++?", "technical"),
        ("Explain how the Raft consensus algorithm works.", "technical"),
        ("What does the uploaded company policy say about sabbatical leave?", "rag"),
        ("According to the attached PDF, what is the warranty period?", "rag"),
    ],
)
def test_router_classification_accuracy(router, query, expected_intent):
    """Test classification on clear representative queries across all 3 intents."""
    result = router.classify(query)
    assert isinstance(result, RoutingResult)
    assert result.intent == expected_intent
    assert result.confidence >= 0.50
    assert result.latency_ms > 0
    assert not result.fallback_applied


def test_router_threshold_fallback(router):
    """When confidence is below threshold, router must fall back to 'general'."""
    # Force an impossible threshold of 0.99
    result = router.classify("Explain how airplanes fly.", threshold=0.99)
    assert result.intent == "general"
    assert result.fallback_applied is True


def test_router_empty_or_whitespace_query(router):
    """Empty or whitespace queries must immediately return 'general' fallback."""
    result = router.classify("   ")
    assert result.intent == "general"
    assert result.confidence == 0.0
    assert result.fallback_applied is True
    assert result.route == "hf_local"


def test_router_to_dict_structure(router):
    """RoutingResult.to_dict() must return exact OpenAI metadata fields."""
    result = router.classify("Format output as JSON")
    d = result.to_dict()
    assert "intent" in d
    assert "confidence" in d
    assert "route" in d
    assert "latency_ms" in d
    assert isinstance(d["confidence"], float)
    assert isinstance(d["latency_ms"], float)
