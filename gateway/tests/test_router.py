"""Tests for the intent router."""

from slm_gateway.router import IntentRouter


def test_router_intents():
    """Each test query should be classified to the expected intent."""
    router = IntentRouter()

    test_cases = [
        ("What is the capital of Germany?", "general"),
        ("Who discovered electricity?", "general"),
        ("Explain the water cycle.", "general"),
        ("What is the tallest mountain?", "general"),

        ("How do I create a Python function?", "technical"),
        ("How does an HTTP API work?", "technical"),
        ("Explain SQL indexing.", "technical"),
        ("How do I deploy a Docker container?", "technical"),

        ("What does the uploaded PDF say about security?", "rag"),
        ("Summarize the company document.", "rag"),
        ("According to the uploaded report, what is the deadline?", "rag"),
        ("Find the leave policy in the document.", "rag"),
    ]

    for query, expected in test_cases:
        result = router.classify(query)
        assert result["intent"] == expected, (
            f"Query: {query!r} — expected {expected!r}, "
            f"got {result['intent']!r}"
        )