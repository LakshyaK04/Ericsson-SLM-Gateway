"""Tests for grounded generation and /answer endpoint in RAG service."""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient
from rag_service.config import Settings
from rag_service.generation import format_context_blocks, generate_grounded_answer
from rag_service.main import app
from rag_service.schemas import QueryResultItem


def test_format_context_blocks():
    """Verify context blocks are formatted with numbered brackets and source attribution."""
    chunks = [
        QueryResultItem(
            chunk_id="c1",
            text="First sentence.",
            source="doc1.pdf",
            page=1,
            strategy="structure",
            dense_score=0.9,
            rerank_score=0.95,
        ),
        QueryResultItem(
            chunk_id="c2",
            text="Second sentence.",
            source="doc2.pdf",
            page=3,
            strategy="structure",
            dense_score=0.8,
            rerank_score=0.85,
        ),
    ]

    formatted = format_context_blocks(chunks)
    assert "[1] Source: doc1.pdf (Page 1)" in formatted
    assert "First sentence." in formatted
    assert "[2] Source: doc2.pdf (Page 3)" in formatted
    assert "Second sentence." in formatted


@pytest.mark.asyncio
async def test_generate_grounded_answer_empty_chunks_refuses():
    """When no chunks are provided, answer should cleanly refuse without HTTP call."""
    res = await generate_grounded_answer("What is X?", [])
    assert "The provided documents do not contain enough information" in res.answer
    assert len(res.sources) == 0
    assert res.usage.total_tokens == 0


@pytest.mark.asyncio
async def test_generate_grounded_answer_calls_gateway_with_bypass_header():
    """Verify HTTP call to gateway includes X-Bypass-Router header and parses answer."""
    chunks = [
        QueryResultItem(
            chunk_id="c1",
            text="UPF is responsible for packet forwarding.",
            source="5g_spec.pdf",
            page=1,
            strategy="structure",
            dense_score=0.88,
            rerank_score=0.92,
        )
    ]

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "According to [1], UPF handles packet forwarding.",
                }
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 15,
            "total_tokens": 115,
        },
    }

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.return_value = mock_resp

    res = await generate_grounded_answer("What does UPF do?", chunks, client=mock_client)

    # Verify mock call parameters
    mock_client.post.assert_called_once()
    call_args = mock_client.post.call_args
    assert call_args.kwargs["headers"]["X-Bypass-Router"] == "true"
    assert "What does UPF do?" in call_args.kwargs["json"]["messages"][1]["content"]

    # Verify response parsing
    assert res.answer == "According to [1], UPF handles packet forwarding."
    assert len(res.sources) == 1
    assert res.sources[0].source == "5g_spec.pdf"
    assert res.usage.total_tokens == 115


def test_answer_endpoint(monkeypatch):
    """POST /answer retrieves chunks and returns grounded AnswerResponse."""
    client = TestClient(app)

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        QueryResultItem(
            chunk_id="c_test",
            text="Grounded factual sentence.",
            source="manual.pdf",
            page=1,
            strategy="structure",
            dense_score=0.89,
            rerank_score=0.94,
        )
    ]

    import rag_service.main as main_mod

    main_mod.retriever = mock_retriever

    async def mock_generate(query, chunks, config=None, client=None):
        from rag_service.schemas import AnswerResponse, SourceItem, UsageInfo

        return AnswerResponse(
            answer="Grounded answer based on [1].",
            sources=[
                SourceItem(
                    chunk_id="c_test",
                    source="manual.pdf",
                    page=1,
                    strategy="structure",
                    dense_score=0.89,
                    rerank_score=0.94,
                    text="Grounded factual sentence.",
                )
            ],
            usage=UsageInfo(prompt_tokens=80, completion_tokens=10, total_tokens=90),
        )

    monkeypatch.setattr(main_mod, "generate_grounded_answer", mock_generate)

    payload = {
        "query": "What does the manual state?",
        "strategy": "structure",
    }
    resp = client.post("/answer", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "Grounded answer based on [1]."
    assert len(data["sources"]) == 1
    assert data["sources"][0]["source"] == "manual.pdf"
    assert data["usage"]["total_tokens"] == 90


def test_apply_token_budget_guard():
    """apply_token_budget_guard drops lowest-ranked chunks or trims to satisfy max_budget."""
    from rag_service.generation import apply_token_budget_guard
    from rag_service.schemas import QueryResultItem

    chunks = [
        QueryResultItem(
            chunk_id="c1",
            text="Chunk one content about 5G radio access network.",
            source="doc.pdf",
            page=1,
            strategy="structure",
            dense_score=0.9,
            rerank_score=0.95,
        ),
        QueryResultItem(
            chunk_id="c2",
            text="Chunk two content about core UPF user plane function.",
            source="doc.pdf",
            page=2,
            strategy="structure",
            dense_score=0.8,
            rerank_score=0.85,
        ),
        QueryResultItem(
            chunk_id="c3",
            text="Chunk three content about network slice management.",
            source="doc.pdf",
            page=3,
            strategy="structure",
            dense_score=0.7,
            rerank_score=0.75,
        ),
    ]

    # Generous budget keeps all chunks
    kept_all, block_all = apply_token_budget_guard(chunks, max_budget=4000, query="What is 5G?")
    assert len(kept_all) == 3
    assert "[1]" in block_all and "[2]" in block_all and "[3]" in block_all

    # Budget sized to fit only chunk 1
    from rag_service.generation import SYSTEM_PROMPT, count_tokens

    base_t = count_tokens(f"{SYSTEM_PROMPT}\nQuestion: What is 5G?\n\nContext:\n\n\nAnswer:")
    c1_tokens = count_tokens(f"[1] Source: doc.pdf (Page 1)\n{chunks[0].text.strip()}")

    kept_one, block_one = apply_token_budget_guard(
        chunks, max_budget=base_t + c1_tokens + 5, query="What is 5G?"
    )
    assert len(kept_one) == 1
    assert kept_one[0].chunk_id == "c1"
    assert "[2]" not in block_one


@pytest.mark.asyncio
async def test_insufficient_context_refusal():
    """generate_grounded_answer returns refusal when top reranker score is below threshold."""
    from rag_service.generation import generate_grounded_answer
    from rag_service.schemas import QueryResultItem

    cfg = Settings(RERANKER_REFUSAL_THRESHOLD=0.5)
    low_score_chunks = [
        QueryResultItem(
            chunk_id="c_low",
            text="Completely irrelevant topic.",
            source="doc.pdf",
            page=1,
            strategy="structure",
            dense_score=0.2,
            rerank_score=0.15,
        )
    ]

    resp = await generate_grounded_answer(
        query="What is the encryption key?",
        chunks=low_score_chunks,
        config=cfg,
    )
    assert (
        resp.answer
        == "The provided documents do not contain enough information to answer this question."
    )
    assert len(resp.sources) == 1
    assert resp.usage.total_tokens == 0


def test_answer_endpoint_streaming(monkeypatch):
    """POST /answer with stream=True returns StreamingResponse with text/event-stream media type."""
    from unittest.mock import MagicMock

    from rag_service.schemas import QueryResultItem

    client = TestClient(app)
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        QueryResultItem(
            chunk_id="c1",
            text="Sample content.",
            source="doc.pdf",
            page=1,
            strategy="structure",
            dense_score=0.9,
            rerank_score=0.9,
        )
    ]
    import rag_service.main as main_mod

    main_mod.retriever = mock_retriever

    async def mock_stream_answer(query, chunks, config=None, client=None, request_id=None):
        yield 'data: {"choices": [{"delta": {"role": "assistant", "content": ""}}]}\n\n'
        yield 'data: {"choices": [{"delta": {"content": "Streamed answer."}}]}\n\n'
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(main_mod, "stream_grounded_answer", mock_stream_answer)

    resp = client.post("/answer", json={"query": "test query", "stream": True})
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    assert "Streamed answer." in resp.text
    assert "[DONE]" in resp.text
