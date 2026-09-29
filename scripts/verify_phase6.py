"""Live verification script for Phase 6 (Gateway + RAG integration).

Uploads a document to RAG service, asks a question through the Gateway only,
and verifies answer generation with sources, followed by a general question skipping RAG.
"""

import asyncio
from pathlib import Path
from fastapi.testclient import TestClient

import slm_gateway.main as gw_main
import rag_service.main as rag_main
from slm_gateway.config import settings as gw_settings
from rag_service.config import settings as rag_settings

# Connect gateway directly to RAG testclient
rag_client_http = TestClient(rag_main.app)
gw_client_http = TestClient(gw_main.app)


class MockLocalBackend:
    def is_ready(self) -> bool:
        return True

    def get_model_name(self) -> str:
        return "mock-phi3-mini"

    async def load(self):
        pass

    async def generate(self, messages, temperature=0.7, top_p=1.0, max_tokens=512):
        # Inspect system/user message
        user_content = ""
        for m in messages:
            if m.get("role") == "user":
                user_content = m.get("content", "")

        if "Capital of Sweden" in user_content or "capital of Sweden" in user_content:
            return "The capital of Sweden is Stockholm.", 15, 8, "stop"
        elif "Context:" in user_content:
            return (
                "Based on the provided documentation [1], the Ericsson AI Platform provides services "
                "for deploying and operating artificial intelligence applications.",
                120,
                24,
                "stop",
            )
        else:
            return "General response from model.", 20, 6, "stop"

    async def close(self):
        pass


def run_live_verification():
    print("=" * 80)
    print("PHASE 6 LIVE INTEGRATION VERIFICATION")
    print("=" * 80)

    # 1. Setup mock backend on Gateway
    backend = MockLocalBackend()
    gw_main.backend = backend
    gw_main.app.state.backend = backend

    # Setup RAG client in Gateway to point to RAG TestClient
    from slm_gateway.rag_client import RAGClient

    # Adapt TestClient as an async client proxy for in-process testing
    class TestClientRAGAdapter:
        async def get(self, url, timeout=None):
            path = url.split("8001")[-1]
            return rag_client_http.get(path)

        async def post(self, url, json=None, timeout=None, headers=None):
            path = url.split("8001")[-1]
            return rag_client_http.post(path, json=json, headers=headers)

    gw_main.rag_client_instance = RAGClient(gw_settings, client=TestClientRAGAdapter())

    # In RAG service, wire generation to call Gateway TestClient
    class TestClientGatewayAdapter:
        async def post(self, url, json=None, timeout=None, headers=None):
            path = url.split("8000")[-1]
            return gw_client_http.post(path, json=json, headers=headers)

    # Override generation client in RAG service
    orig_generate = rag_main.generate_grounded_answer
    async def hooked_generate(query, chunks, config=None, client=None):
        return await orig_generate(query, chunks, config=config, client=TestClientGatewayAdapter())

    rag_main.generate_grounded_answer = hooked_generate

    # 2. Upload sample document to RAG
    sample_pdf = Path("eval/docs/ericsson_rag_sample.pdf")
    with open(sample_pdf, "rb") as f:
        up_resp = rag_client_http.post(
            "/documents",
            files={"file": ("ericsson_rag_sample.pdf", f, "application/pdf")},
            data={"strategies": "character,structure,semantic"},
        )
    assert up_resp.status_code == 200, f"Upload failed: {up_resp.text}"
    up_data = up_resp.json()
    print(f"\n1. Ingested document: {up_data['filename']} (ID: {up_data['doc_id']})")
    print(f"   Chunks indexed: {up_data['chunk_counts']}")

    # 3. Query through Gateway ONLY with RAG intent
    rag_query = "Search the uploaded PDF manual for UPF specifications."
    print(f"\n2. Sending query to Gateway: '{rag_query}'")
    gw_resp = gw_client_http.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": rag_query}]},
    )
    assert gw_resp.status_code == 200, f"Gateway error: {gw_resp.text}"
    gw_data = gw_resp.json()

    print(f"\n   Gateway Response:")
    print(f"   Model: {gw_data['model']}")
    print(f"   Route: {gw_data['x_routing']}")
    print(f"   Answer: {gw_data['choices'][0]['message']['content']}")
    print(f"   Sources returned ({len(gw_data.get('x_sources') or [])}):")
    for s in gw_data.get("x_sources", []):
        print(f"     - [{s['chunk_id']}] Source: {s['source']} (Page {s['page']}), Dense: {s['dense_score']}, Rerank: {s['rerank_score']}")

    assert gw_data["x_routing"]["route"] == "rag_service"
    assert gw_data["x_sources"] is not None
    assert len(gw_data["x_sources"]) > 0

    # 4. Query through Gateway with General intent (skipping RAG)
    gen_query = "What is the capital of Sweden?"
    print(f"\n3. Sending general query to Gateway: '{gen_query}'")
    gen_resp = gw_client_http.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": gen_query}]},
    )
    assert gen_resp.status_code == 200
    gen_data = gen_resp.json()

    print(f"\n   Gateway Response:")
    print(f"   Model: {gen_data['model']}")
    print(f"   Route: {gen_data['x_routing']}")
    print(f"   Answer: {gen_data['choices'][0]['message']['content']}")
    print(f"   x_sources: {gen_data.get('x_sources')}")

    assert gen_data["x_routing"]["intent"] == "general"
    assert gen_data.get("x_sources") is None
    print("\nPhase 6 Live Verification Succeeded!")


if __name__ == "__main__":
    run_live_verification()
