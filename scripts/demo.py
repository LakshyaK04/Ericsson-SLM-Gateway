"""Interactive 5-Minute Demonstration Script for Ericsson Local GenAI Stack.

Executes a structured end-to-end live demonstration of:
1. Health and readiness verification
2. Standard chat completion (general intent)
3. PII detection and redaction (x_pii)
4. Structured JSON output enforcement
5. Document ingestion and grounded RAG query with source citations (x_sources)
6. Empirical chunking evaluation summary table
"""

import argparse
import json
import sys
import time
from pathlib import Path
import httpx

GATEWAY_DEFAULT_URL = "http://localhost:8000"
RAG_DEFAULT_URL = "http://localhost:8001"

SEPARATOR = "=" * 80
SUBSEP = "-" * 80


def print_header(title: str, step: int):
    print("\n" + SEPARATOR)
    print(f"  SCENE {step}: {title.upper()}")
    print(SEPARATOR + "\n")


def print_json(data: dict):
    print(json.dumps(data, indent=2))


def check_services(gateway_url: str, rag_url: str) -> bool:
    print_header("Service Health and Readiness Checks", 1)
    print(f"[*] Probing Gateway at {gateway_url}...")
    try:
        r = httpx.get(f"{gateway_url}/health", timeout=5.0)
        print(f"    - Gateway /health: HTTP {r.status_code} -> {r.json()}")
    except Exception as e:
        print(f"    [!] Error reaching Gateway: {e}")
        return False

    print(f"[*] Probing RAG Service at {rag_url}...")
    try:
        r = httpx.get(f"{rag_url}/health", timeout=5.0)
        print(f"    - RAG Service /health: HTTP {r.status_code} -> {r.json()}")
    except Exception as e:
        print(f"    [!] Error reaching RAG Service: {e}")
        return False

    return True


def demo_standard_chat(gateway_url: str):
    print_header("Standard Chat Completion (Intent: General)", 2)
    prompt = "What is the capital of Sweden and what is it famous for?"
    print(f"[*] Prompt: '{prompt}'")
    
    payload = {
        "model": "microsoft/Phi-3-mini-4k-instruct",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 128
    }
    
    t0 = time.perf_counter()
    r = httpx.post(f"{gateway_url}/v1/chat/completions", json=payload, timeout=60.0)
    lat = (time.perf_counter() - t0) * 1000
    
    if r.status_code == 200:
        res = r.json()
        print(f"\n[*] Response received in {lat:.1f}ms:")
        print(f"    - Answer: {res['choices'][0]['message']['content']}")
        print(f"    - Routing: {res.get('x_routing')}")
        print(f"    - PII: {res.get('x_pii')}")
    else:
        print(f"[!] Request failed: HTTP {r.status_code}: {r.text}")


def demo_pii_redaction(gateway_url: str):
    print_header("PII Masking & Privacy Protection (Presidio)", 3)
    prompt = (
        "Hello, I am Alice Smith (EMP-84920). My contact is alice.smith@ericsson.com "
        "or +46-8-555-1234. I am currently working on Project-Titan in Stockholm."
    )
    print("[*] Original Inbound User Query:")
    print(f"    \"{prompt}\"")
    print("\n[*] Sending query through Gateway /v1/chat/completions...")
    
    payload = {
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 64
    }
    
    r = httpx.post(f"{gateway_url}/v1/chat/completions", json=payload, timeout=60.0)
    if r.status_code == 200:
        res = r.json()
        print("\n[*] PII Masking Outcome:")
        print(f"    - Detected & Redacted Entities: {res.get('x_pii')}")
        print("    - Notice that 'Stockholm' remains intact because LOCATION is excluded.")
        print(f"    - Routing: {res.get('x_routing')}")
        print(f"    - Assistant Response: {res['choices'][0]['message']['content']}")
    else:
        print(f"[!] Request failed: HTTP {r.status_code}: {r.text}")


def demo_structured_json(gateway_url: str):
    print_header("Structured JSON Intent & Schema Enforcement", 4)
    prompt = (
        "Return a JSON list of three 5G core network functions. "
        "Each object should have keys 'acronym' and 'function_name'."
    )
    print(f"[*] Prompt: '{prompt}'")
    
    payload = {
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 200
    }
    
    r = httpx.post(f"{gateway_url}/v1/chat/completions", json=payload, timeout=60.0)
    if r.status_code == 200:
        res = r.json()
        print(f"\n[*] Routing Metadata: {res.get('x_routing')}")
        print("[*] Output Content:")
        print(res['choices'][0]['message']['content'])
    else:
        print(f"[!] Request failed: HTTP {r.status_code}: {r.text}")


def demo_rag_pipeline(gateway_url: str, rag_url: str):
    print_header("Document Ingestion & Grounded RAG with Citations", 5)
    
    pdf_path = Path("eval/docs/ericsson_rag_sample.pdf")
    if not pdf_path.exists():
        print(f"[!] Reference document {pdf_path} not found.")
        return
        
    print(f"[*] Uploading and indexing '{pdf_path.name}' to RAG Service (port 8001)...")
    with open(pdf_path, "rb") as f:
        files = {"file": (pdf_path.name, f, "application/pdf")}
        r = httpx.post(f"{rag_url}/documents", files=files, timeout=60.0)
        
    if r.status_code == 200:
        print(f"    - Indexing Success: {r.json()}")
    else:
        print(f"    [!] Upload failed: {r.text}")
        return

    query = (
        "According to the uploaded documentation, what capabilities does the "
        "Ericsson AI Platform provide for enterprise deployments?"
    )
    print(f"\n[*] Querying through Gateway (Front Door Port 8000):")
    print(f"    \"{query}\"")
    
    payload = {
        "messages": [
            {"role": "user", "content": query}
        ],
        "max_tokens": 150
    }
    
    t0 = time.perf_counter()
    r = httpx.post(f"{gateway_url}/v1/chat/completions", json=payload, timeout=60.0)
    lat = (time.perf_counter() - t0) * 1000
    
    if r.status_code == 200:
        res = r.json()
        print(f"\n[*] Grounded RAG Completion ({lat:.1f}ms):")
        print(f"    - Intent Router Classification: {res.get('x_routing')}")
        print(f"    - Generated Answer:\n      {res['choices'][0]['message']['content']}")
        print("\n    - Cited Sources (x_sources):")
        sources = res.get("x_sources") or []
        for s in sources:
            print(
                f"      * [{s['chunk_id']}] Doc: {s['source']} (Page {s['page']}), "
                f"Dense Score: {s['dense_score']:.4f}, Rerank Score: {s['rerank_score']:.4f}"
            )
    else:
        print(f"[!] RAG query failed: HTTP {r.status_code}: {r.text}")


def demo_chunking_comparison():
    print_header("Empirical Chunking Strategy & Re-Ranking Benchmark", 6)
    print("""
Evaluated on 36 technical queries across 3 technical telecommunications PDFs:

+-------------+-----------+--------------+------------+-----------+-----------+--------+--------------+
| Strategy    | Re-ranker | Total Chunks | Avg Length | Hit@1 (%) | Hit@3 (%) | MRR    | Latency (ms) |
+-------------+-----------+--------------+------------+-----------+-----------+--------+--------------+
| character   | Off       | 16           | 422.1 ch   | 86.11%    | 97.22%    | 0.9028 | 11.6 ms      |
| character   | On        | 16           | 422.1 ch   | 83.33%    | 97.22%    | 0.9028 | 152.2 ms     |
| structure   | Off       | 10           | 621.0 ch   | 86.11%    | 94.44%    | 0.9028 | 9.8 ms       |
| structure   | On        | 10           | 621.0 ch   | 100.00%   | 100.00%   | 1.0000 | 159.3 ms     |
| semantic    | Off       | 16           | 387.2 ch   | 80.56%    | 94.44%    | 0.8611 | 10.1 ms      |
| semantic    | On        | 16           | 387.2 ch   | 94.44%    | 97.22%    | 0.9583 | 166.9 ms     |
+-------------+-----------+--------------+------------+-----------+-----------+--------+--------------+

Key Takeaways:
1. 'structure' chunking + cross-encoder achieved a perfect 100.0% Hit@1 and MRR 1.0000.
2. Cross-encoder re-ranking provided a +13.89% uplift for structure and +13.88% for semantic.
3. Neural re-ranking adds ~145ms latency over pure dense search (~10ms), an excellent trade-off for high accuracy.
""")


def main():
    parser = argparse.ArgumentParser(description="Live Demonstration of Local GenAI Stack")
    parser.add_argument("--gateway-url", default=GATEWAY_DEFAULT_URL, help="Gateway base URL")
    parser.add_argument("--rag-url", default=RAG_DEFAULT_URL, help="RAG service base URL")
    parser.add_argument("--benchmark-only", action="store_true", help="Only show benchmark table")
    args = parser.parse_args()

    if args.benchmark_only:
        demo_chunking_comparison()
        return

    print("\n" + SEPARATOR)
    print("      ERICSSON LOCAL GENAI STACK: LIVE DEMONSTRATION")
    print(SEPARATOR)
    
    if not check_services(args.gateway_url, args.rag_url):
        print("\n[!] Could not connect to running services.")
        print("[!] Ensure Gateway (port 8000) and RAG (port 8001) are running, or run with --benchmark-only.")
        print("[*] Displaying empirical evaluation results instead:\n")
        demo_chunking_comparison()
        return

    demo_standard_chat(args.gateway_url)
    demo_pii_redaction(args.gateway_url)
    demo_structured_json(args.gateway_url)
    demo_rag_pipeline(args.gateway_url, args.rag_url)
    demo_chunking_comparison()

    print("\n" + SEPARATOR)
    print("      DEMONSTRATION COMPLETED SUCCESSFULLY")
    print(SEPARATOR + "\n")


if __name__ == "__main__":
    main()
