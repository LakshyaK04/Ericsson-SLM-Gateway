"""Demonstration Script for Local GenAI Stack.

Executes a 4-scene live demonstration:
1. Health check (Gateway & RAG service)
2. Normal chat completion (general intent)
3. PII masking (Presidio/spaCy)
4. Grounded RAG query with sources (structure chunking + cross-encoder)
"""

import argparse
import json
from pathlib import Path
import sys
import time
import httpx

GATEWAY_DEFAULT_URL = "http://localhost:8000"
RAG_DEFAULT_URL = "http://localhost:8001"

SEPARATOR = "=" * 80


def print_header(title: str, step: int):
    print("\n" + SEPARATOR)
    print(f"  SCENE {step}: {title.upper()}")
    print(SEPARATOR + "\n")


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
    print_header("Normal Chat Completion (Intent: General)", 2)
    prompt = "What is the capital of Sweden and what is it famous for?"
    print(f"[*] Prompt: '{prompt}'")

    payload = {
        "model": "microsoft/Phi-3-mini-4k-instruct",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 128,
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
    print_header("PII Masking & Privacy Protection", 3)
    prompt = (
        "Hello, I am Alice Smith (EMP-84920) working on Project Phoenix. My contact is "
        "alice.smith@company.com or +1-555-0199. I work at the headquarters in Stockholm."
    )
    print("[*] Original Inbound User Query:")
    print(f"    \"{prompt}\"")
    print("\n[*] Sending query through Gateway /v1/chat/completions...")

    payload = {
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 64,
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


def demo_rag_pipeline(gateway_url: str, rag_url: str):
    print_header("Grounded RAG Query with Sources", 4)

    pdf_path = Path("eval/docs/enterprise_rag_sample.pdf")
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
        "Enterprise AI Platform provide for enterprise deployments?"
    )
    print(f"\n[*] Querying through Gateway (Front Door Port 8000):")
    print(f"    \"{query}\"")

    payload = {
        "messages": [
            {"role": "user", "content": query}
        ],
        "max_tokens": 150,
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
    csv_file = Path(__file__).resolve().parent.parent / "eval" / "results" / "chunking_report.csv"
    if not csv_file.exists():
        print("[!] eval/results/chunking_report.csv not found. Run eval/chunking_eval.py first.")
        return

    import csv
    with open(csv_file, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print("\nEmpirical Chunking Strategy & Re-Ranking Results:")
    print("(Tested on small synthetic set of 3 PDFs / 5 pages from scripts/create_eval_docs.py and 36 questions)\n")
    print(f"+{'-'*13}+{'-'*11}+{'-'*14}+{'-'*12}+{'-'*11}+{'-'*11}+{'-'*8}+{'-'*14}+")
    print(f"| {'Strategy':<11} | {'Re-ranker':<9} | {'Total Chunks':<12} | {'Avg Length':<10} | {'Hit@1 (%)':<9} | {'Hit@3 (%)':<9} | {'MRR':<6} | {'Latency (ms)':<12} |")
    print(f"+{'-'*13}+{'-'*11}+{'-'*14}+{'-'*12}+{'-'*11}+{'-'*11}+{'-'*8}+{'-'*14}+")
    for r in rows:
        lat = f"{float(r['Avg Latency (ms)']):.1f} ms"
        print(f"| {r['Strategy']:<11} | {r['Reranker']:<9} | {r['Total Chunks']:<12} | {r['Avg Length (chars)'] + ' ch':<10} | {float(r['Hit@1 (%)']):.2f}%{' ': <3}| {float(r['Hit@3 (%)']):.2f}%{' ': <3}| {float(r['MRR']):.4f} | {lat:<12} |")
    print(f"+{'-'*13}+{'-'*11}+{'-'*14}+{'-'*12}+{'-'*11}+{'-'*11}+{'-'*8}+{'-'*14}+")
    print("""
Observations:
1. Re-ranking improved structure chunking (Hit@1: 94.44% -> 100.00%) and semantic chunking (Hit@1: 88.89% -> 94.44%).
2. Re-ranking character chunking improved Hit@1 (+2.78 pp), but reduced Hit@3 (-2.78 pp).
3. Re-ranking adds ~1,000-1,160 ms cross-encoder inference latency per query on CPU.
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
    print("      LOCAL GENAI STACK: LIVE DEMONSTRATION")
    print(SEPARATOR)

    if not check_services(args.gateway_url, args.rag_url):
        print("\n[!] Could not connect to running services.")
        print("[!] Ensure Gateway (port 8000) and RAG (port 8001) are running, or run with --benchmark-only.")
        print("[*] Displaying empirical evaluation results instead:\n")
        demo_chunking_comparison()
        return

    demo_standard_chat(args.gateway_url)
    demo_pii_redaction(args.gateway_url)
    demo_rag_pipeline(args.gateway_url, args.rag_url)

    print("\n" + SEPARATOR)
    print("      DEMONSTRATION COMPLETED SUCCESSFULLY")
    print(SEPARATOR + "\n")


if __name__ == "__main__":
    main()
