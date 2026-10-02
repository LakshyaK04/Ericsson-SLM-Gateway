# Resume & Portfolio Guide: Local GenAI Gateway & Hybrid RAG

This document provides ready-to-paste resume bullet points, quantifiable metrics, system design justifications, and technical interview talking points for this project.

---

## 1. Resume Bullet Points (Copy & Paste Ready)

### For AI / GenAI / LLM Engineer Roles
* **Architected an enterprise-grade, privacy-first SLM Gateway and Document RAG Stack** deploying open-weights models (`Phi-3-mini-4k-instruct`, `bge-small-en-v1.5`, `bge-reranker-base`) with 100% on-premise execution and zero cloud data egress.
* **Engineered a 2-stage Hybrid Retrieval Engine (BM25 sparse + BGE dense vector search)** fused via Reciprocal Rank Fusion (RRF) and neural cross-encoder reranking, improving Hit@1 retrieval precision from 83.3% to **100.0%** on domain technical documentation.
* **Built real-time token streaming (`stream=true`)** via Server-Sent Events (SSE) conforming to the OpenAI chunk protocol, decoupling async token generation from HTTP transport to minimize Time to First Token (TTFT).
* **Developed automated few-shot semantic intent routing** (93.8% classification accuracy, sub-15ms latency) to dynamically route queries between local SLM generation and document RAG pipelines.
* **Integrated fail-closed Presidio PII sanitizer** with custom regex/deny-list recognizers for employee IDs and project codenames, achieving **100% recall and 0% false positives** on conversational benchmarks while preserving factual location/date entities.

### For Machine Learning Infrastructure & Backend Roles
* **Built high-throughput, OpenAI-compatible FastAPI gateway** featuring 4-bit NF4 bitsandbytes GPU quantization and an `asyncio.Semaphore(1)` concurrency lock to eliminate GPU OOM errors and race conditions.
* **Designed Prometheus observability telemetry** exposing `/metrics` to monitor request rates by intent, PII redaction frequencies, token throughput, and end-to-end inference latency histograms.
* **Authored comprehensive evaluation pipelines** benchmarking character, structural, and semantic chunking strategies across 36 query pairs, plus automated pre-flight train/test data leakage assertions.
* **Constructed an interactive dark-mode Web Playground UI** with real-time SSE typewriter streaming, side-by-side PII sanitizer inspection, and visual hybrid score breakdown (Dense, BM25, RRF, Rerank).

---

## 2. Key Quantifiable Metrics for Interviews

| Area | Baseline / Without Feature | With System Feature | Impact |
|---|---|---|---|
| **Retrieval Hit@1** | 83.3% (Character chunking) | **100.0%** (Structure + Cross-Encoder) | **+16.7% precision gain** |
| **Re-Ranking Precision** | 86.1% (Dense top-20 only) | **100.0%** (Cross-encoder top-3) | Zero false-positive context |
| **PII Redaction Recall** | 0% (Raw unredacted text) | **100.0%** (33/33 test cases) | Zero egress of sensitive IDs/names |
| **PII False-Positive Rate** | N/A | **0.0%** (Preserves locations/dates) | Unmangled natural conversations |
| **Router Accuracy** | Keyword heuristics (unstable) | **93.75%** (45/48 test queries) | Reliable automated delegation |
| **Router Latency** | Cloud LLM routing (1200+ ms) | **12.3 ms** (In-process BGE-small) | **~100x lower latency** |
| **GPU Memory Footprint** | ~7.8 GB (FP16 Phi-3) | **~2.4 GB** (4-bit NF4 quantized) | Fits easily on consumer/edge GPUs |

---

## 3. High-Yield Technical Interview Q&A

### Q1: Why did you choose in-process Hugging Face generation instead of just calling an external API?
> **Answer**:  
> *"In enterprise telecommunications and healthcare environments, strict data sovereignty regulations (e.g., GDPR, internal IP secrecy) prohibit sending proprietary technical specifications or employee PII to cloud endpoints. Running Phi-3-mini locally with 4-bit NF4 quantization keeps inference completely on-premise (~2.4 GB VRAM footprint) while providing an OpenAI-compatible interface so downstream applications can swap `base_url` seamlessly."*

### Q2: Why implement Hybrid Retrieval (BM25 + Dense RRF) instead of pure vector search?
> **Answer**:  
> *"Dense vector bi-encoders (like BGE-small) excel at semantic similarity (understanding that 'sabbatical' relates to 'leave policy'). However, they frequently fail on exact keywords, part numbers, error codes, and telecom acronyms (e.g., `EMP-12345`, `gNodeB 503`, `QoS flow steering`), where dense vectors smooth over exact characters.  
> By pairing BM25 Okapi sparse search with BGE dense search and fusing candidate rankings using Reciprocal Rank Fusion (RRF: $\sum \frac{1}{60 + r}$), we capture both conceptual semantics and exact lexical hits. Feeding these fused candidates into `bge-reranker-base` ensures the top 3 chunks are mathematically optimal."*

### Q3: How did you prevent infinite loops between the Gateway and RAG service?
> **Answer**:  
> *"The Gateway is the front door for users on port 8000, and it routes queries about documents to the RAG service on port 8001. To generate grounded answers, the RAG service must call the Gateway's `/v1/chat/completions` endpoint for Phi-3 inference.  
> If the Gateway routed that second request back to RAG, it would trigger an infinite HTTP loop. We solved this architecturally by injecting an `X-Bypass-Router: true` header in the internal RAG client. The Gateway checks for this header and immediately bypasses semantic routing, sending the grounded prompt straight to the local model."*

### Q4: How does your SSE streaming pipeline handle concurrency and GPU limits?
> **Answer**:  
> *"Phi-3 model inference in Hugging Face is synchronous and GPU-bound. We wrapped generation in `async with self._semaphore:` with a semaphore capacity of 1 to ensure thread safety on a single GPU.  
> To stream tokens without blocking FastAPI's async event loop, we bridged Hugging Face's `TextIteratorStreamer` running in a worker thread to an `asyncio.Queue` using `loop.call_soon_threadsafe()`. This allowed the FastAPI endpoint to yield Server-Sent Events (`data: {json}\n\n`) as async generators while tracking Time to First Token (TTFT)."*

### Q5: What is your fail-closed PII policy, and why did you exclude LOCATION and DATE_TIME?
> **Answer**:  
> *"By default, spaCy/Presidio redacts entities like `LOCATION` and `DATE_TIME`. In testing, we discovered that standard questions like 'What is the capital of Germany?' became 'What is the capital of `<LOCATION>`?', and 'When is the Monday release?' became 'When is the `<DATE_TIME>` release?'. This degraded router confidence and produced confused model completions.  
> We restricted redaction to high-risk identifiers (`PERSON`, `EMAIL_ADDRESS`, `PHONE_NUMBER`, `CREDIT_CARD`, `IP_ADDRESS`, `EMPLOYEE_ID`, `PROJECT_CODENAME`). We also enforced a fail-closed architecture (`PII_FAIL_MODE=closed`): if the Presidio analyzer throws an exception, the request fails with HTTP 500 rather than silently leaking raw PII to the model backend."*

---

## 4. Architectural System Diagram

```
                     ┌──────────────────────────────────────┐
                     │          User / Web Client           │
                     └──────────────────┬───────────────────┘
                                        │ HTTP / SSE Stream
                                        ▼
    ┌───────────────────────────────────────────────────────────────────────┐
    │                      SLM Gateway (Port 8000)                          │
    │                                                                       │
    │  [Bearer Auth] ──> [PII Sanitizer] ──> [BGE Semantic Router]          │
    │  (Optional key)   (Presidio/Regex)    (93.8% intent accuracy)        │
    │                                            │                          │
    │                       ┌────────────────────┴────────────────────┐     │
    │                       ▼                                         ▼     │
    │              [Direct SLM Path]                         [RAG Route]    │
    │              (general/technical)                                │     │
    │                       │                                         │     │
    │                       ▼                                         ▼     │
    │             ┌──────────────────┐                     ┌──────────────┐ │
    │             │  Phi-3 Mini 4bit │                     │  RAG Client  │ │
    │             │ (In-Process NF4) │                     └──────┬───────┘ │
    └─────────────┴─────────▲────────┴────────────────────────────┼─────────┘
                            │                                     │
                   X-Bypass-Router: true                          │ HTTP
                            │                                     │
    ┌───────────────────────┴─────────────────────────────────────▼─────────┐
    │                       RAG Service (Port 8001)                         │
    │                                                                       │
    │  [Parsers] ──> [Chunking Engine] ──> [Vector & Lexical Store]         │
    │  (PyMuPDF/     (Character /          - ChromaDB (Dense Cosine)        │
    │   python-docx)  Structure /          - BM25 Index (Sparse Lexical)    │
    │                 Semantic)                         │                   │
    │                                                   ▼                   │
    │                                     [Reciprocal Rank Fusion (RRF)]    │
    │                                                   │                   │
    │                                                   ▼                   │
    │                                     [bge-reranker-base Cross-Encoder] │
    │                                                   │                   │
    │                                                   ▼                   │
    │                                      [Grounded Prompt Assembler]      │
    └───────────────────────────────────────────────────────────────────────┘
```
