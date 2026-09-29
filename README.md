# Ericsson Local GenAI Stack: SLM Gateway & Hybrid RAG Sandbox

An enterprise-grade, privacy-first, fully local Generative AI infrastructure combining an **OpenAI-Compatible SLM Gateway** with an **Ingestion Pipeline and Hybrid RAG Sandbox**. Built to run 100% locally with open models (`microsoft/Phi-3-mini-4k-instruct`, `BAAI/bge-small-en-v1.5`, `BAAI/bge-reranker-base`), ensuring zero external cloud exposure and strict data sovereignty.

---

## 1. System Architecture

```mermaid
graph TD
    User([Enterprise Client / Application]) -->|POST /v1/chat/completions| GW[SLM Gateway :8000]
    
    subgraph Gateway Core [:8000]
        Auth[Optional Bearer Auth]
        PII[Presidio PII Redactor<br/>Fail-Closed, Custom Recognizers]
        Router[Semantic Intent Router<br/>BGE-small Embeddings]
        LocalLLM[In-Process Phi-3 Mini 4-bit<br/>Single-GPU Semaphore]
    end
    
    GW --> Auth --> PII --> Router
    Router -->|general / technical| LocalLLM
    Router -->|structured_json| JSONRules[Inject JSON Constraint] --> LocalLLM
    Router -->|rag intent| RAGClient[RAG HTTP Client]
    
    subgraph Hybrid RAG Service [:8001]
        Ingest[Document Ingestion<br/>PDF & DOCX Parsers]
        Chunkers[Chunking Strategies<br/>Character | Structure | Semantic]
        Chroma[(ChromaDB Collections)]
        DenseSearch[BGE Dense Retrieval k=20]
        NeuralRerank[BGE Cross-Encoder k=3]
        ContextBuilder[Grounded Prompt Builder<br/>[1], [2] Citations]
    end
    
    RAGClient -->|POST /answer| ContextBuilder
    ContextBuilder --> DenseSearch --> Chroma
    DenseSearch --> NeuralRerank --> ContextBuilder
    ContextBuilder -->|POST /v1/chat/completions<br/>X-Bypass-Router: true| LocalLLM
    
    LocalLLM --> GW
    GW -->|OpenAI-Compatible Response<br/>+ x_routing, x_pii, x_sources| User
```

---

## 2. Key Capabilities & Technical Highlights

| Component | Capabilities | Technical Foundation |
|---|---|---|
| **SLM Gateway** | OpenAI-compatible `/v1/chat/completions` API, in-process quantized serving, token tracking. | FastAPI, HuggingFace Transformers, `bitsandbytes` (4-bit NF4). |
| **PII Protection** | Redacts personal identifiers and proprietary corporate codes (`EMPLOYEE_ID`, `PROJECT_CODENAME`). | Microsoft Presidio, spaCy `en_core_web_sm`, fail-closed policy. |
| **Semantic Router** | Classifies query intent across 4 categories (`general`, `technical`, `structured_json`, `rag`) in ~60ms. | `BAAI/bge-small-en-v1.5`, cosine similarity top-3 mean, threshold sweep. |
| **Hybrid RAG Service** | Multi-format document ingestion, isolated vector collections, two-stage neural re-ranking. | PyMuPDF, python-docx, ChromaDB, `BAAI/bge-reranker-base`. |
| **Grounded Generation** | Strict grounding prompt with numbered source citations (`[1]`, `[2]`), loop prevention. | Header `X-Bypass-Router: true`, provenance tracking in `x_sources`. |

---

## 3. Empirical Evaluation Highlights

All metrics are experimentally measured from reproducible automated test harnesses:

### 3.1 Chunking Strategy & Neural Re-Ranking (Phase 5 Benchmark)
*Evaluated on 36 technical telecom questions across 3 multi-page specifications (`eval/docs/`):*

| Strategy | Re-ranker | Total Chunks | Avg Length | Hit@1 (%) | Hit@3 (%) | MRR | Latency (ms) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `character` | Off | 16 | 422.1 ch | 86.11% | 97.22% | 0.9028 | 11.6 |
| `character` | On | 16 | 422.1 ch | 83.33% | 97.22% | 0.9028 | 152.2 |
| `structure` | Off | 10 | 621.0 ch | 86.11% | 94.44% | 0.9028 | 9.8 |
| **`structure`** | **On** | **10** | **621.0 ch** | **100.00%** | **100.00%** | **1.0000** | **159.3** |
| `semantic` | Off | 16 | 387.2 ch | 80.56% | 94.44% | 0.8611 | 10.1 |
| **`semantic`** | **On** | **16** | **387.2 ch** | **94.44%** | **97.22%** | **0.9583** | **166.9** |

### 3.2 Semantic Intent Router Accuracy (Phase 3 Benchmark)
*Evaluated on 64 out-of-distribution queries with 0 exemplar leakage (`eval/datasets/router_eval.jsonl`):*
- **Overall Accuracy:** `93.75%` (F1-score: `general` 96.8%, `structured_json` 94.1%, `rag` 93.3%, `technical` 90.9%).
- **Operating Threshold:** `0.55` (empirically derived via 0.30 - 0.80 sweep).
- **Mean Classification Latency:** `66.83 ms`.

### 3.3 PII Redaction Precision & Safety (Phase 2 Benchmark)
*Evaluated on 45 test cases including tricky edge cases (`eval/datasets/pii_eval.jsonl`):*
- **Entity Detection Recall:** `100.00%` across standard and custom enterprise entities (`EMPLOYEE_ID`, `PROJECT_CODENAME`).
- **False Positive Rate on Clean Text:** `0.00%` (0 / 10 clean queries redacted). Geographic names (`LOCATION`) and dates (`DATE_TIME`) remain untouched.

---

## 4. Quickstart

### Prerequisites
- Python 3.10
- [uv](https://docs.astral.sh/uv/) package manager (recommended) or standard virtualenv
- NVIDIA GPU with CUDA 12+ recommended for 4-bit in-process model serving (or CPU fallback via `BACKEND=openai_compatible`)

### Option A: Local Execution (Fastest)

1. **Clone and Install:**
   ```bash
   git clone https://github.com/lakshyakapoor/ericsson-genai-stack.git
   cd ericsson-genai-stack
   uv sync
   ```

2. **Launch Services:**
   - **Terminal 1 (RAG Service):**
     ```bash
     uv run uvicorn rag_service.main:app --host 0.0.0.0 --port 8001
     ```
   - **Terminal 2 (SLM Gateway):**
     ```bash
     uv run uvicorn slm_gateway.main:app --host 0.0.0.0 --port 8000
     ```

3. **Verify:**
   ```bash
   curl http://localhost:8000/health
   curl http://localhost:8001/health
   ```

### Option B: Docker Compose

```bash
docker compose up --build
```
*(To enable GPU passthrough in Docker on Linux/WSL2, install `nvidia-container-toolkit` and uncomment the `deploy.resources.reservations` block in `docker-compose.yml`)*

---

## 5. 5-Minute Live Demonstration

Run the automated interactive demonstration script:
```bash
uv run python scripts/demo.py
```

The script executes 6 live scenes:
1. Health and readiness probes
2. Standard chat query with local Phi-3 Mini
3. PII masking in action (`x_pii`)
4. Structured JSON response enforcement
5. Document upload, two-stage vector retrieval, and grounded RAG answer with citations (`x_sources`)
6. Empirical chunking evaluation summary table

*(To inspect the benchmark table without starting servers: `uv run python scripts/demo.py --benchmark-only`)*

---

## 6. Repository Layout

```
ericsson-genai-stack/
├── README.md                      # Executive overview, architecture, quickstart
├── AGENT_BUILD_PLAN.md            # Systematic multi-phase development plan
├── docker-compose.yml             # Container orchestration
├── .env.example                   # Environment configuration template
├── docs/
│   ├── ARCHITECTURE.md            # In-depth system architecture & sequence diagrams
│   ├── SOLUTION_GATEWAY.md        # Gateway design, API spec, router & PII benchmarks
│   ├── SOLUTION_RAG.md            # RAG design, chunking comparison, empirical results
│   ├── LEARNING_NOTES.md          # Comprehensive phase notes & mentor Q&A (Phases 0-7)
│   └── DEMO_SCRIPT.md             # 5-minute scripted presentation guide
├── gateway/
│   ├── Dockerfile                 # Hardened container specification (non-root)
│   ├── pyproject.toml             # Gateway package dependencies
│   ├── README.md                  # Gateway microservice documentation
│   ├── src/slm_gateway/           # FastAPI app, PII engine, BGE router, Phi-3 backends
│   └── tests/                     # Unit and integration test suite
├── rag/
│   ├── Dockerfile                 # Hardened container specification (non-root)
│   ├── pyproject.toml             # RAG service package dependencies
│   ├── README.md                  # RAG microservice documentation
│   ├── src/rag_service/           # Parsers, chunkers, Chroma store, re-ranker, generation
│   └── tests/                     # Unit and integration test suite
├── eval/
│   ├── docs/                      # Curated technical PDFs for benchmarking
│   ├── datasets/                  # Ground-truth evaluation datasets (JSONL)
│   ├── chunking_eval.py           # Chunking strategy & re-ranking evaluation harness
│   ├── router_eval.py             # Intent classification evaluation harness
│   ├── pii_eval.py                # PII recall & false-positive evaluation harness
│   └── results/                   # Generated evaluation markdown and CSV reports
└── scripts/
    ├── demo.py                    # Automated live demonstration script
    ├── demo.sh                    # Bash demonstration launcher
    └── verify_phase6.py           # End-to-end integration verification test
```

---

## 7. Running Verification & Test Suites

```bash
# Run all Gateway unit and integration tests (53 tests)
uv run pytest gateway/tests/ -v -m "not slow"

# Run all RAG service tests (19 tests)
uv run pytest rag/tests/ -v

# Run the complete test suite across both services
uv run pytest gateway/tests/ rag/tests/ -v -m "not slow"

# Re-run all empirical evaluation harnesses
uv run python eval/pii_eval.py
uv run python eval/router_eval.py
uv run python eval/chunking_eval.py
```

---

## 8. Documentation Index

- [Architecture & Sequence Diagrams](docs/ARCHITECTURE.md)
- [Gateway Solution Document](docs/SOLUTION_GATEWAY.md)
- [RAG Service Solution Document](docs/SOLUTION_RAG.md)
- [Learning Notes & Mentor Q&A](docs/LEARNING_NOTES.md)
- [5-Minute Demo Script](docs/DEMO_SCRIPT.md)
