# Ericsson Local GenAI Stack: SLM Gateway & Hybrid RAG

A privacy-focused, fully local Generative AI stack combining an **OpenAI-Compatible SLM Gateway** with a **Hybrid RAG Service**. Built to run locally with open models (`microsoft/Phi-3-mini-4k-instruct`, `BAAI/bge-small-en-v1.5`, `BAAI/bge-reranker-base`), ensuring zero cloud data egress and full data privacy.

---

## 1. System Architecture

```mermaid
graph TD
    Client[Client / OpenAI SDK] -->|POST /v1/chat/completions| Gateway[SLM Gateway :8000]

    subgraph Gateway Subsystems
        PII[Presidio PII Redactor<br/>PERSON, EMAIL, PHONE, CC, IP, EMP-ID]
        Router[BGE Semantic Intent Router<br/>general | technical | rag]
        LocalModel[Phi-3-Mini 4k Model Backend<br/>In-Process 4-bit NF4 / Semaphore]
    end

    Gateway --> PII --> Router
    Router -->|general / technical| LocalModel
    Router -->|rag intent| RAGClient[Gateway RAG Client]

    subgraph RAG Service :8001
        Ingest[Document Ingestion<br/>PDF & DOCX Parsers]
        Chunkers[Chunking Strategies<br/>Character | Structure | Semantic]
        VectorStore[(ChromaDB Collections)]
        DenseSearch[BGE Dense Retrieval k=20]
        Reranker[BGE Cross-Encoder k=3]
        ContextBuilder[Grounded Prompt Builder<br/>[1], [2] Citations]
    end

    RAGClient -->|POST /answer| ContextBuilder
    ContextBuilder --> DenseSearch --> VectorStore
    DenseSearch --> Reranker --> ContextBuilder
    ContextBuilder -->|POST /v1/chat/completions<br/>X-Bypass-Router: true| LocalModel
    LocalModel --> Gateway
    Gateway -->|OpenAI-Compatible Response<br/>+ x_routing, x_pii, x_sources| Client
```

---

## 2. Quickstart

### Prerequisites
- Python 3.10
- `uv` package manager (or standard virtual environment)
- NVIDIA GPU recommended for in-process 4-bit model serving, or CPU/remote fallback via `BACKEND=openai_compatible`

### Run Services
```bash
# Terminal 1: Start RAG Service (Port 8001)
uv run uvicorn rag_service.main:app --host 0.0.0.0 --port 8001

# Terminal 2: Start SLM Gateway (Port 8000)
uv run uvicorn slm_gateway.main:app --host 0.0.0.0 --port 8000
```

### Run the Demo
```bash
# Live 4-scene demo: health check, normal chat, PII masking, RAG query with sources
uv run python scripts/demo.py

# Or inspect the empirical benchmark table directly:
uv run python scripts/demo.py --benchmark-only
```

### Run the Tests
```bash
# Run all unit and integration tests (excluding slow GPU inference tests):
uv run pytest gateway/tests rag/tests -m "not slow"
```

---

## 3. Key Evaluation Results

### 3.1 Chunking Strategy & Re-Ranking
Evaluated across 36 ground-truth questions on a small synthetic corpus (3 PDFs / 5 pages from `scripts/create_eval_docs.py`):

| Strategy | Re-ranker | Total Chunks | Avg Length | Hit@1 (%) | Hit@3 (%) | MRR | Latency (ms) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `character` | Off | 16 | 422.1 ch | 86.11% | 97.22% | 0.9028 | 11.6 |
| `character` | On | 16 | 422.1 ch | 83.33% | 97.22% | 0.9028 | 152.2 |
| `structure` | Off | 10 | 621.0 ch | 86.11% | 94.44% | 0.9028 | 9.8 |
| **`structure`** | **On** | **10** | **621.0 ch** | **100.00%** | **100.00%** | **1.0000** | **159.3** |
| `semantic` | Off | 16 | 387.2 ch | 80.56% | 94.44% | 0.8611 | 10.1 |
| **`semantic`** | **On** | **16** | **387.2 ch** | **94.44%** | **97.22%** | **0.9583** | **166.9** |

*Note: Cross-encoder re-ranking improved Hit@1 for structure (+13.9%) and semantic (+13.9%), but did not improve character chunking on Hit@1 (86.1% vs 83.3%). Re-ranking adds ~140-155ms cross-encoder inference latency.*

### 3.2 Semantic Intent Router Accuracy
Evaluated on 48 out-of-distribution queries with 0 training exemplar leakage (`eval/datasets/router_eval.jsonl`):

| Intent | Support | Precision | Recall | F1-Score |
|---|:---:|:---:|:---:|:---:|
| `general` | 16 | 100.0% | 93.8% | 96.8% |
| `technical` | 16 | 88.2% | 93.8% | 90.9% |
| `rag` | 16 | 93.8% | 93.8% | 93.8% |
| **Overall** | **48** | **93.75% Accuracy** | — | — |

*Operating threshold: `0.55` (empirically tuned via threshold sweep). Mean classification latency: ~11.6ms.*

---

## 4. Documentation
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): Merged end-to-end architecture and request flow.
- [docs/SOLUTION_GATEWAY.md](docs/SOLUTION_GATEWAY.md): Gateway design decisions, API spec, and limitations.
- [docs/SOLUTION_RAG.md](docs/SOLUTION_RAG.md): RAG design decisions, chunking strategies, and limitations.
- [docs/LEARNING_NOTES.md](docs/LEARNING_NOTES.md): Engineering rationales and mentor interview prep.
