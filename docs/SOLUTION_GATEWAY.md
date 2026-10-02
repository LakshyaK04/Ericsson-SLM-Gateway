# Solution Document: SLM Gateway & PII/Routing Engine

This document provides a concise technical overview, API specification, design rationales, and empirical evaluation results for the **SLM Gateway** (`slm_gateway`).

---

## 1. System Overview

The **SLM Gateway** acts as the secure reverse proxy and ingestion gateway for local generative AI workloads. It exposes an industry-standard, OpenAI-compatible chat completions interface (`/v1/chat/completions`), enforces client-side PII privacy policies before inference, and dynamically routes requests using semantic embeddings to the appropriate backend.

### Key Capabilities
- **OpenAI Compatibility**: Drop-in replacement for OpenAI SDKs and tools (`chatcmpl-...` response envelopes, token usage tracking, and SSE streaming).
- **In-Process Model Serving**: Serves `microsoft/Phi-3-mini-4k-instruct` in 4-bit NF4 quantization via HuggingFace Transformers and `bitsandbytes`.
- **Fail-Closed PII Masking**: Identifies and masks personal identifiers (`PERSON`, `EMAIL_ADDRESS`, `PHONE_NUMBER`, `CREDIT_CARD`, `IP_ADDRESS`), custom enterprise IDs (`EMPLOYEE_ID`), and project codenames with typed placeholders.
- **Semantic Intent Routing**: Classifies queries across 3 operational intents (`general`, `technical`, `rag`) in ~13.3ms using `BAAI/bge-small-en-v1.5` embeddings.
- **RAG Delegation**: Orchestrates grounded retrieval-augmented queries to the RAG microservice with automatic loop prevention (`X-Bypass-Router`) and fallback handling.

---

## 2. API Specification

The gateway listens by default on `http://localhost:8000`.

### 2.1 `POST /v1/chat/completions`

Creates a model completion for the provided chat messages.

#### Headers
| Header | Required | Description |
|---|:---:|---|
| `Content-Type` | Yes | `application/json` |
| `X-Bypass-Router` | Optional | `true` or `false`. If `true`, skips intent routing and dispatches directly to local LLM backend. Used by RAG service to prevent loops. |

#### Request Body Schema
```json
{
  "model": "microsoft/Phi-3-mini-4k-instruct",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain 5G network slicing."}
  ],
  "temperature": 0.7,
  "top_p": 1.0,
  "max_tokens": 512,
  "stream": false
}
```
*Note: `stream=true` returns HTTP 400 Bad Request.*

#### Response Body Schema (OpenAI-Compatible + Namespaced Extensions)
```json
{
  "id": "chatcmpl-7d5a86d5beee4e6f",
  "object": "chat.completion",
  "created": 1775010000,
  "model": "microsoft/Phi-3-mini-4k-instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "5G network slicing allows operators to divide a physical network into multiple virtual networks..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 28,
    "completion_tokens": 85,
    "total_tokens": 113
  },
  "x_routing": {
    "intent": "technical",
    "confidence": 0.7234,
    "route": "hf_local",
    "latency_ms": 11.6
  },
  "x_pii": {
    "redactions": 0
  },
  "x_sources": null
}
```

### 2.2 Model & Health Endpoints
- **`GET /v1/models`**: Returns loaded model metadata conforming to OpenAI's schema.
- **`GET /health`**: Returns `{"status": "ok"}` for container liveness probes.
- **`GET /ready`**: Returns 200 OK only when weights, tokenizers, PII pipeline, and router exemplars are loaded in memory.

---

## 3. PII Redaction Pipeline

### 3.1 Design & Recognizers
Implemented in `slm_gateway.pii` using Microsoft Presidio and spaCy's `en_core_web_sm` pipeline.
1. **Restricted Entity List**:
   - Standard: `PERSON`, `EMAIL_ADDRESS`, `PHONE_NUMBER`, `CREDIT_CARD`, `IP_ADDRESS`.
   - **Explicitly Excluded**: `LOCATION` and `DATE_TIME`. Presidio's default location recognizer frequently flags common cities and countries, mutilating knowledge queries.
2. **Custom Enterprise Recognizer**:
   - `EMPLOYEE_ID`: Regex recognizer matching corporate identity badges (`EMP-\d{5,7}`) with high pattern confidence (0.85).

### 3.2 Security Policy: Fail-Closed
Under `PII_FAIL_MODE=closed` (default), if an unexpected exception occurs during string sanitization, the gateway aborts the request with HTTP 500 rather than leaking raw unmasked PII downstream.

---

## 4. Semantic Intent Router

### 4.1 Architecture & Method
Implemented in `slm_gateway.router`:
- **Model**: `BAAI/bge-small-en-v1.5` (sentence-transformers), 384-dimensional cosine space.
- **Exemplars**: Curated training exemplars across 3 intents (`general`, `technical`, `rag`) defined in `gateway/src/slm_gateway/intents.yaml`.
- **Scoring**: Computes cosine similarity between inbound query vector and all exemplars. The intent score is the **mean of its top-3 similarities**.
- **Threshold Fallback**: If $\max(\text{scores}) < 0.55$, the query is classified as `general`.

### 4.2 Empirical Evaluation Results
Evaluated against `eval/datasets/router_eval.jsonl` (48 queries, 16 per intent, 0 exemplar leakage) via `eval/router_eval.py`:

| Intent | Support | Precision | Recall | F1-Score |
|:---|:---:|:---:|:---:|:---:|
| **`general`** | 16 | **100.0%** | 93.8% | 96.8% |
| **`technical`** | 16 | 88.2% | **93.8%** | 90.9% |
| **`rag`** | 16 | 93.8% | 93.8% | 93.8% |
| **Overall** | **48** | **93.75% Accuracy (45/48)** | — | — |

- **Mean Router Latency**: `11.57 ms` (P50: `9.20 ms`, P95: `29.03 ms`)

---

## 5. Model Serving Backends

### 5.1 `hf_local` (Default In-Process Backend)
- **Model**: `microsoft/Phi-3-mini-4k-instruct`.
- **Quantization**: 4-bit NormalFloat (NF4) via `bitsandbytes` with double quantization. Reduces GPU VRAM footprint from 7.6GB (FP16) to ~2.6GB.
- **Concurrency Isolation**: PyTorch generation runs behind an `asyncio.Semaphore(1)` to ensure single-GPU serial execution without thread collisions or CUDA memory corruption.
- **Context Management**: Context window constrained to 4096 tokens. Oldest conversational turns are truncated gracefully while preserving system instructions.

### 5.2 `openai_compatible` (Flexible Proxy Backend)
- Directs queries to any external OpenAI-compatible inference engine (e.g., vLLM, Ollama, TGI, or mock servers) specified via `BACKEND_URL`.
- Enables CPU-only development, unit testing without GPU requirements, and seamless migration to dedicated enterprise model clusters.

---

## 6. Limitations

1. **Single-GPU Semaphore Concurrency**: Because in-process generation runs behind a semaphore of size 1, concurrent requests queue sequentially. For heavy concurrency, deploy multiple replicas or switch `BACKEND=openai_compatible` to route to an external engine like vLLM.
2. **Batch vs Streaming**: Both SSE streaming (`stream=true`) and standard buffered completions (`stream=false`) are supported.
3. **English-Language NER**: The bundled Presidio recognizers and spaCy model (`en_core_web_sm`) are trained on English syntax; non-English prompts may have lower entity detection recall.
4. **Out-of-Distribution Routing**: Queries far removed from exemplar topics that score below the 0.55 similarity threshold fall back cleanly to `general`.
5. **GPU Container Passthrough**: Running GPU inference inside Docker requires the host to have the NVIDIA Container Toolkit installed; otherwise CPU fallback or `openai_compatible` backend must be used.
