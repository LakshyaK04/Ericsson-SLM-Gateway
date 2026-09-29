# Solution Document: SLM Gateway & PII/Routing Engine

This document provides a comprehensive technical overview, API specification, design rationales, and empirical evaluation results for the **SLM Gateway** (`slm_gateway`).

---

## 1. System Overview

The **SLM Gateway** acts as the secure, intelligent "front door" for enterprise generative AI workloads. It exposes an industry-standard, OpenAI-compatible chat completions interface (`/v1/chat/completions`), enforces strict client-side PII privacy policies before inference, and dynamically routes requests using semantic embeddings to the appropriate processing backend.

### Key Capabilities
- **OpenAI Compatibility**: Complete drop-in replacement for OpenAI SDKs and tools (`chatcmpl-...` response envelopes, token usage tracking).
- **In-Process Model Serving**: Serves `microsoft/Phi-3-mini-4k-instruct` in 4-bit NF4 quantization via HuggingFace Transformers and `bitsandbytes`.
- **Fail-Closed PII Masking**: Identifies and masks personal identifiers (`PERSON`, `EMAIL_ADDRESS`, `PHONE_NUMBER`, `CREDIT_CARD`, `IP_ADDRESS`) and proprietary enterprise entities (`EMPLOYEE_ID`, `PROJECT_CODENAME`) with typed placeholders.
- **Semantic Intent Routing**: Classifies queries across 4 operational intents (`general`, `technical`, `structured_json`, `rag`) in ~60ms using `BAAI/bge-small-en-v1.5` embeddings.
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
| `Authorization` | Optional | `Bearer <key>` (Enforced only when `GATEWAY_API_KEY` is set in config) |
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
*Note: `stream=true` returns HTTP 400 Bad Request unless streaming stretch feature is enabled.*

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
    "latency_ms": 61.2
  },
  "x_pii": {
    "redactions": 0
  },
  "x_sources": null
}
```

### 2.2 Model & Health Endpoints

- **`GET /v1/models`**: Returns the loaded model metadata conforming to OpenAI's model listing schema.
- **`GET /health`**: Returns `{"status": "ok"}` for container liveness probes.
- **`GET /ready`**: Returns 200 OK only when weights, tokenizers, PII pipeline, and router exemplars are fully loaded in memory.

---

## 3. PII Redaction Pipeline

### 3.1 Design & Recognizers
Implemented in `slm_gateway.pii` using Microsoft Presidio and spaCy's `en_core_web_sm` pipeline.

1. **Restricted Entity List**:
   - Standard: `PERSON`, `EMAIL_ADDRESS`, `PHONE_NUMBER`, `CREDIT_CARD`, `IP_ADDRESS`.
   - **Explicitly Excluded**: `LOCATION` and `DATE_TIME`. Presidio's default location recognizer frequently flags cities and countries ("capital of Germany" -> "capital of <LOCATION>"), mutilating knowledge queries and breaking semantic router vectors.
2. **Custom Enterprise Recognizers**:
   - `EMPLOYEE_ID`: Regex recognizer matching corporate identity badges (`EMP-\d{5,7}`) with high pattern confidence (0.85).
   - `PROJECT_CODENAME`: Exact deny-list recognizer masking confidential internal initiatives (`Project-Titan`, `Project-Apollo`, `Project-Odin`, `Project-Thor`, `Project-Aegis`).
   - `PHONE_NUMBER`: Multi-national regex capturing international dial codes (`+46-8-555-1234`, `(212) 555-0199`).

### 3.2 Security Policy: Fail-Closed
If Presidio fails to initialize (e.g., missing spaCy weights), the service refuses to start under `PII_FAIL_MODE=closed`. If an unexpected exception occurs during string sanitization, the gateway aborts the request with HTTP 500 rather than risking leaking raw unmasked PII downstream.

### 3.3 Empirical PII Evaluation Results
Evaluated against `eval/datasets/pii_eval.jsonl` (45 test cases) via `eval/pii_eval.py`:

| Metric | Measured Value | Standard |
|---|:---:|:---:|
| **Overall Entity Recall** | **100.00%** (35 / 35 entities detected) | Target: $\ge 95\%$ |
| **False Positive Rate (Clean Queries)** | **0.00%** (0 / 10 clean queries redacted) | Target: $\le 1\%$ |
| **Average Latency per Query** | **42.40 ms** | Target: $< 100\text{ms}$ |

#### Per-Entity Recall Breakdown:
- `CREDIT_CARD`: **100.0%** (5/5)
- `EMAIL_ADDRESS`: **100.0%** (5/5)
- `EMPLOYEE_ID`: **100.0%** (5/5)
- `IP_ADDRESS`: **100.0%** (5/5)
- `PERSON`: **100.0%** (5/5)
- `PHONE_NUMBER`: **100.0%** (5/5)
- `PROJECT_CODENAME`: **100.0%** (5/5)

---

## 4. Semantic Intent Router

### 4.1 Architecture & Method
Implemented in `slm_gateway.router`:
- **Model**: `BAAI/bge-small-en-v1.5` (sentence-transformers), 384-dimensional cosine space.
- **Exemplar Bank**: 112 diverse training exemplars across 4 intents (`general`, `technical`, `structured_json`, `rag`) defined in `gateway/src/slm_gateway/intents.yaml`.
- **Scoring**: Computes cosine similarity between inbound query vector and all exemplars. The intent score is the **mean of its top-3 similarities**.
- **Threshold Fallback**: If $\max(\text{scores}) < 0.55$, the query is classified as `general`.

### 4.2 Empirical Evaluation Results
Evaluated against `eval/datasets/router_eval.jsonl` (64 out-of-distribution queries, 0 exemplar leakage) via `eval/router_eval.py`:

| Intent | Support | Precision | Recall | F1-Score |
|:---|:---:|:---:|:---:|:---:|
| **`general`** | 16 | **100.0%** | 93.8% | 96.8% |
| **`technical`** | 16 | 88.2% | **93.8%** | 90.9% |
| **`structured_json`** | 16 | 88.9% | **100.0%** | 94.1% |
| **`rag`** | 16 | **100.0%** | 87.5% | 93.3% |
| **Overall** | **64** | **93.8%** | **93.8%** | **93.8% Accuracy** |

- **Mean Router Latency**: `66.83 ms` (P50: `60.84 ms`, P95: `79.38 ms`)

### 4.3 Threshold Sweep Analysis (0.30 - 0.80)
| Threshold | Accuracy | Correct / Total | Fallbacks to `general` | Rationale |
|:---:|:---:|:---:|:---:|---|
| `0.30 - 0.50` | 92.2% | 59/64 | 0 - 1 | Too aggressive; routes ambiguous queries into specialized paths. |
| **`0.55` (Optimal)** | **93.8%** | **60/64** | **7** | **Optimal balance of discriminatory accuracy and noise protection.** |
| `0.60` | 92.2% | 59/64 | 12 | Mild under-triggering of RAG intents. |
| `0.65` | 89.1% | 57/64 | 21 | High false fallback rate on technical queries. |
| `0.70 - 0.80` | 64.1% - 26.6% | 41/64 - 17/64 | 38 - 63 | Overly conservative; collapses into general fallback. |

---

## 5. Model Serving Backends

### 5.1 `hf_local` (Default Production Backend)
- **Model**: `microsoft/Phi-3-mini-4k-instruct`.
- **Quantization**: 4-bit NormalFloat (NF4) via `bitsandbytes` with double quantization. Reduces GPU VRAM footprint from 7.6GB (FP16) to ~2.6GB.
- **Concurrency & Concurrency Isolation**: In-process PyTorch generation is CPU/GPU-blocking. Wrapped in `asyncio.to_thread` guarded by an `asyncio.Semaphore(1)` to ensure single-GPU serial execution without thread collisions or CUDA memory corruption.
- **Context Management**: Context window constrained to 4096 tokens. The oldest user/assistant turns are truncated gracefully while preserving system instructions.

### 5.2 `openai_compatible` (Flexible Proxy Backend)
- Directs queries to any external OpenAI-compatible inference engine (e.g., vLLM, Ollama, TGI, or mock test servers) specified via `BACKEND_URL`.
- Enables CPU-only development, unit testing without GPU requirements, and seamless migration to dedicated enterprise model clusters.

---

## 6. Known Limitations
1. **Single-GPU Concurrency**: Because generation runs behind a semaphore of size 1, high concurrent request volume will queue up. For horizontal scaling, deploy multiple Gateway replicas behind a round-robin load balancer.
2. **Streaming Not Supported**: The baseline implementation returns non-streaming completions. `stream=true` returns HTTP 400.
3. **English Focus**: The bundled Presidio recognizers and spaCy language pipeline are tuned for English text (`en_core_web_sm`).
