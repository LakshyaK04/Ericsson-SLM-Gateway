# SLM Gateway

The **SLM Gateway** is a lightweight API gateway and inference router that wraps small language models (`microsoft/Phi-3-mini-4k-instruct`) in an OpenAI-compatible interface, enforces client-side PII privacy policies, and dynamically routes requests based on semantic intent embeddings.

---

## Features

- **OpenAI-Compatible `/v1/chat/completions`**: Seamlessly integrates with the official `openai` Python SDK or any OpenAI-compatible client.
- **In-Process Model Serving (`hf_local`)**: Loads `microsoft/Phi-3-mini-4k-instruct` in 4-bit NF4 quantization via `bitsandbytes`, guarded by an async semaphore for single-GPU stability.
- **Client-Side PII Masking**: Microsoft Presidio analyzer with custom `EMPLOYEE_ID` recognizer and fail-closed privacy policy. Geographic names (`LOCATION`) and dates (`DATE_TIME`) are preserved to prevent query corruption.
- **Semantic Intent Router**: Classifies queries across 3 intents (`general`, `technical`, `rag`) in ~12ms using `BAAI/bge-small-en-v1.5` embeddings.
- **Loop-Safe RAG Delegation**: Routes document questions to the RAG service and accepts generation callbacks safely using `X-Bypass-Router: true`.

---

## Architecture Diagram

```mermaid
graph TD
    Client[Client / SDK] -->|POST /v1/chat/completions| GW[FastAPI Gateway :8000]
    GW --> PII[Presidio PII Redactor]
    PII --> Router[BGE Semantic Router]
    
    Router -->|general / technical| LocalModel[In-Process Phi-3 Mini]
    Router -->|rag intent| RAGClient[RAG Client HTTP] --> RAGService[RAG Service :8001]
    
    LocalModel --> ResponseBuilder[OpenAI Response Builder]
    ResponseBuilder --> Client
```

---

## Quickstart

### 1. Local Environment (uv)

```bash
# Navigate to repo root and start gateway
uv run uvicorn slm_gateway.main:app --host 0.0.0.0 --port 8000
```

Verify service readiness:
```bash
curl http://localhost:8000/ready
```

### 2. Docker Container

Build and run the standalone gateway container:
```bash
docker build -t slm-gateway -f gateway/Dockerfile ./gateway
docker run -p 8000:8000 slm-gateway
```

---

## Configuration Reference

Key settings can be configured via environment variables or a `.env` file:

| Variable | Type | Default | Description |
|---|:---:|:---:|---|
| `HOST` | string | `0.0.0.0` | Bind host address |
| `PORT` | int | `8000` | Bind port |
| `BACKEND` | string | `hf_local` | LLM backend: `hf_local` (in-process) |
| `MODEL_ID` | string | `microsoft/Phi-3-mini-4k-instruct` | HuggingFace model identifier |
| `QUANTIZE` | string | `4bit` | Quantization: `4bit` (NF4 via bitsandbytes) or `none` |
| `DEVICE` | string | `auto` | PyTorch device mapping (`auto`, `cuda`, `cpu`) |
| `MAX_CONTEXT_LENGTH` | int | `4096` | Maximum token context window |
| `DEFAULT_MAX_TOKENS` | int | `512` | Default completion tokens |
| `DEFAULT_TEMPERATURE` | float | `0.7` | Sampling temperature |
| `DEFAULT_TOP_P` | float | `1.0` | Nucleus sampling probability |
| `PII_FAIL_MODE` | string | `closed` | `closed` (abort on error) or `open` (warn and bypass) |
| `ROUTER_MODEL_NAME` | string | `BAAI/bge-small-en-v1.5` | Embedding model for semantic router |
| `ROUTER_THRESHOLD` | float | `0.55` | Cosine similarity threshold for intent fallback |
| `RAG_SERVICE_URL` | string | `http://localhost:8001` | RAG service base URL |
| `RAG_TIMEOUT_SECONDS` | float | `30.0` | HTTP timeout when delegating to RAG service |

---

## API Specification & Examples

### Chat Completion via `curl`

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is 5G SBA?"}
    ],
    "temperature": 0.7,
    "max_tokens": 128
  }'
```

### Python OpenAI SDK Example

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="microsoft/Phi-3-mini-4k-instruct",
    messages=[
        {"role": "user", "content": "Explain 5G network slicing."}
    ],
    temperature=0.7,
    max_tokens=128,
)

print(response.choices[0].message.content)
```
