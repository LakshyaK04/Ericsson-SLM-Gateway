# 5-Minute Technical Demonstration Script

This script guides the presenter through an end-to-end technical demonstration of the **Ericsson Local GenAI Stack**. Every command is runnable either via the automated CLI script or via standard `curl` commands.

---

## Preparation & Prerequisites

1. Ensure the services are running:
   - **Terminal 1 (RAG Service):**
     ```bash
     uv run uvicorn rag_service.main:app --host 0.0.0.0 --port 8001
     ```
   - **Terminal 2 (SLM Gateway):**
     ```bash
     uv run uvicorn slm_gateway.main:app --host 0.0.0.0 --port 8000
     ```
2. Alternatively, run the automated Python demonstration:
   ```bash
   uv run python scripts/demo.py
   ```
   *(To display the empirical benchmark table without active servers, run `uv run python scripts/demo.py --benchmark-only`)*

---

## Scene 1: Service Liveness & Health (30 Seconds)

### Goal
Demonstrate that both microservices are decoupled, operational, and exposing health probes.

### Command
```bash
curl -s http://localhost:8000/health
curl -s http://localhost:8001/health
```

### Expected Output
```json
{"status": "ok"}
{"status": "ok"}
```

### Mentor Talking Points
- *"The Gateway on port 8000 is the public front door. The RAG service on port 8001 is an internal subsystem. Both services initialize their respective models and vector stores at startup during FastAPI lifespan."*

---

## Scene 2: Standard OpenAI Chat Completion (1 Minute)

### Goal
Demonstrate drop-in OpenAI API compatibility, token counting, and intent routing to local Phi-3 Mini.

### Command
```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is the capital of Sweden?"}],
    "temperature": 0.7,
    "max_tokens": 64
  }'
```

### Expected Output
```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1775010000,
  "model": "microsoft/Phi-3-mini-4k-instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The capital of Sweden is Stockholm."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 10,
    "total_tokens": 25
  },
  "x_routing": {
    "intent": "general",
    "confidence": 0.5484,
    "route": "hf_local",
    "latency_ms": 7.85
  },
  "x_pii": {
    "redactions": 0
  },
  "x_sources": null
}
```

### Mentor Talking Points
- *"Notice standard OpenAI envelope fields (`id`, `choices`, `usage`). Any OpenAI Python or LangChain client works simply by setting `base_url='http://localhost:8000/v1'`."*
- *"Notice `x_routing`: BGE embeddings classified this query as `general` with ~8ms latency, dispatching it to local in-process Phi-3."*

---

## Scene 3: Enterprise PII Masking & Privacy (1 Minute)

### Goal
Demonstrate client-side privacy protection. Sensitive entities are redacted before LLM generation; clean text is untouched.

### Command
```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Hello, I am Alice Smith (EMP-84920). My contact is alice.smith@ericsson.com and +46-8-555-1234. I work on Project-Titan in Stockholm."
      }
    ],
    "max_tokens": 64
  }'
```

### Expected Output
```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "Hello <PERSON>, I have received your request regarding <PROJECT_CODENAME>..."
      }
    }
  ],
  "x_pii": {
    "redactions": 5
  }
}
```

### Mentor Talking Points
- *"Presidio redacted `PERSON` (Alice Smith), `EMPLOYEE_ID` (EMP-84920), `EMAIL_ADDRESS`, `PHONE_NUMBER`, and `PROJECT_CODENAME` (Project-Titan)."*
- *"Crucially, 'Stockholm' was NOT redacted because we deliberately excluded `LOCATION` and `DATE_TIME`. In testing, Presidio's default location recognizer mutilated geography queries like 'What is the capital of Germany?'. Our custom policy achieved 100% recall with 0% false positives on clean queries."*

---

## Scene 4: Structured JSON Intent Enforcement (45 Seconds)

### Goal
Show dynamic system prompt injection when the query requests JSON output.

### Command
```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Return a JSON list of three 5G core network functions with acronym and name."
      }
    ],
    "temperature": 0.2,
    "max_tokens": 150
  }'
```

### Expected Output
```json
{
  "choices": [
    {
      "message": {
        "content": "[\n  {\"acronym\": \"AMF\", \"name\": \"Access and Mobility Management Function\"},\n  {\"acronym\": \"SMF\", \"name\": \"Session Management Function\"},\n  {\"acronym\": \"UPF\", \"name\": \"User Plane Function\"}\n]"
      }
    }
  ],
  "x_routing": {
    "intent": "structured_json",
    "confidence": 0.6421,
    "route": "hf_local"
  }
}
```

### Mentor Talking Points
- *"The semantic router recognized the request for structured formatting and automatically injected a strict JSON system prompt without requiring the client to explicitly engineer prompts."*

---

## Scene 5: RAG Document Upload & Grounded Question Answering (1.5 Minutes)

### Goal
Demonstrate PDF ingestion, two-stage vector retrieval, loop prevention (`X-Bypass-Router`), and grounded citations (`x_sources`).

### Step 5a: Upload PDF to RAG Service
```bash
curl -s -X POST http://localhost:8001/documents \
  -F "file=@eval/docs/ericsson_rag_sample.pdf"
```
**Output:**
```json
{
  "doc_id": "...",
  "filename": "ericsson_rag_sample.pdf",
  "total_chunks": 16,
  "chunks_per_strategy": {"character": 6, "structure": 4, "semantic": 6}
}
```

### Step 5b: Ask Grounded Question Through Gateway (Port 8000 Only)
```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "According to the uploaded documentation, what capabilities does the Ericsson AI Platform provide?"
      }
    ],
    "max_tokens": 128
  }'
```

### Expected Output
```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "Based on the provided documentation [1], the Ericsson AI Platform provides services for deploying and operating artificial intelligence applications, supporting OpenAI-compatible APIs and automated ingestion pipelines."
      }
    }
  ],
  "x_routing": {
    "intent": "rag",
    "confidence": 0.6807,
    "route": "rag_service"
  },
  "x_sources": [
    {
      "chunk_id": "ericsson_rag_sample.pdf_struct_p1_1",
      "source": "ericsson_rag_sample.pdf",
      "page": 1,
      "dense_score": 0.5108,
      "rerank_score": 0.8902
    }
  ]
}
```

### Mentor Talking Points
- *"The client sent the query to port 8000. The Gateway classified it as `rag` and delegated to RAG port 8001. RAG retrieved top-20 candidates from ChromaDB, re-ranked them with `bge-reranker-base` down to 3, formulated a grounded prompt with bracketed `[1]` context, and called back to the Gateway `/v1/chat/completions`."*
- *"Loop Prevention: RAG passed `X-Bypass-Router: true`. The Gateway bypassed the router and went straight to Phi-3, avoiding infinite recursion."*
- *"Notice `x_sources`: The response carries full citation provenance with document name, page, dense similarity score, and cross-encoder re-ranking score."*

---

## Scene 6: Empirical Chunking Strategy Benchmark (30 Seconds)

### Goal
Present real benchmark findings proving why `structure` chunking with re-ranking was chosen as the default.

### Benchmark Summary Table
```
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
```

### Mentor Talking Points
- *"Structure chunking + cross-encoder achieved 100% Hit@1 and MRR 1.0000 across all 36 evaluation questions."*
- *"The cross-encoder added ~145ms latency (from ~10ms to ~159ms), but provided a +13.89% boost in retrieval precision for structure chunking and +13.88% for semantic chunking."*
- *"On character chunking, the re-ranker showed no improvement because severed clauses lack sufficient syntactic context for cross-attention."*

---

## Untested Environment Disclosure

> [!NOTE]
> **Docker GPU Passthrough (`nvidia-container-toolkit`):**
> Docker image specifications, multi-stage Dockerfiles, and `docker-compose.yml` have been crafted to industry standards. However, in this container development environment, the Docker daemon does not have direct access to physical host GPU passthrough.
> To run the containerized stack with physical GPU acceleration on host Linux / WSL2 machines:
> 1. Install `nvidia-container-toolkit`: `sudo apt install -y nvidia-container-toolkit && sudo systemctl restart docker`
> 2. Uncomment the `deploy.resources.reservations.devices` block in `docker-compose.yml`.
> 3. Run: `docker compose up --build`.
