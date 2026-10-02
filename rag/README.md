# RAG Service: Ingestion, Retrieval & Re-Ranking

The **RAG Service** is a decoupled microservice providing document ingestion, multi-strategy text chunking, dense vector retrieval, cross-encoder neural re-ranking, and grounded generation for technical documentation.

---

## Features

- **Document Parsing**: High-fidelity text extraction from PDF (PyMuPDF) and DOCX (`python-docx`) preserving page markers, headings, and paragraph boundaries. Rejects empty/scanned PDFs with clear 400 errors.
- **Selectable Chunking**: Supports 3 distinct segmentation strategies:
  - `character`: Fixed-size sliding window (500 chars, 50 overlap) with backward whitespace snapping.
  - `structure`: Heading-aware hierarchy merging paragraphs up to 1000 chars. **Achieved 100% Hit@1 in empirical evaluation.**
  - `semantic`: Dynamic boundary detection based on sentence embedding cosine similarity drops.
- **Persistent Vector Store**: ChromaDB client maintaining 3 isolated collections (`chunks_character`, `chunks_structure`, `chunks_semantic`) to prevent cross-strategy index skew.
- **Two-Stage Retrieval**: Dense vector search (top-20) via `BAAI/bge-small-en-v1.5`, followed by deep cross-encoder re-ranking (top-3) via `BAAI/bge-reranker-base`.
- **Grounded Answer Synthesis**: Assembles structured prompt with numbered context brackets (`[1]`, `[2]`) and calls the Gateway's `/v1/chat/completions` endpoint passing `X-Bypass-Router: true`.

---

## Architecture Diagram

```mermaid
graph TD
    Upload[Upload PDF / DOCX] --> Parsers[PyMuPDF / python-docx]
    Parsers --> Chunkers[Chunking Strategies: character | structure | semantic]
    Chunkers --> Embedder[BGE-small Embedding Model]
    Embedder --> Chroma[(ChromaDB Collections)]
    
    Query[POST /query or /answer] --> Retriever[Dense Vector Search: Top 20]
    Chroma --> Retriever
    Retriever --> Reranker[BGE-Reranker Cross-Encoder: Top 3]
    Reranker --> Formatter[Context Formatter: [1], [2], [3]]
    Formatter --> GatewayClient[HTTP Client: X-Bypass-Router: true]
    GatewayClient --> Gateway[SLM Gateway :8000]
    Gateway --> AnswerResponse[Answer + Ranked Sources + Token Usage]
```

---

## Quickstart

### 1. Local Environment (uv)

```bash
# Navigate to repo root and start RAG service
uv run uvicorn rag_service.main:app --host 0.0.0.0 --port 8001
```

Verify service liveness:
```bash
curl http://localhost:8001/health
```

### 2. Docker Container

Build and run the standalone RAG service:
```bash
docker build -t rag-service -f rag/Dockerfile ./rag
docker run -p 8001:8001 -v rag-data:/app/data/chroma rag-service
```

---

## Configuration Reference

Configure via environment variables or `.env`:

| Variable | Type | Default | Description |
|---|:---:|:---:|---|
| `HOST` | string | `0.0.0.0` | Bind host address |
| `PORT` | int | `8001` | Bind port |
| `CHROMA_PERSIST_DIR` | string | `data/chroma` | Persistent ChromaDB storage path |
| `EMBEDDING_MODEL_NAME` | string | `BAAI/bge-small-en-v1.5` | HuggingFace embedding model ID |
| `RERANKER_MODEL_NAME` | string | `BAAI/bge-reranker-base` | Cross-encoder re-ranking model ID |
| `DEFAULT_STRATEGY` | string | `structure` | Default chunking strategy |
| `RETRIEVE_K` | int | `20` | Candidate chunk count for dense stage |
| `FINAL_K` | int | `3` | Final chunk count after re-ranking |
| `GATEWAY_URL` | string | `http://localhost:8000` | Gateway URL for text generation |
| `GATEWAY_TIMEOUT` | float | `30.0` | HTTP timeout when calling Gateway |

---

## API Specification & Examples

### 1. Ingest Document (`POST /documents`)

```bash
curl -X POST http://localhost:8001/documents \
  -F "file=@eval/docs/5g_core_architecture.pdf" \
  -F "strategies=character,structure,semantic"
```

### 2. List Indexed Documents (`GET /documents`)

```bash
curl http://localhost:8001/documents
```

### 3. Retrieve Re-Ranked Chunks (`POST /query`)

```bash
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Which network function manages user registration and authentication?",
    "strategy": "structure",
    "retrieve_k": 20,
    "final_k": 3
  }'
```

### 4. Grounded Question Answering (`POST /answer`)

```bash
curl -X POST http://localhost:8001/answer \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the primary duties of the AMF and SMF?",
    "strategy": "structure"
  }'
```

---

## Testing

```bash
# Run all RAG service unit and integration tests
uv run pytest rag/tests/ -v
```

---

## Troubleshooting

- **ChromaDB Lock Error on Windows**: Ensure multiple test runners are not writing concurrently to the same local directory.
- **Empty PDF Error**: If an uploaded PDF contains scanned raster images without a digital text layer, the parser returns HTTP 400. OCR is not supported.
- **Gateway Connection Refused on `/answer`**: Verify the SLM Gateway is active on `http://localhost:8000`.
