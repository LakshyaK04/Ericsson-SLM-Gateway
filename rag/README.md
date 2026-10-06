# RAG Service: Ingestion, Retrieval & Re-Ranking

The **RAG Service** is a decoupled microservice providing document ingestion, multi-strategy text chunking, dense vector retrieval, cross-encoder neural re-ranking, and grounded generation for technical documentation.

---

## Features

- **Document Parsing**: High-fidelity text extraction from PDF (PyMuPDF) and DOCX (`python-docx`) preserving page markers, headings, and paragraph boundaries. Digital text PDFs are parsed directly. Scanned pages fall back to Tesseract OCR only when Tesseract is installed on the host (or in the Docker image); otherwise scanned PDFs are rejected with HTTP 400.
- **Selectable Chunking**: Supports 3 distinct segmentation strategies:
  - `character`: Fixed-size sliding window (500 chars, 50 overlap) with backward whitespace snapping.
  - `structure`: Heading-aware hierarchy merging paragraphs up to 1000 chars. Achieved 94.44% Hit@1 (100% with re-ranker) in empirical evaluation on 36 questions.
  - `semantic`: Dynamic boundary detection based on sentence embedding cosine similarity drops.
- **Persistent Vector Store**: ChromaDB client maintaining 3 isolated collections (`chunks_character`, `chunks_structure`, `chunks_semantic`) to prevent cross-strategy index skew.
- **Hybrid Retrieval & Optional Re-Ranking**: Hybrid retrieval combining BM25 sparse keyword search and BGE dense vector search via Reciprocal Rank Fusion (RRF, $k=60$), with optional cross-encoder re-ranking via `BAAI/bge-reranker-base`.
- **Grounded Answer Synthesis**: Assembles structured prompt with numbered context brackets (`[1]`, `[2]`) and calls the Gateway's `/v1/chat/completions` endpoint passing `X-Bypass-Router: true`.

---

## Architecture Diagram

```mermaid
graph TD
    Upload[Upload PDF / DOCX] --> Parsers[PyMuPDF / python-docx]
    Parsers --> Chunkers[Chunking Strategies: character | structure | semantic]
    Chunkers --> Embedder[BGE-small Embedding Model]
    Embedder --> Chroma[(ChromaDB Collections)]
    
    Query[POST /query or /answer] --> Retriever[Hybrid Retrieval: BM25 + Dense RRF]
    Chroma --> Retriever
    Retriever --> Reranker[Optional Cross-Encoder Re-Ranking]
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
uv run --project rag uvicorn rag_service.main:app --host 0.0.0.0 --port 8001
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
| `GATEWAY_TIMEOUT` | float | `180.0` | HTTP timeout when calling Gateway |
| `TESSERACT_CMD` | string | `None` | Optional path to Tesseract OCR binary |

---

### Optical Character Recognition (OCR) Setup
Digital text PDFs are parsed directly. Scanned pages fall back to Tesseract OCR only when Tesseract is installed on the host (or in the Docker image); otherwise scanned PDFs are rejected with HTTP 400.
- **Windows**: Install [Tesseract OCR for Windows](https://github.com/UB-Mannheim/tesseract/wiki). Add it to your system PATH or configure `TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe` in `.env`.
- **Linux (Debian/Ubuntu)**: Install the system package:
  ```bash
  sudo apt-get update && sudo apt-get install -y tesseract-ocr
  ```
- **Docker**: The provided `rag/Dockerfile` includes `tesseract-ocr` via `apt-get` (untested in this local environment).

*Note on testing*: OCR quality was tested only on synthetic test fixtures and sample slide PDFs with Tesseract 5.x on Windows (and mocked in CI); real-world scan accuracy is not benchmarked.

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
- **Empty / Scanned PDF Error**: If an uploaded PDF contains scanned raster images without a digital text layer, the parser falls back to Tesseract OCR if installed. If Tesseract is not installed or finds no text, the parser returns HTTP 400: *"No extractable text found in PDF. The document appears empty or scanned, and OCR is unavailable or found no text (install Tesseract to enable OCR for scanned pages)."*
- **Gateway Connection Refused on `/answer`**: Verify the SLM Gateway is active on `http://localhost:8000`.

