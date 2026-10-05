# Known Limitations

This document catalogs the current operational boundaries and known limitations of the Local GenAI Stack.

---

## 1. Document Ingestion

| Limitation | Impact | Mitigation |
|---|---|---|
| **OCR requires Tesseract engine** | Digital text PDFs are parsed directly. Scanned pages fall back to Tesseract OCR only when Tesseract is installed on the host (or in the Docker image); otherwise scanned PDFs with no extractable text are rejected with HTTP 400. OCR quality was tested only on synthetic test fixtures and sample slide PDFs with Tesseract 5.x on Windows (mocked in CI); real-world scan accuracy is not benchmarked. | Install Tesseract OCR on the host OS (set `TESSERACT_CMD` or add to PATH) or run via the container image containing `tesseract-ocr`. |
| **No table layout preservation** | PyMuPDF extracts table text as flat strings. Complex multi-column tables lose structural relationships. | For table-heavy documents, consider table-aware parsers (e.g., Camelot, Tabula) as a preprocessing step. |
| **Single-page DOCX grouping** | Word documents without explicit page breaks return all content as page 1. | Section-based pagination can be added if finer granularity is required. |
| **No HTML, Markdown, or CSV parsing** | Only `.pdf` and `.docx` file formats are supported. Other extensions return HTTP 400. | Extend `parsers/__init__.py` with additional format handlers as needed. |

---

## 2. Concurrency and Throughput

| Limitation | Impact | Mitigation |
|---|---|---|
| **Single-GPU / CPU semaphore serialization** | The `asyncio.Semaphore(1)` around `model.generate()` serializes all inference requests. Under concurrent load, requests queue sequentially. | Switch to `BACKEND=openai_compatible` and point to a dedicated inference server (vLLM, TGI) with continuous batching. |
| **In-process model loading** | Loading Phi-3 Mini weights takes ~20-30 seconds on startup. During this window, `/ready` returns 503. | Docker Compose `depends_on: service_healthy` prevents traffic until ready. |

---

## 3. Privacy and PII Redaction

| Limitation | Impact | Mitigation |
|---|---|---|
| **English-only NER** | The spaCy `en_core_web_sm` model supports English text only. Non-English person names and addresses may not be detected. | Load additional spaCy language models and configure Presidio's `NlpEngineProvider` with multi-language support. |
| **Static codename deny-list** | The `PROJECT_CODENAME` recognizer uses a fixed regex pattern. New project codenames require updating the `PROJECT_CODENAMES` configuration. | Set `PROJECT_CODENAMES` as a comma-separated environment variable for dynamic configuration. |
| **No contextual PII detection** | Presidio relies on pattern matching and statistical NER. Context-dependent PII (e.g., "my address is 42 Oak Street") may not be caught if it does not match known entity patterns. | Supplement with custom recognizers for domain-specific PII patterns. |
| **LOCATION and DATE_TIME excluded** | Excluded to avoid false positives on normal conversational queries (e.g., questions mentioning cities or days). Physical addresses embedded in text are therefore not redacted. | Re-enable these entity types if the use case requires it, accepting higher false-positive rates. |

---

## 4. Retrieval and RAG

| Limitation | Impact | Mitigation |
|---|---|---|
| **Hybrid retrieval vs. BM25 on small set** | Hybrid retrieval (BM25 + BGE dense vectors fused via RRF) is implemented, but on our small 24-query evaluation set, it did not outperform BM25 alone on Hit@1 (66.67% vs 70.83%) or MRR (0.7083 vs 0.7292). | Benchmark on larger, domain-specific corpora to determine whether fusion yields a measurable advantage. |
| **Small evaluation corpus** | The chunking evaluation uses 3 PDFs totaling 5 pages and 36 queries; the hybrid evaluation uses 4 documents with 24 queries. Real corpora contain hundreds of pages. | Expand the evaluation corpus with larger, messier documents to validate retrieval at scale. |
| **Cross-encoder latency on CPU** | The `bge-reranker-base` cross-encoder adds ~1,000-1,750 ms per query when re-ranking candidates on CPU without GPU acceleration. | Run on GPU for sub-second re-ranking, or use distilled bi-encoders if low CPU latency is required. |
| **No chunk deduplication** | If the same document is uploaded twice, duplicate chunks are indexed. | Implement document fingerprinting (e.g., content hash) to detect and reject duplicate uploads. |

---

## 5. Model and Generation

| Limitation | Impact | Mitigation |
|---|---|---|
| **4096 token context window** | Phi-3 Mini's context window limits the amount of RAG context that can be injected. With 3 retrieved chunks, the effective context is ~1500-2000 tokens for source material. | Use larger context models (e.g., 128k context) or implement sliding-window chunking. |
| **No fine-tuning** | The model runs with base weights. Domain-specific terminology may produce less precise responses than a fine-tuned variant. | Consider LoRA/QLoRA fine-tuning on domain-specific instruction datasets. |
| **Hallucination risk** | Small language models can still produce inaccurate statements not present in the provided context. | The grounded prompt explicitly instructs the model to cite sources and refuse when information is absent. |

---

## 6. Environment and Deployment

| Limitation | Impact | Mitigation |
|---|---|---|
| **Windows ChromaDB file locking** | On Windows, rapid sequential test runs or abrupt process termination can leave SQLite WAL files locked, causing `OperationalError: database is locked`. | Use `--forked` pytest flag on Windows, or run tests in isolated temporary directories. |
| **Docker GPU passthrough unverified** | The `docker-compose.yml` GPU block is commented out by default. GPU container execution has not been verified in this development environment. | Follow NVIDIA Container Toolkit installation instructions and uncomment the `deploy.resources.reservations.devices` block. |
| **No HTTPS/TLS termination** | Both services run plain HTTP. | Deploy behind a reverse proxy (Nginx, Envoy, Traefik) with TLS termination for network security. |
| **Wide-open CORS (`allow_origins=["*"]`)** | Both the Gateway and RAG services configure `allow_origins=["*"]` for frictionless local development, testing, and UI access across ports 8000 and 8001. In production, this allows requests from any origin. | Restrict `allow_origins` to explicitly trusted frontend domains in production environment configurations. |
| **Optional Bearer authentication** | The gateway enforces Bearer token authentication only when `GATEWAY_API_KEY` is configured. When unset, requests are unauthenticated for local development. The Web Playground includes an optional API key field that attaches Bearer tokens when configured. | Set `GATEWAY_API_KEY` in environment or deploy behind an API gateway with OAuth2/OIDC. |

