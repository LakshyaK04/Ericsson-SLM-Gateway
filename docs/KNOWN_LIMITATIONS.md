# Known Limitations

This document catalogs the current operational boundaries and known limitations of the Local GenAI Stack. It is intended for engineers evaluating production readiness.

---

## 1. Document Ingestion

| Limitation | Impact | Mitigation |
|---|---|---|
| **No OCR support** | Scanned image-only PDFs return HTTP 400 immediately with a clear error message. | Use text-based PDFs or pre-process scans with an external OCR tool (e.g., Tesseract, Azure Form Recognizer). |
| **No table layout preservation** | PyMuPDF extracts table text as flat strings. Complex multi-column tables lose structural relationships. | For table-heavy documents, consider table-aware parsers (e.g., Camelot, Tabula) as a preprocessing step. |
| **Single-page DOCX grouping** | Word documents without explicit page breaks return all content as page 1. | This is acceptable for most use cases. Consider section-based pagination if needed. |
| **No HTML, Markdown, or CSV parsing** | Only `.pdf` and `.docx` file formats are supported. Other extensions return HTTP 400. | Extend `parsers/__init__.py` with additional format handlers as needed. |

---

## 2. Concurrency and Throughput

| Limitation | Impact | Mitigation |
|---|---|---|
| **Single-GPU semaphore serialisation** | The `asyncio.Semaphore(1)` around `model.generate()` serialises all inference requests. Under concurrent load, requests queue sequentially. | Switch to `BACKEND=openai_compatible` and point to a dedicated inference server (vLLM, HuggingFace TGI) with PagedAttention and continuous batching. |
| **No SSE streaming** | The gateway returns `stream=true` requests with HTTP 400. Clients expecting token-by-token streaming are not supported. | Implement SSE streaming or switch to a backend that supports it natively. |
| **In-process model loading** | Loading Phi-3 Mini weights takes ~25-30 seconds on startup. During this window, `/ready` returns 503. | Docker Compose `depends_on: service_healthy` prevents traffic until ready. For faster cold starts, consider pre-warming containers. |

---

## 3. Privacy and PII Redaction

| Limitation | Impact | Mitigation |
|---|---|---|
| **English-only NER** | The spaCy `en_core_web_sm` model supports English text only. Non-English person names and addresses may not be detected. | Load additional spaCy language models and configure Presidio's `NlpEngineProvider` with multi-language support. |
| **Static codename deny-list** | The `PROJECT_CODENAME` recognizer uses a fixed regex pattern. New project codenames require updating the `PROJECT_CODENAMES` environment variable. | Set `PROJECT_CODENAMES` as a comma-separated environment variable for dynamic configuration. |
| **No contextual PII detection** | Presidio relies on pattern matching and statistical NER. Context-dependent PII (e.g., "my address is 42 Oak Street") may not be caught if it doesn't match known entity patterns. | Supplement with custom recognizers for domain-specific PII patterns. |
| **LOCATION and DATE_TIME excluded** | Deliberately excluded to prevent false positives on normal conversational queries. This means physical addresses embedded in text are not redacted. | Re-enable these entity types if the use case requires it, accepting higher false-positive rates. |

---

## 4. Retrieval and RAG

| Limitation | Impact | Mitigation |
|---|---|---|
| **Dense-only retrieval (no BM25/sparse)** | The retrieval pipeline uses dense bi-encoder search only. Exact keyword matches (e.g., specific model numbers, error codes) may rank lower than semantically similar but lexically different chunks. | Implement hybrid retrieval with BM25 sparse search and reciprocal rank fusion (RRF). |
| **Small evaluation corpus** | The chunking evaluation uses 3 PDFs totaling 5 pages and 36 queries. Real enterprise corpora contain thousands of pages. | Expand the evaluation corpus with larger, messier documents to validate retrieval at scale. |
| **Cross-encoder latency** | The `bge-reranker-base` cross-encoder adds ~150ms per query when re-ranking 20 candidates. | Acceptable for conversational RAG (<500ms total). For high-throughput batch processing, consider faster distilled re-rankers. |
| **No chunk deduplication** | If the same document is uploaded twice, duplicate chunks are indexed. | Implement document fingerprinting (e.g., content hash) to detect and reject duplicate uploads. |

---

## 5. Model and Generation

| Limitation | Impact | Mitigation |
|---|---|---|
| **4096 token context window** | Phi-3 Mini's context window limits the amount of RAG context that can be injected. With 3 retrieved chunks, the effective context is ~1500-2000 tokens for source material. | Use larger context models (Phi-3 Medium 128k) or implement sliding-window chunking. |
| **No fine-tuning** | The model runs with base weights. Domain-specific terminology (telecom jargon, internal acronyms) may produce less accurate responses than a fine-tuned variant. | Consider LoRA/QLoRA fine-tuning on domain-specific instruction datasets. |
| **Hallucination risk** | Despite grounded prompting, small language models can still hallucinate facts not present in the provided context. | The grounded prompt explicitly instructs the model to cite sources and refuse when information is absent. Human review is recommended for critical decisions. |

---

## 6. Environment and Deployment

| Limitation | Impact | Mitigation |
|---|---|---|
| **Windows ChromaDB file locking** | On Windows, rapid sequential test runs or abrupt process termination can leave SQLite WAL files locked, causing `OperationalError: database is locked`. | Use `--forked` pytest flag on Windows, or run tests in isolated temporary directories. |
| **Docker GPU passthrough untested** | The `docker-compose.yml` GPU block is commented out. GPU container execution has not been verified in this development environment. | Follow NVIDIA Container Toolkit installation instructions and uncomment the `deploy.resources.reservations.devices` block. |
| **No HTTPS/TLS termination** | Both services run plain HTTP. | Deploy behind a reverse proxy (Nginx, Envoy, Traefik) with TLS termination for production. |
| **No authentication** | The gateway accepts all requests without API key or token verification. | Implement Bearer token authentication or deploy behind an API gateway with OAuth2/OIDC. |
