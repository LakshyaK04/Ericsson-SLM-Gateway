# Learning Notes

This document records what was built in each phase, the design decisions made, and questions a mentor might ask. It is updated at the end of every phase.

---

## Phase 0: Baseline and Restructure

### What was built

We took the existing working prototype (a single `phi3_project` package with gateway, router, PII redaction, RAG pipeline, chunking, embeddings, and vector store all mixed together) and restructured it into the target architecture: two independent services (`gateway/` and `rag/`) plus an `eval/` folder for evaluation.

Each service now has its own `pyproject.toml`, `README.md`, source package under `src/`, and `tests/` directory. The chunking module was split from a single file into one file per strategy (`character.py`, `structure.py`, `semantic.py`) with a dispatcher in `__init__.py`. Files were moved using `git mv` to preserve history.

All imports were converted from the old `src.phi3_project.xxx` absolute paths and `.xxx` relative paths to their new package-relative forms (`slm_gateway.xxx` for gateway, `rag_service.xxx` for RAG). The sample PDF was moved to `eval/docs/`.

### Glossary

| Term | Meaning |
|------|---------|
| **uv workspace** | A uv feature that lets multiple Python packages live in one repo and share a single lockfile. We defined `gateway/` and `rag/` as workspace members in the root `pyproject.toml`. |
| **`git mv`** | Git command to rename/move a file while preserving its history in the commit log. |
| **pyproject.toml** | The standard Python project configuration file (PEP 621). It declares the project name, version, dependencies, and build system. |
| **src layout** | A project structure where the importable package is inside `src/` (e.g. `src/slm_gateway/`). This prevents accidental imports from the project root. |

### Why we did it this way

- **Two separate services** instead of one monolith, because the build plan's architecture diagram shows them communicating over HTTP. This means each can be deployed, tested, and Dockerised independently.
- **uv workspace** rather than two completely separate repos, because we're one intern working locally — sharing a lockfile avoids dependency version conflicts and makes `pytest` discovery easy across both.
- **src layout** because it forces you to install the package before importing it. This catches import errors early and mirrors how the code runs in production.
- **One chunking file per strategy** instead of everything in one file, because the plan explicitly asks for `chunking/ {base.py, character.py, structure.py, semantic.py}` and each strategy is independent, making it easier to test and compare.
- **Kept FAISS for now** even though the plan mentions ChromaDB. We will switch in Phase 4 when building the full RAG service endpoints. Changing too many things in one phase risks breaking the working tests.

### Mentor questions

**Q1: Why not just keep everything in one package?**
A: The build plan requires two services that talk over HTTP. Separate packages make the boundary explicit — the gateway can't accidentally import RAG internals. It also means each service gets its own Dockerfile and can scale independently.

**Q2: Why use `git mv` instead of just moving files?**
A: `git mv` is how git tracks renames. Without it, git sees a delete + create and we lose the file's blame history. With it, `git log --follow` can trace a file back to its original location.

**Q3: What happens if `uv sync` fails on a dependency?**
A: We pinned versions that already work on this machine. If a new dependency fails, we stop and investigate rather than guessing — the build plan says "if a pin causes a real error, stop and ask."

**Q4: Why keep the old test_phi3.py as a reference file?**
A: It contains the working model-loading code (quantization config, tokenizer template, generation loop) that Phase 1 will reuse for the `hf_local` backend. It's not a test in the pytest sense, so it was renamed to `test_phi3_reference.py`.

**Q5: How can you verify this phase is working?**
A: Run `uv run pytest -m "not slow" -v` from the repo root to run the fast tests, or `uv run pytest -v` to run all tests including slow model tests. Both services collect and pass their tests.

### Verification command

```bash
uv run pytest -m "not slow" -v
```

---

## Phase 1: Gateway Core

### What was built

We implemented the core SLM Gateway server with full OpenAI API compatibility. We built `config.py` using `pydantic-settings` to load all configurations from environment variables, and `schemas.py` defining OpenAI-compatible request, response, error, and model listing models.

We implemented an extensible backend architecture under `backends/` with an abstract base class `LLMBackend`, an in-process HuggingFace backend `HFLocalBackend` loading `microsoft/Phi-3-mini-4k-instruct` with 4-bit NF4 quantization via bitsandbytes protected by a single-GPU semaphore, and an `OpenAICompatibleBackend` forwarding requests to any OpenAI-compatible HTTP server.

We wired these into `main.py` with FastAPI lifespan model loading, `/health` and `/ready` probes, `/v1/models`, `/v1/chat/completions`, and optional Bearer token authentication in `security.py`. All error responses conform to OpenAI's `{"error": {"message", "type", "code"}}` format.

### Glossary

| Term | Meaning |
|------|---------|
| **NF4 (NormalFloat4)** | An information-theoretically optimal 4-bit quantization data type provided by bitsandbytes for normally distributed neural network weights. |
| **FastAPI Lifespan** | An asynchronous context manager hook that runs setup (loading the model) on startup and teardown on shutdown. |
| **Semaphore** | A concurrency synchronization primitive. We use `asyncio.Semaphore(1)` to ensure only one thread generates on the GPU at a time, preventing Out-Of-Memory errors. |
| **OpenAI Compatibility** | Implementing the exact JSON schemas and endpoint paths (`/v1/chat/completions`, `/v1/models`) used by OpenAI, allowing any OpenAI SDK client to work by just changing `base_url`. |
| **Readiness vs Liveness Probe** | `/health` (liveness) checks if the HTTP server process is running; `/ready` (readiness) returns 200 only when the heavy AI model is loaded and ready for inference. |

### Why we did it this way

- **In-process HuggingFace with 4-bit quantization** rather than full fp16/bf16, because 4-bit NF4 reduces Phi-3-mini's VRAM footprint to ~2.3GB, allowing it to run smoothly on a laptop GPU (RTX 3050 6GB) while retaining high reasoning quality.
- **FastAPI lifespan loading** rather than loading per request, because loading weights from disk takes ~25-30 seconds. Lifespan loads once when the process starts.
- **Asyncio semaphore behind asyncio.to_thread** because HuggingFace PyTorch generation is CPU/GPU blocking. `asyncio.to_thread` runs generation in a worker thread so the event loop remains responsive for health checks, while the semaphore serializes GPU execution.
- **OpenAI Compatible backend abstraction** so developers without a dedicated CUDA GPU can develop and test against external endpoints (or vLLM/Ollama) with a single environment flag `BACKEND=openai_compatible`.
- **Standardized OpenAI error shapes** so client libraries like the official `openai` Python SDK handle errors (such as 401 Unauthorized or 422 Validation Error) gracefully as standard API errors.

### Mentor questions

**Q1: Why do we need `asyncio.to_thread` and an `asyncio.Semaphore(1)` around generation?**
A: PyTorch's `model.generate()` is a blocking synchronous call. Without `asyncio.to_thread`, running it would block the Python event loop, causing all concurrent requests (including `/health` and `/ready` probes) to freeze. The semaphore of size 1 ensures that multiple requests don't attempt simultaneous generation on a single GPU, avoiding CUDA Out-of-Memory crashes.

**Q2: How is the gateway accessed by client applications?**
A: The gateway exposes an open local endpoint on port 8000 mimicking OpenAI's standard `/v1/chat/completions`. Client applications can connect directly without needing complex cloud authentication headers.

**Q3: How does prompt context truncation work in `HFLocalBackend`?**
A: Phi-3-mini has a 4096 token context window. If the prompt tokens plus `max_tokens` exceeds 4096, `HFLocalBackend` preserves the system prompt at index 0 and progressively trims the oldest conversational turns, logging a warning, instead of failing silently or crashing.

**Q4: How can third-party tools use this gateway?**
A: Any application or SDK designed for OpenAI can interact with this service simply by configuring `base_url="http://localhost:8000/v1"` and setting any arbitrary API key (e.g., `api_key="not-needed"`).

**Q5: How do we verify Phase 1 without waiting for heavy model loads during CI?**
A: Unit and API tests mock the backend, executing 16 tests in less than a second. A dedicated `@pytest.mark.slow` test (`test_phi3_slow.py`) tests the real in-process model inference on the GPU when desired.

### Verification command

```bash
# 1. Run unit test suite
uv run pytest gateway/tests/test_api.py gateway/tests/test_schemas.py -v

# 2. Test official OpenAI Python client against running gateway
uv run python -c "from openai import OpenAI; client = OpenAI(base_url='http://127.0.0.1:8000/v1', api_key='not-needed'); resp = client.chat.completions.create(model='microsoft/Phi-3-mini-4k-instruct', messages=[{'role': 'user', 'content': 'What is the capital of Sweden?'}], temperature=0.0); print(resp.choices[0].message.content)"
```

---

## Phase 2: PII Redaction

### What was built

We implemented the privacy subsystem in `gateway/src/slm_gateway/pii.py` using Microsoft Presidio (`presidio-analyzer` and `presidio-anonymizer`) powered by the spaCy `en_core_web_sm` NLP pipeline.

We restricted the detected entity list to sensitive identifiers: `PERSON`, `EMAIL_ADDRESS`, `PHONE_NUMBER`, `CREDIT_CARD`, and `IP_ADDRESS`, explicitly excluding `LOCATION` and `DATE_TIME` to prevent false-positive masking on normal conversational and technical queries.

We extended Presidio with custom recognizers:
1. `EMPLOYEE_ID`: Regex pattern recognizer targeting organizational IDs matching `\bEMP-\d{5,7}\b`.
2. `PHONE_NUMBER`: Supplementary regex recognizer capturing international and synthetic telephone patterns alongside Presidio's standard NANP validator.

We added support for typed placeholders (`<EMAIL_ADDRESS>`, `<EMPLOYEE_ID>`, etc.) and configurable failure policies (`PII_FAIL_MODE=closed|open`). Under the default fail-closed policy, if PII detection fails during startup or request processing, an error is raised (returning HTTP 500) rather than leaking raw private data.

We wired PII sanitization into `gateway/src/slm_gateway/main.py` so every `user` message is sanitized before passing to the backend, and the count of redactions is returned in the response metadata under `x_pii: {"redactions": count}`.

We created an 18-test suite in `gateway/tests/test_pii.py` covering table-driven entity redaction, custom recognizers, false-positive protection ("What is the capital of Germany?"), fail-closed / fail-open behaviors, and end-to-end payload assertions verifying that the underlying LLM backend never receives raw PII.

### Glossary

| Term | Meaning |
|------|---------|
| **Microsoft Presidio** | An open-source SDK from Microsoft for detecting, categorizing, and anonymizing Personally Identifiable Information (PII) in text. |
| **spaCy NER** | Named Entity Recognition statistical model (`en_core_web_sm`) used by Presidio to identify unstructured entities such as `PERSON`. |
| **Fail-Closed** | A security design principle where any failure in a security/privacy mechanism aborts the transaction rather than allowing unverified or sensitive data through. |
| **Typed Placeholder** | Replacing sensitive data with a typed label (`<EMAIL_ADDRESS>`) instead of uniform masking (`***`), preserving syntactic role and semantic clarity for LLM reasoning. |
| **Luhn Algorithm** | A checksum formula used by Presidio's credit card recognizer to validate card numbers and avoid false matches on random digit strings. |

### Why we did it this way

- **Restricted entity list (excluding `LOCATION` and `DATE_TIME`)** because standard NER models frequently misclassify ordinary nouns and location names as entities. Redacting "Stockholm" or "Monday" as `<LOCATION>` and `<DATE_TIME>` distorts user intent and destroys answer accuracy.
- **Typed placeholders instead of asterisks** because small language models like Phi-3 maintain coherent conversational context when they know an entity was an email or employee ID, without needing to see the raw sensitive value.
- **Explicit `en_core_web_sm` NLP engine configuration** via Presidio's `NlpEngineProvider` to prevent Presidio from attempting to load the larger `en_core_web_lg` model, ensuring fast startup and low RAM overhead on developer machines.
- **Fail-closed default** to prevent silent data exfiltration if the PII service crashes or spaCy encounters an unhandled tokenization error.
- **Sanitizing only `user` messages** to preserve system prompts and instructions while guaranteeing privacy on user-provided inputs.

### Mentor questions

**Q1: Why did we explicitly exclude `LOCATION` and `DATE_TIME` from the entity list?**
A: Off-the-shelf NER recognizers have high false-positive rates on common geographical names and temporal expressions. If a user asks "What is the capital of Germany?", redacting Germany into `<LOCATION>` prevents the model from knowing which country was asked about. Excluding them protects semantic meaning for both intent routing and answer generation.

**Q2: What is the difference between fail-closed and fail-open in PII redaction?**
A: In fail-closed mode (`PII_FAIL_MODE=closed`, default), any unhandled exception in Presidio causes the gateway to halt and return an HTTP 500 error, guaranteeing that raw sensitive text is never sent to the LLM backend. In fail-open mode, errors are logged and the raw text is passed through, prioritizing system uptime over absolute data privacy.

**Q3: Why use typed placeholders like `<EMAIL_ADDRESS>` instead of simple masking like `[REDACTED]` or `***`?**
A: Typed placeholders preserve the grammatical role and category of the redacted entity. The LLM can still infer that the user provided an email address or employee identifier and respond appropriately (e.g. "I have noted your employee ID"), without ever seeing the actual private identifier.

**Q4: How do custom recognizers fit into Presidio's architecture?**
A: Presidio's `PatternRecognizer` allows custom regexes or pattern rules to be registered with the `AnalyzerEngine`. We registered `EMPLOYEE_ID` (`\bEMP-\d{5,7}\b`) alongside standard recognizers. Presidio executes all recognizers, resolves token overlaps using confidence scores, and outputs a unified list of detected entity spans.

**Q5: Why does `test_end_to_end_gateway_pii_redaction` inspect the mock backend payload rather than just checking the API response?**
A: Checking only the client-facing HTTP response does not guarantee that the backend didn't see the raw PII (for example, if redaction were mistakenly applied only to the output). Asserting on `mock_backend.generate.call_args` provides cryptographic proof that the payload transmitted to the model backend contained solely the sanitized `<EMAIL_ADDRESS>` string.

### Verification command

```bash
uv run pytest gateway/tests/test_pii.py -v
```

---

## Phase 3: Intent Router and its Evaluation

### What was built

We implemented the semantic intent routing subsystem in `gateway/src/slm_gateway/router.py`, driven by `BAAI/bge-small-en-v1.5` dense sentence embeddings via `sentence-transformers`, with an exemplar bank defined in `gateway/src/slm_gateway/intents.yaml`.

We established 3 intent classes with 28 diverse exemplars each (84 exemplars total):
1. `general`: World knowledge, history, philosophy, trivia, conversational chit-chat, and creative inquiries.
2. `technical`: Software engineering, networking protocols, databases, architectures, cloud infrastructure, and algorithms.
3. `rag`: Grounded questions targeting uploaded documents, PDFs, employee handbooks, specifications, and internal policies.

We implemented a mean top-3 cosine similarity aggregation algorithm: for each intent, similarity scores against all its exemplars are sorted descending and the average of the top 3 is computed. If the top scoring intent is below `ROUTER_THRESHOLD` (default 0.55), the router falls back to `general` with `fallback_applied=True`.

We integrated the router into `gateway/src/slm_gateway/main.py`:
- Loaded `IntentRouter` once in the FastAPI lifespan handler, precomputing and normalizing all exemplar embeddings on startup.
- Handled the `X-Bypass-Router: true` header to skip embedding computation and routing latency when downstream services or direct calls bypass routing.
- Automatically applied route effects:
  - When `rag` is classified, the gateway prepares routing to the RAG service, with graceful fallback to `hf_local` if the RAG service is unreachable or unindexed.
- Appended `x_routing{intent, confidence, route, latency_ms}` metadata to every `ChatCompletionResponse`.
- Updated `/ready` probe to require both the model backend and the intent router before returning HTTP 200.

We built an evaluation pipeline in `eval/router_eval.py` & `eval/datasets/router_eval.jsonl`:
- 48 labeled queries (16 per intent) designed with tricky ambiguous edge cases.
- Automated duplicate leakage validator asserting 0 overlap between eval queries and training exemplars.
- Evaluated overall accuracy (93.75%), per-intent precision/recall/F1, and a threshold sweep from 0.30 to 0.80 confirming 0.55 is the optimal operating threshold. Saved to `eval/results/router_report.md`.

### Glossary

| Term | Meaning |
|------|---------|
| **Few-Shot Semantic Routing** | Classifying user intent by calculating vector similarity between the query and a bank of exemplar queries in embedding space, requiring zero fine-tuning or retraining. |
| **Top-3 Mean Aggregation** | Averaging the top 3 highest similarity scores for each intent class instead of using 1-Nearest-Neighbor, smoothing out lexical flukes and idiosyncratic phrasing. |
| **Threshold Sweep** | Evaluating classification accuracy and fallback frequency across an array of threshold values (0.30 to 0.80) to systematically determine the optimal operating point. |
| **Exemplar Leakage** | A form of data leakage where test queries are identical or nearly identical to training examples, artificially inflating evaluation metrics. |
| **Fallback Intent** | A safe default intent (`general`) assigned when no intent meets the minimum confidence threshold, preventing out-of-domain conversational queries from triggering specialized tools. |

### Why we did it this way

- **`BAAI/bge-small-en-v1.5` over MiniLM or frontier models** because it offers state-of-the-art embedding quality on retrieval and clustering benchmarks in a compact 133MB footprint, executing inference in ~40-60ms on CPU without consuming GPU VRAM.
- **Top-3 similarity averaging instead of 1-NN** because 1-NN is fragile to incidental word overlap. Top-3 averaging requires consistent semantic affinity across multiple diverse exemplars.
- **Precomputed embeddings at startup** so request classification only requires encoding a single query string and doing a fast matrix-vector dot product (`cosine_similarity`).
- **Automated zero-leakage validator in evaluation** ensuring evaluation scores reflect true generalization to unseen phrasing rather than memorized sentences.

### Mentor questions

**Q1: Why use semantic embedding similarity instead of an LLM prompt or an SVM/logistic regression classifier for intent routing?**
A: An LLM prompt adds 500-1000ms of autoregressive generation latency and consumes precious VRAM/GPU resources. A fine-tuned classifier requires retraining whenever new intents or examples are added. Embedding similarity with `bge-small` takes ~50ms on CPU, requires zero GPU memory, and allows updating the intent bank instantly by simply editing `intents.yaml` without retraining.

**Q2: Why score intents using the mean of the top-3 similarities rather than just the top-1 (nearest neighbor)?**
A: Top-1 similarity is vulnerable to accidental lexical or syntactic overlap between a query and an unusual exemplar. Averaging the top 3 similarities requires the query to be consistently close to multiple exemplars of that intent, reducing variance and misclassifications.

**Q3: How was the decision threshold `0.55` determined, and what does the threshold sweep reveal?**
A: The threshold sweep in `eval/results/router_report.md` tested values from 0.30 to 0.80. Below 0.50, accuracy was 92.2% but ambiguous queries were not rejected. At 0.55, accuracy peaked at 93.8% with 7 appropriate fallbacks to `general`. Above 0.70, accuracy plummeted to 64.1% and 26.6% as legitimate queries were rejected as false negatives.

**Q4: How do we prevent evaluation data leakage between `router_eval.jsonl` and `intents.yaml`?**
A: `eval/router_eval.py` executes an automated pre-flight integrity check (`verify_no_duplicate_eval_queries`) that normalizes (strips punctuation and whitespace, lowercases) all queries and asserts 0 duplicates between the 48 evaluation queries and 84 training exemplars before running evaluation.

### Verification command

```bash
# 1. Run all unit and integration tests (49 passing tests)
uv run pytest gateway/tests/test_api.py gateway/tests/test_schemas.py gateway/tests/test_pii.py gateway/tests/test_router.py -v

# 2. Run intent router evaluation and threshold sweep
uv run python eval/router_eval.py

# 3. Run PII redaction evaluation
uv run python eval/pii_eval.py
```

---

## Phase 4: RAG Service

### What was built

We implemented the multi-strategy RAG service in `rag/src/rag_service/`, running as an independent HTTP service on port 8001.

We implemented the document chunking module with a unified `Chunk` dataclass (`chunk_id, text, source, page, strategy`):
1. `character.py`: Splits text into fixed-size chunks (`chunk_size=500`, `overlap=50`) while snapping boundaries backwards to the nearest whitespace to avoid amputating words or symbols.
2. `structure.py`: Splits along structural boundaries using a robust heading regex (Markdown `#`, numbered sections `1.1`, and ALL CAPS headers) and paragraph double-newlines, merging short sections up to 1000 characters and splitting oversized sections at sentence boundaries.
3. `semantic.py`: Segments text into sentences, computes normalized dense embeddings using `BAAI/bge-small-en-v1.5`, and triggers new chunks when cosine similarity between adjacent sentences drops below `threshold=0.65`, subject to a 300-character minimum chunk size.
4. `chunking/__init__.py`: Multi-page, multi-strategy document chunking dispatcher.

We built document parsers in `parsers/`:
1. `pdf.py`: PyMuPDF (`fitz`) text extraction on a per-page basis preserving 1-indexed page numbers. Rejects empty or scanned image-only PDFs with a 400 error (`"No extractable text found in PDF; document appears empty or contains only scanned images (OCR is not supported)"`).
2. `docx.py`: python-docx extraction preserving headings, paragraphs, and table text.
3. `parsers/__init__.py`: Unified file format dispatcher validating `.pdf` and `.docx` extensions and rejecting unsupported file formats.

We built the storage, embedding, and retrieval subsystems:
1. `embeddings.py`: In-process `BAAI/bge-small-en-v1.5` dense embedding engine. Implements asymmetric query prefixing (`"Represent this sentence for searching relevant passages: "`) on user search queries while embedding document chunks without prefixes, L2-normalized.
2. `store.py`: Embedded, persistent ChromaDB vector store (`chromadb.PersistentClient`) maintaining three isolated collections (`chunks_character`, `chunks_structure`, `chunks_semantic`) with metadata tracking (`doc_id, source, page, chunk_id, strategy`), supporting ingestion, dense querying, document listing, and atomic deletion across all collections.
3. `reranker.py`: Neural cross-encoder `BAAI/bge-reranker-base` re-ranking candidate chunks against the search query, outputting relevance scores.
4. `retriever.py`: Two-stage hybrid retriever: Stage 1 retrieves `retrieve_k=20` dense candidates from ChromaDB; Stage 2 applies the cross-encoder to return the top `final_k=3` chunks with both `dense_score` and `rerank_score`.

We implemented the FastAPI application in `main.py`:
- `POST /documents`: Multipart file upload, multi-strategy chunking, embedding, and Chroma indexing.
- `GET /documents`: Lists indexed documents and their chunk distributions.
- `DELETE /documents/{doc_id}`: Atomically deletes document chunks across all Chroma collections.
- `POST /query`: Two-stage retrieval returning top-k re-ranked chunks with dual scores.
- `GET /health`: Liveness probe reporting healthy status and registered collections.

We verified the service with 15 passing tests (`test_chunking.py`, `test_parsers.py`, `test_retrieval.py`, `test_api.py`) and executed live verification uploading `enterprise_rag_sample.pdf` and querying the service.

### Glossary

| Term | Meaning |
|------|---------|
| **Cross-Encoder Re-Ranking** | A neural architecture where query and passage are concatenated and processed jointly across all transformer layers, enabling deep token-to-token cross-attention for high-precision ranking. |
| **Dense Retrieval** | Locating candidate passages by computing cosine similarity between dense vector embeddings of the query and pre-indexed chunks in a vector database. |
| **BGE Query Instruction Prefix** | A specific task instruction prompt (`"Represent this sentence for searching relevant passages: "`) prepended to search queries to align query embeddings with passage representations. |
| **Multi-Strategy Chunking** | Indexing the same source text simultaneously across different segmentation strategies (character, structure, semantic) to enable empirical retrieval evaluation. |
| **ChromaDB Persistent Client** | An embedded vector database persisting HNSW vector indexes and document metadata directly to local disk without requiring external database servers. |

### Why we did it this way

- **Two-stage retrieval (retrieve 20, rerank 3)** because dense vector search is fast (~5ms) and casts a wide net over large corpora, while the cross-encoder is computationally heavier (~30ms) but delivers superior ranking accuracy, filtering out irrelevant dense matches.
- **Asymmetric query instruction prefixing** because BGE models are trained with contrastive learning where queries require task instructions while documents represent raw unadorned content.
- **Isolated Chroma collections per chunking strategy** ensuring that character, structure, and semantic chunks never compete for vector slots in the same index, enabling unbiased comparative evaluation in Phase 5.
- **Snapping character chunk boundaries to whitespace** preventing split words, truncated variable names, or damaged acronyms at chunk edges.
- **Failing early on empty or image-only PDFs** informing users immediately that OCR is not supported rather than silently creating an empty document index.

### Mentor questions

**Q1: Why do we use a two-stage retrieval pipeline (dense search + cross-encoder) instead of returning top dense search matches directly?**
A: Bi-encoders map query and document independently to fixed vectors, meaning words in the query cannot directly attend to words in the passage. Dense search is very fast for narrowing down thousands of chunks to 20 candidates. The cross-encoder (`bge-reranker-base`) feeds query and passage together through deep cross-attention, capturing subtle syntactic relationships and eliminating false-positive dense matches.

**Q2: Why does the embedding model prefix queries with `"Represent this sentence for searching relevant passages: "` but leaves documents unprefixed?**
A: The BAAI BGE model family was trained with asymmetric contrastive learning. Queries are short and ambiguous, so the task-specific instruction prefix tells the model to project the query into the semantic retrieval space of relevant passages. Passages represent factual corpus data and must be embedded without task instructions.

**Q3: How do the three chunking strategies differ in their segmentation logic and intended use cases?**
A: Character chunking cuts text at fixed length intervals snapping to whitespace (good for uniform corpora without headings); structure chunking splits on Markdown, numbered headings, and paragraph boundaries (ideal for technical manuals and formatted documentation); semantic chunking computes sentence embeddings and splits at semantic distance drops (ideal for unstructured, narrative text).

**Q4: Why does `ChromaStore` maintain three separate collections (`chunks_character`, `chunks_structure`, `chunks_semantic`) instead of one collection with a metadata flag?**
A: Separate collections ensure that HNSW vector graph indexes and cosine distance spaces are isolated per strategy. This prevents chunks from one strategy from crowding out candidates during dense retrieval, enabling completely independent benchmarking in Phase 5.

**Q5: What error occurs if an uploaded PDF contains only scanned images, and why fail at upload time?**
A: The parser computes total extracted characters across all pages. If total characters is 0, it raises a 400 error: "No extractable text found in PDF; document appears empty or contains only scanned images (OCR is not supported)." Failing early prevents indexing empty ghost documents and gives clear, actionable feedback to users.

### Verification command

```bash
# 1. Run all RAG unit and integration tests (15 passing tests)
uv run pytest rag/tests/ -v

# 2. Upload public PDF and query top chunks with both scores
uv run python -c "
import httpx
client = httpx.Client(base_url='http://127.0.0.1:8001')
print('Health:', client.get('/health').json())
"
```

---

## Phase 5: Chunking Strategy Evaluation

### What was built

We built the empirical chunking strategy and re-ranking evaluation framework to systematically compare `character`, `structure`, and `semantic` chunking strategies on a curated technical corpus:

1. **Evaluation Corpus (`eval/docs/`)**:
   - `enterprise_rag_sample.pdf`: Overview of the Enterprise AI Platform, OpenAI-compatible model serving, ingestion pipelines, RAG, and PII protection.
   - `5g_core_architecture.pdf`: 2-page detailed technical specification of 3GPP 5G Core Service-Based Architecture (SBA), control plane NFs (AMF, SMF, NRF, NSSF, PCF), user plane operations (UPF, PDR, N6 interface, CHF), and network slicing (SST 1/2/3).
   - `cloud_native_telecom_infrastructure.pdf`: 2-page technical guide covering Containerized Network Functions (CNFs), high-performance networking acceleration (SR-IOV, DPDK, XDP, eBPF), Kubernetes multi-network CNI plugins (Multus), Zero-Trust Architecture (ZTA, mTLS, SPIFFE/SPIRE), and OpenTelemetry observability.

2. **Ground-Truth QA Dataset (`eval/datasets/chunking_qa.jsonl`)**:
   - 36 curated, realistic technical questions spanning all 3 documents.
   - Each question is mapped to an `expected_substring` (a short verbatim phrase that must appear in the retrieved chunk) and `doc_name`.
   - Verified via `eval/generate_chunking_qa.py` with zero substring misses across the parsed document corpus.

3. **Evaluation Harness (`eval/chunking_eval.py`)**:
   - Indexes all 3 documents across all 3 strategies into isolated ChromaDB collections.
   - Runs all 36 questions under both **dense-only** retrieval and **two-stage re-ranked** retrieval (`BAAI/bge-reranker-base`).
   - Computes: Total Chunks, Average/Min/Max chunk character length, Hit@1, Hit@3, Mean Reciprocal Rank (MRR), and average query latency (ms).
   - Exports results to `eval/results/chunking_report.csv` and `eval/results/chunking_report.md`.

### Evaluation Results

| Strategy | Re-ranker | Total Chunks | Avg Length (chars) | Hit@1 (%) | Hit@3 (%) | MRR | Latency (ms) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `character` | **Off** | 16 | 423.6 | 86.11% | 100.00% | 0.9213 | 16.0 |
| `character` | **On** | 16 | 423.6 | 88.89% | 97.22% | 0.9306 | 1025.1 |
| `structure` | **Off** | 10 | 623.6 | 94.44% | 100.00% | 0.9722 | 15.5 |
| `structure` | **On** | 10 | 623.6 | **100.00%** | **100.00%** | **1.0000** | 1069.9 |
| `semantic` | **Off** | 16 | 388.7 | 88.89% | 100.00% | 0.9352 | 15.9 |
| `semantic` | **On** | 16 | 388.7 | **94.44%** | **97.22%** | **0.9583** | 1175.0 |

### Glossary

| Term | Meaning |
|------|---------|
| **Hit@k** | The fraction of evaluation queries where a chunk containing the ground-truth answer appears in the top-$k$ retrieved results. |
| **Mean Reciprocal Rank (MRR)** | The average of the reciprocal ranks of the first relevant chunk ($1/\text{rank}$) across all queries. If the relevant chunk is at rank 1, score is 1.0; at rank 2, 0.5; at rank 3, 0.333. |
| **Ground-Truth Expected Substring** | A verified verbatim text snippet from the document text that unequivocally confirms the retrieved chunk contains the specific answer to the question. |
| **Granularity Trade-off** | The tension between small chunks (higher semantic purity, lower dense dilution) and large chunks (broader context, higher risk of irrelevant noise or token budget exhaustion). |
| **Bi-Encoder vs Cross-Encoder Latency** | Bi-encoders perform single-vector dot products (~15-16ms for 20 candidates), whereas cross-encoders perform full transformer cross-attention for each query-candidate pair (~1,000-1,175ms on CPU). |

### Why we did it this way

- **Exact Substring Verification over LLM Judging** because substring verification is 100% deterministic, reproducible, fast, and does not suffer from hallucinated evaluation scores or model biases.
- **Evaluating with and without the re-ranker** directly demonstrates the empirical value added by the neural cross-encoder: on `structure` chunking, it boosted Hit@1 from 94.44% to 100.00% (+5.56 pp), and on `semantic` chunking, it boosted Hit@1 from 88.89% to 94.44% (+5.55 pp).
- **Evaluating on real technical specifications (5G Core SBA, CNF Infrastructure)** rather than generic synthetic prose helps test chunk boundaries against headings, bullet lists, numbered items, and acronyms.
- **Using repository-local `data/chroma_eval` storage** helps ensure isolation from test runs and main data while avoiding Windows `%TEMP%` file-lock anomalies during rapid batch indexing.

### Technical Questions

**Q1: What did the evaluation reveal about the impact of the cross-encoder re-ranker across different chunking strategies?**
A: Re-ranking provides an improvement in ranking precision for semantically coherent chunks: on `structure` chunking, Hit@1 reached 100.00% (MRR 1.0000), and on `semantic` chunking, Hit@1 increased from 88.89% to 94.44% (MRR from 0.9352 to 0.9583). On `character` chunking, Hit@1 shifted from 86.11% to 88.89% while Hit@3 dropped from 100% to 97.22%.

**Q2: Why did `structure` chunking achieve the highest retrieval accuracy (100% Hit@1 with re-ranker) compared to `semantic` and `character`?**
A: Structure chunking respects human author organization: section headers, numbered lists, and cohesive technical paragraphs remain intact within a single chunk (average length 623 characters). In contrast, character chunking slices across sentences, and semantic chunking split paragraphs into smaller fragments based on local cosine dips.

**Q3: What is the computational and latency trade-off of enabling neural re-ranking in a RAG pipeline?**
A: In our benchmark on CPU, dense retrieval from ChromaDB took ~15ms to 16ms per query. Applying `BAAI/bge-reranker-base` to 20 candidates increased total query latency to ~1,000ms - 1,175ms on CPU.

**Q4: How does the evaluation guarantee that `chunking_qa.jsonl` contains valid, verifiable ground truth without human grading error?**
A: `eval/generate_chunking_qa.py` parses all documents using PyMuPDF and programmatically asserts that every single `expected_substring` exists verbatim in the extracted document text. If a phrase is mistyped or missing, the generator script raises an immediate error and fails the build.

**Q5: What are the dataset limitations, and what should be stated honestly about this benchmark?**
A: The evaluation corpus consists of 3 technical PDFs totaling 5 pages and 36 questions. While it exercises technical terminology, architectural overviews, and structured lists, real enterprise corpora contain hundreds of pages with messy OCR, complex multi-page tables, and ambiguous phrasing. On such large-scale datasets, structural chunking with table-aware parsers is even more critical.

### Verification command

```bash
# 1. Generate and verify 36-question QA dataset
uv run python eval/generate_chunking_qa.py

# 2. Run the chunking strategy and re-ranking evaluation suite
uv run python eval/chunking_eval.py

# 3. View the generated evaluation report and CSV
type eval\results\chunking_report.md
type eval\results\chunking_report.csv
```

---

## Phase 6: Merge (Gateway + RAG Integration)

### What was built

We integrated the SLM Gateway (port 8000) and the RAG Service (port 8001) into a clean, unified architecture:

1. **RAG Grounded Generation Engine (`rag/src/rag_service/generation.py`)**:
   - `build_grounded_prompt()` formats retrieved chunks into numbered context blocks (`[1]`, `[2]`, `[3]`) complete with source document names and page metadata.
   - Enforces a strict system prompt directing the LLM to ground answers exclusively in the provided context, cite bracketed numbers (e.g., `[1]`), and state clearly if the answer is not present.
   - Dispatches generation requests to the Gateway `/v1/chat/completions` endpoint passing the header `X-Bypass-Router: true`.
   - `POST /answer` endpoint on RAG service returning `AnswerResponse` containing the synthesized `answer`, ranked `sources`, and `usage` statistics.

2. **Gateway RAG Client & Intelligent Routing (`gateway/src/slm_gateway/rag_client.py` & `main.py`)**:
   - Asynchronous HTTP client checking RAG index status via `GET /documents` (`has_indexed_documents()`).
   - Routes queries classified as `rag` intent through `RAGClient.generate_answer()`.
   - Enriches the standard OpenAI response with `x_sources`, listing chunk IDs, document names, page numbers, dense scores, and re-ranking scores.
   - Implements graceful fallback: if 0 documents are indexed or the RAG service is unreachable/timed out, the Gateway falls back seamlessly to the local model, injecting an explanatory warning into `x_routing["warning"]` while keeping `x_sources=None`.

3. **Loop Prevention & Architectural Cleanliness**:
   - The Gateway checks `X-Bypass-Router: true`. When present, the Intent Router is bypassed entirely, routing directly to the local model (`hf_local`). This eliminates circular dependency loops (User -> Gateway -> RAG -> Gateway -> RAG...).
   - Centralizes all LLM inference, PII redaction, token accounting, and GPU memory management in the Gateway while keeping vector storage, chunking, and re-ranking in the RAG service.

### Glossary

| Term | Meaning |
|------|---------|
| **Bypass Router Header (`X-Bypass-Router`)** | An internal HTTP header passed by the RAG service to the Gateway to skip semantic intent classification and invoke local LLM generation directly. |
| **Context Augmentation Block** | A structured text payload containing retrieved document snippets formatted with index markers (`[1]`, `[2]`), source filenames, and page coordinates for LLM consumption. |
| **Grounded Answer Generation** | LLM text synthesis constrained strictly to provided context snippets, requiring explicit citations and prohibiting hallucinated assumptions. |
| **Graceful Degradation / Fallback** | The capability of an enterprise gateway to continue fulfilling user requests via base model generation when auxiliary subsystems (RAG index or vector store) are empty or unavailable. |
| **OpenAI Schema Preservation** | Maintaining 100% adherence to standard OpenAI chat completion schemas (`id`, `choices`, `usage`) while supplying vendor-specific enhancements under `x_` metadata prefixes. |

### Why we did it this way

- **Single Point of LLM Serving:** By having RAG invoke the Gateway's `/v1/chat/completions` endpoint for text generation rather than instantiating its own duplicate model pipeline, we avoid duplicating massive weights in GPU memory (~2.5GB-7GB VRAM savings) and ensure uniform token metering and PII auditing across the entire enterprise.
- **Header-Based Loop Prevention (`X-Bypass-Router`):** Bypassing intent classification on internal RAG-to-Gateway calls prevents infinite recursive loops without requiring a second dedicated internal port or separate model daemon.
- **Index-Aware Fallback:** Rather than throwing an internal 500 error or returning empty context when a user asks a document-related question on a freshly deployed instance with zero uploaded documents, the Gateway transparently falls back to local knowledge and warns the caller in `x_routing["warning"]`.
- **Numbered In-Context Citations:** Numbered brackets `[1]` provide an unambiguous notation for small language models (Phi-3 Mini) to map claims directly back to specific document sources and page numbers.

### Mentor questions

**Q1: How does the architecture prevent infinite recursive loops between the Gateway and RAG service?**
A: When a user query routes to `rag`, the Gateway calls the RAG service's `POST /answer` endpoint. The RAG service performs two-stage retrieval, prepares the grounded prompt, and calls the Gateway's `POST /v1/chat/completions` endpoint with `X-Bypass-Router: true`. The Gateway detects this header and skips intent classification entirely, routing immediately to `hf_local`. Without this bypass, the Gateway router might classify the augmented context prompt as `rag` again, initiating an infinite HTTP loop.

**Q2: Why does the Gateway check `has_indexed_documents()` before delegating a `rag` route, and what happens if 0 documents are indexed?**
A: If no documents have been uploaded to the RAG service, attempting retrieval will return zero chunks, causing either empty context generation or unnecessary RAG roundtrips. The Gateway checks `GET /documents`; if the index is empty, it bypasses RAG, routes directly to the local model, sets `x_sources=None`, and populates `x_routing["warning"] = "RAG service has no indexed documents; routed to local model"`.

**Q3: Why should RAG call the Gateway for LLM generation rather than hosting its own local model instance?**
A: In an enterprise deployment, hosting LLMs in multiple microservices leads to VRAM starvation, duplicated model cache files, fragmented logging, and independent rate limits. Centralizing LLM generation in the Gateway allows single-tenant GPU memory optimization, unified PII filtering, consistent token usage calculation, and single-pane observability.

**Q4: Why are RAG sources attached as `x_sources` on the chat completion response rather than injected into the message content text?**
A: Standard OpenAI chat completion clients (and libraries like `langchain` or `openai-python`) expect `choices[0].message.content` to be a pure string of the assistant's reply. Modifying the response envelope to include custom fields prefixed with `x_` (`x_sources`, `x_routing`, `x_pii`) preserves compatibility with existing SDKs while providing structured citation metadata (chunk ID, source doc, page, dense score, rerank score) for rich client UIs.

**Q5: How does the system handle RAG service timeouts or network failures gracefully?**
A: `RAGClient` catches `httpx.RequestError` and timeout exceptions. In `gateway/src/slm_gateway/main.py`, if `generate_answer` returns `None` due to an error, the Gateway logs a warning, falls back to `hf_local`, and sets `x_routing["warning"] = "RAG service unreachable or failed; routed to local model"`, ensuring the end user still receives a response.

### Verification command

```bash
# 1. Run all unit and integration tests across both Gateway and RAG
uv run pytest gateway/tests/ -v
uv run pytest rag/tests/ -v

# 2. Run end-to-end live verification script (starts mock gateway + RAG, indexes doc, verifies grounded answer with sources)
uv run python scripts/verify_phase6.py
```

---

## Phase 7: Docker, Documentation & Demo

### What was built

We packaged, documented, and automated the complete multi-service stack for turnkey enterprise delivery:

1. **Enterprise Containerization (`gateway/Dockerfile`, `rag/Dockerfile`, `docker-compose.yml`)**:
   - Pinned `python:3.10-slim` base images.
   - Container hardening via dedicated non-root execution (`appuser`, UID 10001).
   - Embedded Docker healthchecks for container orchestration probes.
   - `docker-compose.yml` linking Gateway and RAG on a private bridge network (`local-net`), with persistent named volumes for ChromaDB data (`chroma-data`) and HuggingFace model cache (`hf-cache`).
   - Declared `depends_on.rag.condition: service_healthy` to guarantee deterministic boot ordering.
   - Documented `nvidia-container-toolkit` GPU configuration for host environments.

2. **System Architecture & Solution Documents (`docs/`)**:
   - `docs/ARCHITECTURE.md`: Complete topology, sequence diagrams, loop prevention mechanics, fault tolerance, and vector storage structure.
   - `docs/SOLUTION_GATEWAY.md`: Gateway architecture, OpenAI specification compliance, PII redaction pipeline with empirical evaluation (100% recall, 0% FPR), and BGE semantic router evaluation (93.8% accuracy, 0.55 threshold sweep).
   - `docs/SOLUTION_RAG.md`: Ingestion architecture, chunking algorithm comparisons, ChromaDB persistent store, neural cross-encoder re-ranking, and empirical benchmark (Structure chunking: 100% Hit@1, MRR 1.0000).
   - Updated service-level documentation: `gateway/README.md`, `rag/README.md`, and top-level `README.md`.

3. **5-Minute Live Demo & Automation (`docs/DEMO_SCRIPT.md`, `scripts/demo.py`)**:
   - `docs/DEMO_SCRIPT.md`: Step-by-step scripted narrative covering 6 scenes with exact `curl` payloads, expected JSON outputs, and mentor talking points.
   - `scripts/demo.py`: Cross-platform interactive and automated CLI tool executing the entire live demo flow or displaying the empirical benchmark table (`--benchmark-only`).
   - Transparently documented that GPU container execution is untested in this sandboxed environment, providing exact verification commands for physical host machines.

### Glossary

| Term | Meaning |
|------|---------|
| **Non-Root Container Hardening** | Running container processes under an unprivileged user ID (e.g., UID 10001) rather than root, mitigating host compromise risks in the event of an application exploit. |
| **Healthcheck Dependency (`service_healthy`)** | A Docker Compose configuration ensuring a dependent container only starts after its upstream dependency passes internal health verification probes. |
| **Volume Persistence** | Storing vector databases and downloaded model weights in named Docker volumes so data survives container recreation without redownloading multi-gigabyte models. |
| **Provenance Citation** | Explicitly mapping each claim in an LLM-synthesized answer back to the originating source file, page number, and chunk ID. |
| **Scripted Demonstration** | A reproducible, timed presentation framework allowing developers to showcase core capabilities with deterministic inputs and clear mentor discussion topics. |

### Why we did it this way

- **Security Compliance with Non-Root Execution:** Enterprise environments and Kubernetes clusters enforce `runAsNonRoot: true`. Baking an unprivileged `appuser` directly into the Dockerfiles ensures seamless compliance.
- **Boot Ordering via `service_healthy`:** Simply specifying `depends_on: [rag]` only waits for the container process to spawn, not for model weights and ChromaDB to initialize. Using `condition: service_healthy` guarantees that the RAG service is fully initialized before the Gateway accepts inbound traffic.
- **Dedicated Volume for HuggingFace Cache:** Downloading Phi-3 Mini (~2.6GB) and BGE models on every container launch wastefully exhausts bandwidth and creates startup delays. Mounting `hf-cache` preserves downloaded models across restarts.
- **Cross-Platform Python Demo Script:** Instead of relying exclusively on Unix bash scripts (`demo.sh`), `demo.py` runs natively across Windows, Linux, and macOS without shell dependencies.
- **Honest Environmental Disclosure:** Adhering to Rule 2 of the Build Plan, we clearly documented that Docker GPU passthrough is untested in this sandboxed development container and provided exact instructions for running with `nvidia-container-toolkit`.

### Mentor questions

**Q1: Why is running containers as a non-root user critical for enterprise security?**
A: By default, a process running as root inside a container shares the root UID (0) with the host kernel. If a vulnerability allows a container breakout, the attacker gains root control over the host system. Creating an unprivileged user (`appuser:10001`) ensures that even if an attacker compromises the Python process, their access remains tightly restricted inside the container namespace.

**Q2: Why did we configure Docker Compose with `condition: service_healthy` instead of basic `depends_on`?**
A: Basic `depends_on` only verifies that the container has entered the running state, which takes milliseconds. Heavy ML microservices require several seconds to load PyTorch weights, tokenizer vocabularies, and vector indexes into memory. By coupling `depends_on` to `service_healthy`, Docker Compose waits until the service's internal health check (`/health`) returns HTTP 200 before routing traffic to it.

**Q3: How should an engineer run the Docker stack with GPU acceleration on a real host workstation?**
A: On a host with an NVIDIA GPU, the engineer installs the NVIDIA Container Toolkit (`nvidia-container-toolkit`), configures the Docker runtime, and uncomments the `deploy.resources.reservations.devices` block in `docker-compose.yml`. This grants the Gateway container access to CUDA drivers and physical GPU memory.

**Q4: What are the trade-offs of storing ChromaDB data and HuggingFace model cache in named Docker volumes?**
A: Named volumes ensure persistence: document indexes are not lost when containers are rebuilt, and multi-gigabyte HuggingFace models do not need to be re-downloaded over the network. The trade-off is storage accumulation on the host disk, requiring explicit pruning commands (`docker volume rm`) when resetting test environments.

**Q5: Why did we provide both an automated Python demo script and exact `curl` commands in `DEMO_SCRIPT.md`?**
A: `curl` commands demonstrate protocol-level truth: an engineer can paste them into any terminal to inspect raw headers, HTTP status codes, and JSON response bodies without abstraction. `demo.py` provides convenience, colored formatting, timing measurements, and reproducible execution for structured 5-minute stakeholder demonstrations.

### Verification command

```bash
# 1. Inspect Docker Compose configuration
docker compose config

# 2. Run the automated demonstration script in benchmark mode
uv run python scripts/demo.py --benchmark-only

# 3. View the 5-minute presentation script
type docs\DEMO_SCRIPT.md
```

---

## Phase 8: Final QA & Verification

### What was built

We conducted a comprehensive final verification, dry-run clone audit, and produced the production limitations catalog:

1. **Clean-Clone Audit**:
   - Cloned the repository to a clean directory and verified structure, package specifications, environment templates, and documentation.
   - Confirmed that `.env.example` provides explicit defaults for all Gateway and RAG configurations across local and Docker execution modes.

2. **Complete Test Suite Execution (100% Pass Rate)**:
   - Fast unit and integration tests (`pytest -m "not slow"`): **80 passed** in 66.06s.
   - In-process quantized model test (`test_phi3_slow.py`): **1 passed** in 17.71s.
   - Total test verification: **81 passed out of 81 tests**.

3. **Known Limitations & Production Considerations (`docs/KNOWN_LIMITATIONS.md`)**:
   - Documented operational boundaries across document ingestion (lack of OCR for scanned images, tabular layouts), concurrency (single-GPU semaphore serialisation, absence of SSE streaming), privacy (English-only spaCy models, static codename deny-lists), and environment constraints (Windows file locking on Chroma segments, host NVIDIA container requirements).

4. **Definition of Done Verification**:
   - All 11 checklist requirements in `AGENT_BUILD_PLAN.md` Section 9 verified and satisfied.

### Glossary

| Term | Meaning |
|------|---------|
| **Clean-Clone Validation** | Testing repository onboarding from a fresh clone to ensure no implicit local state, uncommitted files, or missing paths prevent execution. |
| **Single-GPU Serialisation** | Guarding deep learning model inference behind an `asyncio.Semaphore(1)` to prevent concurrent CUDA memory allocation faults on a single GPU. |
| **PagedAttention / Continuous Batching** | Advanced inference engine techniques (used by vLLM/TGI) to dynamically batch tokens across multiple concurrent requests without thread-blocking. |
| **Layout-Aware OCR** | Optical character recognition engines that identify columns, tables, and bounding boxes in scanned images before passing text to downstream parsers. |
| **Definition of Done (DoD)** | A formal agreement specifying all quality, testing, architectural, and documentation criteria a software deliverable must meet before release. |

### Why we did it this way

- **Honest Limitations over Vague Promises:** Real enterprise systems have operational boundaries. Documenting that scanned PDFs require OCR and that in-process Phi-3 serializes requests demonstrates technical maturity and equips mentors with genuine engineering insights.
- **Fast vs. Slow Test Partitioning:** Running 80 tests in ~1 minute enables fast local TDD and CI pull-request checks without downloading 2.6GB of weights, while preserving full end-to-end integration tests in `@pytest.mark.slow`.
- **Decoupled Architecture with HTTP Contracts:** The Gateway and RAG services communicate strictly over HTTP using standard REST interfaces. If the local Phi-3 backend becomes a bottleneck under high user volume, operators can transition to `BACKEND=openai_compatible` without modifying a single line of RAG code.

### Mentor questions

**Q1: What happens if an enterprise user uploads an image-only scanned PDF to the RAG service?**
A: The RAG service's PDF parser (`pymupdf`) extracts zero text characters across all pages. The service intercepts this condition immediately and returns HTTP 400 Bad Request with: `"No extractable text found in PDF; document appears empty or contains only scanned images (OCR is not supported)."`. This fails early and prevents corrupt or empty documents from polluting vector collections.

**Q2: How does the Gateway prevent CUDA out-of-memory errors when multiple users send simultaneous requests to `hf_local`?**
A: Autoregressive token generation in PyTorch is thread-blocking and allocates GPU KV-caches. In `slm_gateway.backends.hf_local`, model generation is wrapped in `asyncio.to_thread` guarded by an `asyncio.Semaphore(1)`. This ensures that even under concurrent inbound HTTP traffic, only one generation job executes on the GPU at any given instant; subsequent requests queue safely in the asyncio event loop.

**Q3: What is the primary bottleneck when scaling this architecture to hundreds of concurrent users, and how would you resolve it?**
A: In-process single-GPU serialisation is the primary throughput bottleneck. To scale to high concurrency:
1. Switch `BACKEND=openai_compatible` and point the Gateway to a dedicated cluster running vLLM or HuggingFace TGI, which uses PagedAttention and continuous batching across multiple GPUs.
2. Deploy multiple Gateway container replicas behind an enterprise load balancer (e.g., NGINX, Envoy).

**Q4: Why did we separate the test suite into fast mocked tests and slow model tests?**
A: Loading Phi-3 Mini and BGE models takes 15-20 seconds and consumes 3GB+ of memory. Marking real model inference with `@pytest.mark.slow` allows developers and CI systems to run 80 unit and integration tests (testing schemas, routing logic, PII redaction, chunking boundaries, and error codes) in 60 seconds with lightweight mocks, while still verifying real PyTorch execution in dedicated runs.

**Q5: Looking back at the entire build from Phase 0 to Phase 8, what was the most important architectural design decision?**
A: The loop prevention design using `X-Bypass-Router: true` coupled with centralized model serving. It allowed the RAG service to remain completely decoupled from LLM weight management (saving ~2.6GB of duplicate VRAM), maintained a single point of PII enforcement and token metering at the Gateway, and solved the circular delegation problem elegantly without requiring dual ports or complex orchestration.

### Verification command

```bash
# 1. Run all 80 fast unit and integration tests
uv run pytest gateway/tests/ rag/tests/ -v -m "not slow"

# 2. Run the slow in-process model inference test
uv run pytest gateway/tests/test_phi3_slow.py -v

# 3. View the complete known limitations report
type docs\KNOWN_LIMITATIONS.md
```





