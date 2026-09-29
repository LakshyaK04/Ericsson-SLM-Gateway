# Known System Limitations & Production Considerations

This document provides a transparent, engineering-grade account of known constraints, design trade-offs, and boundary conditions within the **Ericsson Local GenAI Stack**.

---

## 1. Document Ingestion & Parsing Limitations

### 1.1 Scanned or Image-Only Documents (Lack of OCR)
- **Constraint**: The ingestion pipeline utilizes PyMuPDF (`fitz`) and `python-docx` to extract text from the digital layout stream.
- **Impact**: Uploaded PDFs containing scanned pages, photocopied receipts, or raster screenshots contain zero extractable digital text.
- **Handling**: The system detects this condition immediately upon upload and returns HTTP 400 with the message:
  ```text
  No extractable text found in PDF; document appears empty or contains only scanned images (OCR is not supported).
  ```
- **Remediation Path**: For enterprise deployments requiring scanned document processing, integrate an OCR engine (such as Tesseract or paddleOCR) or an optical document understanding vision model (such as Donut or Nougat) before the text parsing stage.

### 1.2 Multi-Column & Tabular Layouts
- **Constraint**: Complex multi-column layouts and borderless tables in PDFs can occasionally result in interleaved reading order when extracted as raw sequential text blocks.
- **Impact**: While `structure` chunking reliably isolates major sections and paragraphs, fine-grained tabular relationships (e.g., column-to-row alignments across pages) may lose strict tabular formatting.
- **Remediation Path**: Implement a table-aware parser (such as `pdfplumber` or `layoutparser`) that serializes structured tables into Markdown or HTML table representations before passing to chunkers.

---

## 2. In-Process Model Serving & Concurrency

### 2.1 Single-GPU Semaphore Serialisation
- **Constraint**: The default production backend (`hf_local`) loads `microsoft/Phi-3-mini-4k-instruct` in 4-bit NormalFloat (NF4) directly in-process using PyTorch. In-process PyTorch autoregressive generation blocks the execution thread.
- **Impact**: Generation calls are executed behind an `asyncio.Semaphore(1)` to safeguard against CUDA memory conflicts and out-of-memory errors on a single GPU.
- **Behavior Under Load**: Concurrent chat completion requests will queue up sequentially. Under heavy traffic, client requests will experience latency proportional to the queue depth.
- **Remediation Path**:
  - Horizontal scaling: Deploy multiple Gateway replicas behind a round-robin load balancer.
  - Dedicated inference engines: Switch `BACKEND=openai_compatible` to route requests to an external high-throughput inference engine (e.g., vLLM or Hugging Face TGI) utilizing continuous batching and PagedAttention.

### 2.2 Lack of Server-Sent Events (SSE) Streaming
- **Constraint**: The `/v1/chat/completions` endpoint currently generates complete responses before returning (`stream=false`).
- **Impact**: Setting `stream=true` returns HTTP 400 Bad Request. For lengthy completions, time-to-first-token (TTFT) is equal to total generation latency.
- **Remediation Path**: Implement an asynchronous HuggingFace `TextIteratorStreamer` yielding SSE data chunks in standard OpenAI format (`data: {"choices": [{"delta": {"content": "..."}}]}`).

---

## 3. Privacy & PII Redaction Constraints

### 3.1 Language Specificity (English Only)
- **Constraint**: The bundled Presidio pipeline and spaCy model (`en_core_web_sm`) are trained strictly on English vocabulary, grammar, and naming patterns.
- **Impact**: Prompts in other languages (e.g., Swedish, German, French) may not trigger `PERSON` or custom entity recognizers reliably.
- **Remediation Path**: Load multilingual spaCy models (such as `xx_ent_wiki_sm`) or multi-engine Presidio pipelines for cross-border deployments.

### 3.2 Deny-List Maintenance for Project Codenames
- **Constraint**: The custom enterprise `PROJECT_CODENAME` recognizer operates via a configured string list (`PROJECT_CODENAMES`).
- **Impact**: New project codenames or confidential initiative titles are not masked unless explicitly updated in configuration or environment variables.

---

## 4. Semantic Intent Router Limitations

### 4.1 Exemplar Bank Coverage
- **Constraint**: The intent router uses 112 curated exemplar sentences across 4 classes (`general`, `technical`, `structured_json`, `rag`) mapped via `BAAI/bge-small-en-v1.5` embeddings.
- **Impact**: Highly atypical, novel, or colloquial queries far removed from the exemplars may yield cosine similarity scores below the operating threshold (`0.55`), resulting in fallback to `general`.
- **Remediation Path**: Continuously expand `intents.yaml` with synthetic queries generated from domain query logs or fine-tune a lightweight SetFit classification head.

---

## 5. Storage & Environment Constraints

### 5.1 Windows File-Locking on ChromaDB
- **Constraint**: ChromaDB's underlying Rust HNSW implementation creates memory-mapped segments on disk. On Windows systems, active file locks can conflict if multiple test runners or processes attempt rapid sequential directory recreation in the same workspace folder.
- **Mitigation**: Unit and integration tests must create fresh temporary directories per run or isolate collection instances cleanly.

### 5.2 Containerized GPU Passthrough
- **Constraint**: Docker containers cannot access host GPU hardware unless the host machine has the NVIDIA Container Toolkit (`nvidia-container-toolkit`) installed and configured.
- **Handling**: The codebase transparently provides `deploy.resources.reservations.devices` configurations in `docker-compose.yml` and documents exact steps for host execution, while supporting CPU fallback via `BACKEND=openai_compatible`.
