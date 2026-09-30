# Core Source Files Study Guide

This guide lists the **exact source files** you should study to confidently explain the project to your mentor. All other files (tests, docker configs, utilities) are secondary infrastructure.

---

## Part 1: SLM Gateway Files (`gateway/src/slm_gateway/`)

### 1. `main.py`
- **Why study it**: It is the entry point and "front door" of the system.
- **Key concepts to look for**:
  - `lifespan`: Where the model, PII redactor, and intent router are loaded once into memory on startup.
  - `@app.post("/v1/chat/completions")`: The main endpoint. Trace how it redacts PII, calls the router, checks for `X-Bypass-Router`, and either delegates to the RAG service or calls local Phi-3.
  - Attaching metadata: How `x_routing`, `x_pii`, and `x_sources` are added to the standard OpenAI response envelope.

### 2. `pii.py`
- **Why study it**: Demonstrates privacy compliance and prompt sanitization.
- **Key concepts to look for**:
  - `PIIRedactor.__init__`: Uses Microsoft Presidio Analyzer and Anonymizer with spaCy's `en_core_web_sm` model.
  - Exclusion list: Why `LOCATION` and `DATE_TIME` are excluded so queries like "What is the capital of Germany?" don't get damaged.
  - `EMPLOYEE_ID` pattern: Custom regex recognizer (`\bEMP-\d{5,7}\b`).
  - `redact()`: How entities are replaced with typed placeholders (`<EMAIL_ADDRESS>`) and redaction counts are tracked.

### 3. `router.py`
- **Why study it**: Explains how queries are routed without hardcoded keywords or an expensive LLM call.
- **Key concepts to look for**:
  - `intents.yaml`: The curated bank of exemplar sentences for `general`, `technical`, and `rag`.
  - `classify()`: Encodes the user query using `BAAI/bge-small-en-v1.5`, computes cosine similarities against all exemplars, and averages the top 3 highest scores for each intent.
  - Operating threshold (`0.55`): If the top intent score is below 0.55, it safely falls back to `general`.

### 4. `backends/hf_local.py`
- **Why study it**: Shows how the open-weights model is served in-process on the GPU.
- **Key concepts to look for**:
  - 4-bit Quantization: `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)`. Reduces memory footprint to ~2.6 GB VRAM.
  - `asyncio.Semaphore(1)`: Serializes generation requests on a single GPU to prevent CUDA Out-of-Memory crashes.
  - Context Truncation: How conversations exceeding 4096 tokens are trimmed while preserving system instructions.

### 5. `schemas.py`
- **Why study it**: Proves drop-in OpenAI API compatibility.
- **Key concepts to look for**:
  - `ChatCompletionRequest`: Validates incoming messages, temperature, top_p, and max_tokens.
  - `ChatCompletionResponse`: Structures the response to match OpenAI's exact JSON shape (`id`, `object`, `created`, `model`, `choices`, `usage`).

---

## Part 2: RAG Service Files (`rag/src/rag_service/`)

### 6. `main.py`
- **Why study it**: Shows the RAG HTTP interface and lifecycle.
- **Key concepts to look for**:
  - `POST /documents`: File upload endpoint that dispatches files to parsers and indexes chunks across ChromaDB collections.
  - `POST /query`: Two-stage retrieval endpoint returning top-3 re-ranked chunks.
  - `POST /answer`: End-to-end grounded generation endpoint that retrieves chunks, formats context, and calls back to the Gateway.

### 7. `parsers/pdf.py` (Representative Parser)
- **Why study it**: Shows clean digital text extraction from documents.
- **Key concepts to look for**:
  - `extract_text_by_page()`: Uses PyMuPDF (`fitz`) to extract text while keeping track of page numbers (1-indexed).
  - Empty/Scanned PDF rejection: If extracted text is empty, raises HTTP 400 informing the user that OCR is not supported.

### 8. `chunking/character.py`
- **Why study it**: The baseline fixed-size chunking strategy.
- **Key concepts to look for**:
  - Sliding window of 500 characters with 50-character overlap.
  - Snaps chunk boundary backwards to the nearest whitespace so words are never cut in half.

### 9. `chunking/structure.py`
- **Why study it**: The highest-performing chunking strategy in our benchmark.
- **Key concepts to look for**:
  - Heading detection: Regex detecting Markdown `#`, numbered sections `1.1`, and uppercase headers.
  - Paragraph grouping: Splits on double-newlines (`\n\n`) and merges short sections up to 1000 characters to keep context cohesive.

### 10. `chunking/semantic.py`
- **Why study it**: The embedding-driven chunking strategy.
- **Key concepts to look for**:
  - Splits text into individual sentences.
  - Embeds sentences with BGE-small and computes cosine similarity between consecutive sentences.
  - Creates a new chunk boundary whenever similarity drops below the threshold (default 0.65).

### 11. `embeddings.py`
- **Why study it**: Shows how text is mapped into vector space.
- **Key concepts to look for**:
  - Loads `BAAI/bge-small-en-v1.5` (384 dimensions, L2-normalized).
  - Asymmetric instruction prefixing: Prepends `"Represent this sentence for searching relevant passages: "` to queries, but embeds document chunks without prefix.

### 12. `retriever.py`
- **Why study it**: The heart of the two-stage retrieval pipeline.
- **Key concepts to look for**:
  - Stage 1 (Dense Search): Queries ChromaDB for `retrieve_k=20` candidate chunks using vector similarity.
  - Stage 2 (Cross-Encoder Re-Ranking): Calls `reranker.py` to evaluate the 20 candidates and select the top `final_k=3`.
  - Attaches both `dense_score` and `rerank_score` to each returned chunk.

### 13. `reranker.py`
- **Why study it**: Explains why re-ranking improves retrieval precision.
- **Key concepts to look for**:
  - Uses `BAAI/bge-reranker-base` cross-encoder.
  - Feeds `(query, passage)` pairs together into full transformer cross-attention rather than comparing isolated vectors.

### 14. `generation.py`
- **Why study it**: Shows how context is formatted and grounded answers are synthesized.
- **Key concepts to look for**:
  - `build_grounded_prompt()`: Formats chunks into numbered blocks (`[1] Document: doc.pdf (Page 1)\nText...`).
  - Strict system prompt: Instructs the model to answer *only* from provided context and cite bracketed numbers.
  - `X-Bypass-Router: true`: Passed in the HTTP header when calling Gateway `/v1/chat/completions` to prevent an infinite routing loop.
