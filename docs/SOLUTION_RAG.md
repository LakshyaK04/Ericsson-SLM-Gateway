# Solution Document: RAG Ingestion, Retrieval & Evaluation Engine

This document outlines the architecture, component implementation, empirical benchmarks, and operational design of the **RAG Service** (`rag_service`).

---

## 1. System Overview

The **RAG Service** provides document ingestion, multi-strategy text segmentation, dense vector search, neural cross-encoder re-ranking, and grounded generation for technical documentation.

### Core Capabilities
- **Multi-Format Parsing**: PyMuPDF-based PDF extraction and `python-docx` parser supporting native headings and bullet lists.
- **Selectable Segmentation**: Independent chunking pipelines (`character`, `structure`, `semantic`).
- **Isolated Vector Storage**: Persistent ChromaDB collections segregating chunking strategies to eliminate indexing bias.
- **Two-Stage Retrieval**: Dense search (top-20) via `BAAI/bge-small-en-v1.5`, followed by cross-attention re-ranking (top-3) via `BAAI/bge-reranker-base`.
- **Grounded Synthesis**: Formulates numbered citation context blocks (`[1]`, `[2]`, `[3]`) and dispatches generation requests to the Gateway with loop prevention (`X-Bypass-Router: true`).

---

## 2. API Specification

The RAG service listens by default on `http://localhost:8001`.

### 2.1 `POST /documents`
Uploads and indexes a technical document under one or more chunking strategies.
- **Content-Type**: `multipart/form-data`
- **Form Fields**:
  - `file`: PDF or DOCX binary stream. (Unsupported formats return HTTP 400).
  - `strategies`: Comma-separated list (`character,structure,semantic`).
- **Validation**:
  - Rejects empty files or image-only scanned PDFs with 400 error: *"No extractable text found in PDF; document appears empty or contains only scanned images (OCR is not supported)."*

### 2.2 `GET /documents`
Lists all currently indexed documents across ChromaDB collections.

### 2.3 `DELETE /documents/{doc_id}`
Removes all indexed chunks associated with `doc_id` across all strategy collections.

### 2.4 `POST /query`
Performs two-stage retrieval and re-ranking without text generation.
- **Request Body**:
  ```json
  {
    "query": "Which network function manages user registration and authentication?",
    "strategy": "structure",
    "retrieve_k": 20,
    "final_k": 3,
    "doc_ids": null
  }
  ```
- **Response**: List of chunks with text, source, page, `dense_score`, and `rerank_score`.

### 2.5 `POST /answer`
Performs two-stage retrieval, re-ranking, context augmentation, and calls the Gateway `/v1/chat/completions` endpoint for final grounded answer synthesis.
- **Response**: Synthesized answer with citations, source chunk provenance, and token usage.

### 2.6 `GET /health`
Liveness probe returning `{"status": "ok"}`.

---

## 3. Chunking Strategies & Design Rationale

| Strategy | Algorithm | Boundary Mechanics | Trade-Offs |
|---|---|---|---|
| **`character`** | Sliding window | Fixed length (500 chars, 50 overlap), snapped backward to nearest whitespace. | Predictable size, simple; can cut across logical sentences. |
| **`structure`** | Heading & paragraph aware | Splits on Markdown `#`, numbered `1.1`, uppercase headers, and `\n\n`. Merges small units up to 1000 chars. | **Highest retrieval accuracy in this benchmark (100% Hit@1)**. Preserves cohesive document sections. |
| **`semantic`** | Sentence similarity dips | Splits into sentences, embeds with BGE-small, cuts when cosine similarity between adjacent sentences drops below threshold. | Conceptually unified fragments; higher ingestion latency due to per-sentence embeddings. |

---

## 4. Empirical Evaluation Results

Evaluated on a small synthetic benchmark (3 PDFs / 5 pages from `scripts/create_eval_docs.py` and 36 questions):
- `enterprise_rag_sample.pdf` (Enterprise AI Platform)
- `5g_core_architecture.pdf` (5G SBA, AMF, SMF, UPF, Slicing)
- `cloud_native_telecom_infrastructure.pdf` (CNFs, SR-IOV, DPDK, Multus CNI, ZTA)

Evaluated via `eval/chunking_eval.py` comparing **Dense-Only** vs. **Two-Stage Re-Ranking**:

| Strategy | Re-ranker | Total Chunks | Avg Length (chars) | Hit@1 (%) | Hit@3 (%) | MRR | Latency (ms) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`character`** | **Off** | 16 | 423.6 | 86.11% | 100.00% | 0.9213 | 16.9 |
| **`character`** | **On** | 16 | 423.6 | 88.89% | 97.22% | 0.9306 | 2340.3 |
| **`structure`** | **Off** | 10 | 623.6 | 94.44% | 100.00% | 0.9722 | 14.7 |
| **`structure`** | **On** | **10** | **623.6** | **100.00%** | **100.00%** | **1.0000** | **5634.3** |
| **`semantic`** | **Off** | 16 | 388.7 | 88.89% | 100.00% | 0.9352 | 114.9 |
| **`semantic`** | **On** | 16 | 388.7 | **94.44%** | **97.22%** | **0.9583** | **4183.3** |

### Benchmark Takeaways
1. **Structure Chunking Accuracy**: Structure chunking achieved 100% Hit@1 with the re-ranker. Technical documents organized around clear headings benefit when sections are kept whole.
2. **Selective Re-Ranking Gain**: The cross-encoder improved Hit@1 on `structure` (+5.56%) and `semantic` (+5.55%).
3. **Measured Latency Cost**: Pure dense lookup takes ~15-20ms, while neural cross-encoder re-ranking adds significant CPU inference overhead without GPU acceleration.

---

## 5. Grounded Prompt Formulation & LLM Generation

In `rag_service.generation`, retrieved top-3 chunks are assembled into a structured prompt:

```text
Context from uploaded documents:
[1] Document: 5g_core_architecture.pdf (Page 1)
The Access and Mobility Management Function (AMF) handles connection and mobility tasks...

[2] Document: 5g_core_architecture.pdf (Page 1)
The Session Management Function (SMF) allocates IP addresses and controls UPF data paths...

Query: What does the AMF do in 5G Core?
```

### Strict System Prompt:
```text
You are a factual assistant. Answer the user question strictly using only the provided context.
Cite sources using bracketed numbers like [1], [2].
If the context does not contain the answer, state clearly: 'The provided documents do not contain enough information to answer this question.'
Do not speculate or extrapolate beyond the text.
```

The generation request is dispatched to the Gateway:
```http
POST http://gateway:8000/v1/chat/completions
X-Bypass-Router: true
```

---

## 6. Limitations

1. **Scanned & Image-Only PDFs**: PyMuPDF extracts text directly from the digital PDF text layer. Scanned pages or raster screenshots contain zero extractable text and are rejected with HTTP 400 (OCR is not integrated).
2. **Complex Multi-Column / Tabular Layouts**: Multi-column text flow and borderless tables may interleave text blocks when extracted sequentially, requiring table-aware parsers for strict row-column formatting.
3. **Small Synthetic Evaluation Set**: The benchmark corpus consists of 3 PDFs totaling 5 pages and 36 questions generated via `scripts/create_eval_docs.py`. Real-world enterprise corpora are substantially larger and messier.
4. **Re-Ranking Overhead on Fixed Chunks**: Re-ranking did not improve character chunking on Hit@1 and adds ~140-155ms cross-encoder inference latency.
5. **ChromaDB File-Locking on Windows**: Fast consecutive test runs on Windows can encounter file locks on ChromaDB's persistent storage; tests isolate storage per run to avoid conflicts.
