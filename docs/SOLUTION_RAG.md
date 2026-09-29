# Solution Document: Hybrid RAG Ingestion, Retrieval & Evaluation Engine

This document outlines the architecture, component implementation, empirical benchmarks, and operational design of the **Hybrid RAG Service** (`rag_service`).

---

## 1. System Overview

The **Hybrid RAG Service** provides document ingestion, multi-strategy text segmentation, dense vector search, neural cross-encoder re-ranking, and grounded generation for technical documentation.

### Core Capabilities
- **Multi-Format Parsing**: PyMuPDF-based PDF extraction and `python-docx` parser supporting native headings and bullet lists.
- **Selectable Segmentation**: Independent chunking pipelines (`character`, `structure`, `semantic`).
- **Isolated Vector Storage**: Persistent ChromaDB collections segregating chunking strategies to eliminate indexing bias.
- **Two-Stage Retrieval**: High-recall dense search (top-20) via `BAAI/bge-small-en-v1.5`, followed by high-precision cross-attention re-ranking (top-3) via `BAAI/bge-reranker-base`.
- **Grounded Synthesis**: Formulates numbered citation context blocks (`[1]`, `[2]`, `[3]`) and dispatches generation requests to the Gateway with loop prevention (`X-Bypass-Router: true`).

---

## 2. API Specification

The RAG service listens by default on `http://localhost:8001`.

### 2.1 `POST /documents`
Uploads and indexes a technical document under one or more chunking strategies.
- **Content-Type**: `multipart/form-data`
- **Form Fields**:
  - `file`: PDF or DOCX binary stream. (Unsupported formats return HTTP 400).
  - `strategies`: Comma-separated list of strategies (`character`, `structure`, `semantic`). Defaults to `character,structure,semantic`.
- **Validation**:
  - Rejects empty files or image-only scanned PDFs with clear 400 error: *"No extractable text found in PDF; document appears empty or contains only scanned images (OCR is not supported)."*
- **Response**:
  ```json
  {
    "doc_id": "9f32b8aa0c44439c",
    "filename": "ericsson_5g_core.pdf",
    "total_chunks": 16,
    "chunks_per_strategy": {
      "character": 6,
      "structure": 4,
      "semantic": 6
    }
  }
  ```

### 2.2 `GET /documents`
Lists all currently indexed documents across ChromaDB collections.
- **Response**:
  ```json
  [
    {
      "doc_id": "9f32b8aa0c44439c",
      "filename": "ericsson_5g_core.pdf",
      "total_chunks": 16,
      "strategies": ["character", "structure", "semantic"]
    }
  ]
  ```

### 2.3 `DELETE /documents/{doc_id}`
Removes all indexed chunks associated with `doc_id` across all strategy collections.

### 2.4 `POST /query`
Performs two-stage retrieval and re-ranking without text generation.
- **Request**:
  ```json
  {
    "query": "Which network function manages user registration and authentication?",
    "strategy": "structure",
    "retrieve_k": 20,
    "final_k": 3,
    "doc_ids": null
  }
  ```
- **Response**: Returns a list of chunks enriched with source metadata and both retrieval scores:
  ```json
  [
    {
      "chunk_id": "9f32b8aa_struct_p1_1",
      "text": "The Access and Mobility Management Function (AMF) handles connection...",
      "source": "ericsson_5g_core.pdf",
      "page": 1,
      "strategy": "structure",
      "dense_score": 0.5412,
      "rerank_score": 0.8932
    }
  ]
  ```

### 2.5 `POST /answer`
Performs two-stage retrieval, re-ranking, context augmentation, and calls the Gateway `/v1/chat/completions` endpoint for final answer synthesis.
- **Request**: Same body as `POST /query`.
- **Response**:
  ```json
  {
    "answer": "Based on the provided documentation [1], the Access and Mobility Management Function (AMF) manages connection and mobility tasks...",
    "sources": [
      {
        "chunk_id": "9f32b8aa_struct_p1_1",
        "source": "ericsson_5g_core.pdf",
        "page": 1,
        "dense_score": 0.5412,
        "rerank_score": 0.8932
      }
    ],
    "usage": {
      "prompt_tokens": 312,
      "completion_tokens": 48,
      "total_tokens": 360
    }
  }
  ```

### 2.6 `GET /health`
Liveness probe returning `{"status": "ok"}`.

---

## 3. Chunking Strategies & Design Rationale

| Strategy | Algorithm | Boundary Mechanics | Trade-Offs |
|---|---|---|---|
| **`character`** | Sliding window | Fixed length (500 chars, 50 overlap), snapped backward to nearest whitespace. | High throughput, predictable size; but can slice through logical paragraphs and table rows. |
| **`structure`** | Heading & paragraph aware | Splits on Markdown `#`, numbered `1.1`, uppercase headers, and `\n\n`. Merges small units up to 1000 chars. | **Highest retrieval precision (100% Hit@1)**. Preserves complete architectural descriptions. |
| **`semantic`** | Sentence similarity dips | Splits into sentences, embeds with BGE-small, and cuts when cosine similarity between adjacent sentences drops below threshold. | Conceptually unified fragments; higher ingestion latency due to per-sentence vector calculations. |

---

## 4. Empirical Evaluation Results (Phase 5 Benchmark)

Evaluated across 36 ground-truth technical questions on a 3-document corpus (`eval/docs/`):
- `ericsson_rag_sample.pdf` (Ericsson AI Platform)
- `ericsson_5g_core_architecture.pdf` (5G SBA, AMF, SMF, UPF, Slicing)
- `cloud_native_telecom_infrastructure.pdf` (CNFs, SR-IOV, DPDK, Multus CNI, ZTA)

Evaluated via `eval/chunking_eval.py` comparing **Dense-Only** vs. **Two-Stage Re-Ranking**:

| Strategy | Re-ranker | Total Chunks | Avg Length (chars) | Hit@1 (%) | Hit@3 (%) | MRR | Latency (ms) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`character`** | **Off** | 16 | 422.1 | 86.11% | 97.22% | 0.9028 | 11.6 |
| **`character`** | **On** | 16 | 422.1 | 83.33% | 97.22% | 0.9028 | 152.2 |
| **`structure`** | **Off** | 10 | 621.0 | 86.11% | 94.44% | 0.9028 | 9.8 |
| **`structure`** | **On** | **10** | **621.0** | **100.00%** | **100.00%** | **1.0000** | **159.3** |
| **`semantic`** | **Off** | 16 | 387.2 | 80.56% | 94.44% | 0.8611 | 10.1 |
| **`semantic`** | **On** | 16 | 387.2 | **94.44%** | **97.22%** | **0.9583** | 166.9 |

### Key Benchmark Takeaways
1. **Dominance of `structure` Chunking**: Structure chunking achieved a perfect **100.00% Hit@1 and MRR 1.0000** with the neural re-ranker. Technical documentation naturally follows hierarchical section outlines; respecting these boundaries preserves complete functional contexts.
2. **Re-Ranking Uplift**: The cross-encoder provided a massive uplift for semantically coherent chunks:
   - `structure`: Hit@1 increased from 86.11% to 100.00% (+13.89%).
   - `semantic`: Hit@1 increased from 80.56% to 94.44% (+13.88%).
3. **Character Chunking Limitation**: The cross-encoder failed to improve `character` chunking (83.33% vs 86.11%) because when clauses are severed mid-sentence, cross-attention cannot synthesize missing information.
4. **Latency Budget**: Pure dense lookup takes ~10ms. Cross-encoder re-ranking adds ~145ms overhead. In return, ranking accuracy improves dramatically, fitting comfortably within typical interactive SLAs (<500ms).

---

## 5. Grounded Prompt Formulation & LLM Generation

In `rag_service.generation`, retrieved top-3 chunks are assembled into a structured prompt:

```text
Context from uploaded documents:
[1] Document: ericsson_5g_core.pdf (Page 1)
The Access and Mobility Management Function (AMF) handles connection and mobility tasks...

[2] Document: ericsson_5g_core.pdf (Page 1)
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

## 6. Known Limitations & Dataset Commentary

1. **Corpus Size**: The empirical benchmark tested 3 technical PDFs totaling 5 pages and 36 questions. While it exercises domain acronyms (AMF, SMF, UPF, DPDK, SR-IOV), enterprise collections often span thousands of pages.
2. **Tabular Data & OCR**: The current pipeline uses PyMuPDF's text layer extraction. Documents containing rasterized screenshots, architectural diagrams without text tags, or multi-column layout tables require OCR / layout-aware vision models.
3. **Single Vector Store Engine**: ChromaDB was chosen for zero-dependency local embedding storage. For distributed multi-node enterprise environments, a transition to Milvus or Qdrant would be indicated.
