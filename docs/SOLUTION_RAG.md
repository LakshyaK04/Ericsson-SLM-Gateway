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
  - Rejects empty files or image-only scanned PDFs (when Tesseract is unavailable or yields no text) with 400 error: *"No extractable text found in PDF. The document appears empty or scanned, and OCR is unavailable or found no text (install Tesseract to enable OCR for scanned pages)."*

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

### 4.1 Large-Scale Public Retrieval Benchmark (SQuAD v2.0 Slice)
To evaluate retrieval performance beyond small author-written sets, we benchmarked our retrieval pipeline against a slice of the public **Stanford Question Answering Dataset (SQuAD v2.0)** consisting of **500 passages** and **150 realistic queries** (`eval/datasets/squad_retrieval_corpus.jsonl` and `squad_retrieval_queries.jsonl`).

**Execution & Hardware Environment:**
- **Date:** 2026-10-05 23:33:13
- **Script:** `eval/benchmark_retrieval.py`
- **Dense Model:** `BAAI/bge-small-en-v1.5` (384-dim, normalized)
- **Sparse Engine:** Okapi BM25 ($k_1=1.5, b=0.75$)
- **Neural Cross-Encoder:** `BAAI/bge-reranker-base`
- **Hardware:** Intel Core (Family 6 Model 183), NVIDIA GeForce RTX 3050 6GB Laptop GPU (`cuda`), PyTorch 2.13.0+cu130.

| Configuration | Re-Ranker | RRF $k$ | Dense / Sparse Weights | Hit@1 (%) | Hit@3 (%) | Hit@10 (%) | MRR | nDCG@10 | Mean Latency (ms) | P95 Latency (ms) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **BM25 Sparse Lexical Only** | Off | — | N/A | **86.00%** | 94.67% | 97.33% | 0.9060 | 0.9230 | **7.07 ms** | 10.96 ms |
| **BGE Dense Vector Only** | Off | — | N/A | 80.67% | 91.33% | 96.00% | 0.8643 | 0.8879 | 101.29 ms | 179.81 ms |
| **Hybrid RRF (k=20)** | Off | 20 | 1.0 / 1.0 | **86.00%** | 94.67% | **99.33%** | **0.9103** | **0.9309** | 98.72 ms | 153.37 ms |
| **Hybrid RRF (k=60)** | Off | 60 | 1.0 / 1.0 | **86.00%** | 94.00% | **99.33%** | 0.9092 | 0.9300 | 107.28 ms | 180.67 ms |
| **Hybrid RRF (k=100)** | Off | 100 | 1.0 / 1.0 | **86.00%** | 93.33% | **99.33%** | 0.9086 | 0.9294 | 102.01 ms | 156.34 ms |
| **Hybrid Weighted RRF (0.7 / 0.3)** | Off | 60 | 0.7 / 0.3 | 85.33% | 94.67% | 98.00% | 0.9023 | 0.9216 | 101.75 ms | 134.33 ms |
| **Hybrid Weighted RRF (0.3 / 0.7)** | Off | 60 | 0.3 / 0.7 | 85.33% | **95.33%** | 98.67% | 0.9050 | 0.9253 | 106.77 ms | 191.49 ms |
| **Hybrid + Cross-Encoder Re-Ranking** | On | 60 | 1.0 / 1.0 | 74.00% | 87.33% | 98.67% | 0.8242 | 0.8638 | 617.17 ms | 847.76 ms |

### 4.2 Critical Benchmark Insights & Honest Analysis
1. **Where Hybrid Wins (Recall & Overall Rank Quality):**
   - Hybrid RRF achieved **99.33% Hit@10** and the highest overall **MRR (0.9103)** and **nDCG@10 (0.9309)** across the entire benchmark, outperforming pure BM25 (97.33% Hit@10, 0.9060 MRR) and pure Dense (96.00% Hit@10, 0.8643 MRR).
   - In 149 out of 150 queries, the relevant document was captured in the top-10 candidate pool by Hybrid RRF.
2. **Where BM25 Remains Highly Competitive:**
   - BM25 alone matched Hybrid RRF on Hit@1 (86.00%) at a fraction of the computational latency (**7.07 ms** vs. **107.28 ms**).
   - *Why?* Reading comprehension benchmarks like SQuAD contain questions with distinctive named entities (e.g. "Normans", "Herve", "Turing machines"). In inverted lexical indices, queries with high-IDF entities immediately match the target document without vector embedding overhead.
3. **Why Cross-Encoder Re-Ranking Decreased Hit@1 on SQuAD:**
   - Adding `BAAI/bge-reranker-base` dropped Hit@1 from 86.00% to 74.00% while increasing mean latency from 107 ms to 617 ms.
   - *Root Cause Analysis:* `bge-reranker-base` was trained primarily on web search pairs (MS MARCO) where queries are short search strings matched against varied documents. In dense Wikipedia article corpora, multiple adjacent paragraphs share the exact same topic and entities. The cross-encoder can assign higher semantic relevance to a topical but non-gold paragraph than the exact paragraph containing the specific answer span.
   - *Architectural Recommendation:* For latency-sensitive production workloads (<100ms) or corpora where exact keyword/entity matching is critical, pure **Hybrid RRF ($k=20$ or $k=60$)** is the recommended default.

### 4.3 Answer Quality, Faithfulness & Citation Evaluation
Evaluated via `eval/faithfulness_eval.py` across 25 representative scenarios (15 grounded queries, 5 hallucinated adversarial queries, and 5 out-of-domain unanswerable queries):

| Dimension | Metric | Observed Value | Standard | Assessment |
|:---|:---|:---:|:---:|:---|
| **Citation Compliance** | Citation Presence Rate | **100.0%** | $\ge 95\%$ | All grounded answers correctly cite bracketed context blocks `[N]`. |
| **Citation Accuracy** | Citation Precision | **100.0%** | $100\%$ | All citations map to valid in-bounds retrieved context chunks. |
| **Factual Grounding** | Mean Factual Grounding | **69.48%** | $\ge 65\%$ | Factual entities and terms in answers are grounded in reference text. |
| **Hallucination Catch** | Detection Sensitivity | **100.0%** | $100\%$ | Rule-based evaluator flagged 100% of adversarial fabricated claims. |
| **Refusal Integrity** | Unanswerable Refusal Rate | **100.0%** | $100\%$ | Emits exact refusal string on ungrounded/out-of-domain questions. |

### 4.4 Synthetic Smoke Benchmark (Historical Baseline)
Evaluated on a small synthetic benchmark (3 PDFs / 5 pages from `scripts/create_eval_docs.py` and 36 questions):
- `enterprise_rag_sample.pdf` (Enterprise AI Platform)
- `5g_core_architecture.pdf` (5G SBA, AMF, SMF, UPF, Slicing)
- `cloud_native_telecom_infrastructure.pdf` (CNFs, SR-IOV, DPDK, Multus CNI, ZTA)

Evaluated via `eval/chunking_eval.py` comparing **Dense-Only** vs. **Two-Stage Re-Ranking**:

| Strategy | Re-ranker | Total Chunks | Avg Length (chars) | Hit@1 (%) | Hit@3 (%) | MRR | Latency (ms) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`character`** | **Off** | 16 | 423.6 | 86.11% | 100.00% | 0.9213 | 16.0 |
| **`character`** | **On** | 16 | 423.6 | 88.89% | 97.22% | 0.9306 | 1025.1 |
| **`structure`** | **Off** | 10 | 623.6 | 94.44% | 100.00% | 0.9722 | 15.5 |
| **`structure`** | **On** | **10** | **623.6** | **100.00%** | **100.00%** | **1.0000** | **1069.9** |
| **`semantic`** | **Off** | 16 | 388.7 | 88.89% | 100.00% | 0.9352 | 15.9 |
| **`semantic`** | **On** | 16 | 388.7 | **94.44%** | **97.22%** | **0.9583** | **1175.0** |

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

1. **Scanned & Image-Only PDFs**: Digital text PDFs are parsed directly via PyMuPDF. Scanned pages fall back to Tesseract OCR only when Tesseract is installed on the host (or in the Docker image); otherwise scanned PDFs with no extractable text are rejected with HTTP 400. OCR quality was tested only on synthetic test fixtures and sample slide PDFs with Tesseract 5.x on Windows (mocked in CI); real-world scan accuracy is not benchmarked.
2. **Complex Multi-Column / Tabular Layouts**: Multi-column text flow and borderless tables may interleave text blocks when extracted sequentially, requiring table-aware parsers for strict row-column formatting.
3. **Small Synthetic Evaluation Set**: The benchmark corpus consists of 3 PDFs totaling 5 pages and 36 questions generated via `scripts/create_eval_docs.py`. Real-world corpora are substantially larger and messier.
4. **Re-Ranking Overhead on CPU**: Cross-encoder re-ranking adds ~1,000-1,160 ms inference latency per query on CPU.
5. **ChromaDB File-Locking on Windows**: Fast consecutive test runs on Windows can encounter file locks on ChromaDB's persistent storage; tests isolate storage per run to avoid conflicts.
