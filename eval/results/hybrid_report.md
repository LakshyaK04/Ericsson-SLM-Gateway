# Hybrid Retrieval & Reciprocal Rank Fusion (RRF) Benchmark Report

**Date:** 2026-10-04 19:35:27
**Dense Model:** `BAAI/bge-small-en-v1.5`
**Sparse Model:** Okapi BM25 ($k_1=1.5, b=0.75$)
**Cross-Encoder:** `BAAI/bge-reranker-base`
**Evaluation Corpus:** 4 technical documents (16 structure chunks)
**Test Set:** `eval/datasets/hybrid_eval.jsonl` (24 queries: 12 keyword/acronym + 12 conceptual)

---

## 1. Executive Summary

This empirical evaluation measures the performance gains of combining lexical BM25 sparse search with dense semantic bi-encoders via Reciprocal Rank Fusion (RRF) compared against individual retrieval modalities.

### Benchmark Matrix

| Configuration | Re-Ranker | RRF $k$ | Overall Hit@1 | Overall Hit@3 | MRR | Keyword Hit@1 | Conceptual Hit@1 | Latency (ms) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **BM25 Sparse Lexical Only** | Off | 60 | 70.83% | 75.0% | 0.7292 | 100.0% | 41.7% | 0.17 |
| **BGE Dense Vector Only** | Off | 60 | 62.5% | 75.0% | 0.6806 | 91.7% | 33.3% | 17.35 |
| **Hybrid (BM25 + Dense RRF k=60)** | Off | 60 | 66.67% | 75.0% | 0.7083 | 91.7% | 41.7% | 17.59 |
| **Hybrid + Cross-Encoder Re-Ranking** | On | 60 | 62.5% | 79.17% | 0.7014 | 100.0% | 25.0% | 1735.37 |
| **Hybrid RRF (k=20)** | Off | 20 | 66.67% | 75.0% | 0.7083 | 91.7% | 41.7% | 15.67 |
| **Hybrid RRF (k=100)** | Off | 100 | 66.67% | 75.0% | 0.7083 | 91.7% | 41.7% | 16.05 |

---

## 2. Key Insights & Empirical Findings

### 2.1 Modality Comparison on the Evaluation Set
- On this 24-query set, BM25 alone matched or beat the hybrid configurations on Hit@1 (70.83% vs 66.67%) and MRR (0.7292 vs 0.7083).
- BM25 alone achieved 100.0% Hit@1 on keyword queries and 41.7% on conceptual queries.
- Dense vector search achieved 91.7% Hit@1 on keyword queries and 33.3% on conceptual queries.

### 2.2 Re-Ranking Impact
- Adding the `bge-reranker-base` cross-encoder raised Hit@3 by one query (75.0% to 79.17%), but lowered conceptual Hit@1 (41.7% to 25.0%) and overall Hit@1 (66.67% to 62.5%).
- With 24 queries, each query represents approximately 4.17 pp, so observed differences reflect shifts of only 1-2 queries.

### 2.3 RRF Parameter Sensitivity & Test Set Limitations
- Varying the smoothing parameter $k$ across 20, 60, and 100 resulted in identical retrieval metrics (66.67% Hit@1, 75.0% Hit@3, 0.7083 MRR) on this 16-chunk corpus.
- Because the evaluation set is small (24 queries over 4 documents), these findings reflect behavior on this specific sample rather than generalized statistical superiority.
