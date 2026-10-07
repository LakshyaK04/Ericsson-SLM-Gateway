# Empirical Retrieval & Re-Ranking Benchmark Report

- **Benchmark Run Date:** 2026-10-05 23:33:13
- **Dataset:** SQuAD v2.0 Public Slice (500 passages, 150 queries)
- **Dense Bi-Encoder:** `BAAI/bge-small-en-v1.5` (384-dim, normalized)
- **Sparse Lexical Engine:** Okapi BM25 ($k_1=1.5, b=0.75$)
- **Neural Cross-Encoder:** `BAAI/bge-reranker-base`
- **Hardware Platform:** Windows 10 (v10.0.26300)
- **CPU:** Intel64 Family 6 Model 183 Stepping 1, GenuineIntel
- **GPU / Device:** NVIDIA GeForce RTX 3050 6GB Laptop GPU (`cuda`)
- **PyTorch Version:** 2.13.0+cu130

---

## 1. Retrieval Performance Matrix

| Configuration | Re-Ranker | RRF $k$ | Dense / Sparse Weights | Hit@1 (%) | Hit@1 (95% CI) | Hit@3 (%) | Hit@10 (%) | MRR | MRR (95% CI) | nDCG@10 | Mean Latency (ms) | P95 Latency (ms) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **BM25 Sparse Lexical Only** | Off | 60 | N/A | **86.0%** | [80.0%, 91.3%] | 94.67% | 97.33% | **0.906** | [0.8647, 0.9433] | **0.923** | 7.07 ms | 10.96 ms |
| **BGE Dense Vector Only** | Off | 60 | N/A | **80.67%** | [74.0%, 86.7%] | 91.33% | 96.0% | **0.8643** | [0.8168, 0.9091] | **0.8879** | 101.29 ms | 179.81 ms |
| **Hybrid RRF (k=60)** | Off | 60 | 1.0 / 1.0 | **86.0%** | [80.0%, 91.3%] | 94.0% | 99.33% | **0.9092** | [0.8708, 0.9443] | **0.93** | 107.28 ms | 180.67 ms |
| **Hybrid RRF (k=20)** | Off | 20 | 1.0 / 1.0 | **86.0%** | [80.0%, 91.3%] | 94.67% | 99.33% | **0.9103** | [0.8724, 0.9449] | **0.9309** | 98.72 ms | 153.37 ms |
| **Hybrid RRF (k=100)** | Off | 100 | 1.0 / 1.0 | **86.0%** | [80.0%, 91.3%] | 93.33% | 99.33% | **0.9086** | [0.8692, 0.9446] | **0.9294** | 102.01 ms | 156.34 ms |
| **Hybrid Weighted RRF (Dense 0.7 / Sparse 0.3)** | Off | 60 | 0.7 / 0.3 | **85.33%** | [79.3%, 90.7%] | 94.67% | 98.0% | **0.9023** | [0.8608, 0.9401] | **0.9216** | 101.75 ms | 134.33 ms |
| **Hybrid Weighted RRF (Dense 0.3 / Sparse 0.7)** | Off | 60 | 0.3 / 0.7 | **85.33%** | [79.3%, 90.7%] | 95.33% | 98.67% | **0.905** | [0.8648, 0.9416] | **0.9253** | 106.77 ms | 191.49 ms |
| **Hybrid + Cross-Encoder Re-Ranking** | On (bge-reranker-base) | 60 | 1.0 / 1.0 | **74.0%** | [66.7%, 80.7%] | 87.33% | 98.67% | **0.8242** | [0.7743, 0.8728] | **0.8638** | 617.17 ms | 847.76 ms |

---

## 2. In-Depth Empirical Analysis & Honest Insights

### 2.1 Dense vs. Sparse Modality Comparison
- **BGE Dense Vector Search** achieved **80.67% Hit@1** (95% CI: [74.0%, 86.7%]) and **0.8643 MRR**, compared to **86.0% Hit@1** (95% CI: [80.0%, 91.3%]) and **0.906 MRR** for **BM25 Sparse Search**.
- Dense embeddings excel on conceptual questions where queries do not share verbatim tokens with passages, whereas BM25 performs strongly on exact keyword, entity name, and numeric constraints.

### 2.2 Impact of Reciprocal Rank Fusion (RRF) & Parameter Sensitivity
- Standard Hybrid RRF ($k=60$) achieved **86.0% Hit@1** (95% CI: [80.0%, 91.3%]) and **0.9092 MRR**.
- **Parameter Sensitivity ($k=20, 60, 100$) and Weighting**: Hit@1 remained flat at 86.0% across $k$ variations, while weighted variants scored 85.33% — a difference of exactly 1 query out of 150 (128 vs 129 queries). Because the bootstrap 95% confidence intervals overlap completely (~80% to ~91%), differences of 1–3 queries out of 150 represent expected statistical sampling noise rather than systematic algorithmic divergence.

### 2.3 Cross-Encoder Re-Ranking Tradeoff
- Hybrid + Cross-Encoder Re-Ranking (`bge-reranker-base`) achieved **74.0% Hit@1** (95% CI: [66.7%, 80.7%]), **87.33% Hit@3**, and **0.8242 MRR** with **nDCG@10 of 0.8638**.
- **Observed Regression**: On this SQuAD slice, adding `bge-reranker-base` lowered Hit@1 by 12.00 pp (from 86.0% to 74.0%) and added neural cross-attention latency (617.17 ms mean vs 107.28 ms).
- **Empirical Root Cause (Tested)**: Diagnostic analysis (`eval/results/rerank_diagnostics.md`) verified that the pipeline is bug-free (zero alignment or indexing faults). Instead, 79.3% of demotions were caused by same-article neighbor passages scoring higher on broad topical overlap than the specific passage containing the short answer span. Furthermore, testing an alternative cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) on the identical candidate pool raised Hit@1 to **94.67%**, confirming that cross-encoder precision on factoid QA is highly sensitive to model pre-training domain alignment.

### 2.4 How to Read These Results
- **Sample Size & Single-Run Nature**: Evaluated on an empirical 500-passage, 150-query slice of SQuAD v2.0. Benchmark metrics reflect a single deterministic evaluation run.
- **Bootstrap 95% Confidence Intervals**: Reported confidence intervals (10,000 resamples) illustrate the margin of uncertainty. Differences of ≤2% on a 150-query test set represent variations of 1–3 queries and fall within sampling noise.
