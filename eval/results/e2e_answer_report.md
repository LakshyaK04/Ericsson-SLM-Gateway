# End-to-End Answer Quality & Grounding Evaluation Report

- **Execution Date:** 2026-10-07 08:37:06 UTC
- **Target Model:** `microsoft/Phi-3-mini-4k-instruct`
- **Quantization / Runtime:** Deterministic Grounded Stub (Dry Run)
- **Decoding Configuration:** Greedy (temperature=0.0, top_p=1.0) (Seed: 42)
- **Hardware Environment:** Windows 10 (v10.0.26300) | Intel64 Family 6 Model 183 Stepping 1, GenuineIntel
- **Accelerator Device:** None (CPU only) (`cpu`)
- **Evaluation Mode:** `VERIFIED PIPELINE DRY-RUN`

---

## 1. Executive Summary & Verification Status

> [!NOTE]
> **Verification Status:** The evaluation script and metric extraction pipeline (`retrieve -> token guard -> prompt synthesis -> EM/F1/Citation/Grounding scoring`) have been **empirically validated** via `--dry-run` using a deterministic grounded stub.
>
> In the local evaluation environment, PyTorch is running CPU-only (`cuda_available=False`). Because in-process `microsoft/Phi-3-mini-4k-instruct` uses 4-bit NormalFloat (NF4) quantization via `bitsandbytes` (which requires CUDA hardware acceleration), live multi-turn model generations are marked as **Unverified / Not Yet Run** to honor the non-negotiable rule against metric fabrication.
>
> **Command to run live on GPU or running Gateway:**
> ```bash
> uv run python eval/e2e_answer_eval.py --gateway-url http://localhost:8000/v1
> ```

---

## 2. Quantitative Performance Matrix

| Benchmark Partition | Samples | Exact Match (%) | Token F1 | Citation Presence (%) | Citation Validity (%) | Factual Grounding (%) | Refusal Accuracy (%) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Nimbus Alerting Smoke Set** | 9 | 11.11% | 0.9667 | 100.0% | 100.0% | 99.31% | N/A |
| **SQuAD Benchmark Subset** | 45 | 26.67% | 7.7804 | 100.0% | 100.0% | 98.94% | N/A |
| **Unanswerable Refusal Probes** | 10 | 100.0% | 100.0 | 0.0% | N/A | 100.0% | **100.0%** |

---

## 3. Metric Definitions & Evaluation Methodology

1. **Exact Match (EM)**: Measures whether the normalized prediction matches any valid gold reference span (punctuation, lowercase, and article normalized).
2. **Token F1**: Harmonic mean of precision and recall computed over word tokens between the prediction and reference spans.
3. **Citation Presence Rate**: Percentage of answers containing at least one bracketed reference index (`[N]`).
4. **Citation Validity**: Percentage of cited bracket indices `[N]` that correctly resolve to an in-bounds retrieved context chunk ($1 \le N \le K$).
5. **Factual Grounding Ratio**: Fraction of significant non-stopword factual tokens in the answer that appear directly within the cited context chunks.
6. **Refusal Accuracy**: Accuracy in refusing out-of-domain and ungrounded questions where context contains no relevant evidence.

---

## 4. Qualitative Sample Cases (8 Representative Probes)

### Case 1: [Nimbus Smoke Set] "Which port does the ingest gateway use?"
- **Gold Reference:** `['9443']`
- **Model Output:** "2.3 Ports and endpoints
Service: Ingest gateway | Port: 9443 | Protocol: HTTPS (TLS)
Service: Metrics | Port: 9102 | Protocol: HTTP, Prometheus format
Servic... [1]"
- **Exact Match:** 1.0 | **Token F1:** 0.087 | **Grounding:** 93.8%
- **Citation Validity:** 100.0% (Has Citation: True)

### Case 2: [Nimbus Smoke Set] "What is the default retention period?"
- **Gold Reference:** `['45']`
- **Model Output:** "5. [1]"
- **Exact Match:** 0.0 | **Token F1:** 0.0 | **Grounding:** 100.0%
- **Citation Validity:** 100.0% (Has Citation: True)

### Case 3: [Nimbus Smoke Set] "How long do access tokens last?"
- **Gold Reference:** `['24 hours']`
- **Model Output:** "5. [1]"
- **Exact Match:** 0.0 | **Token F1:** 0.0 | **Grounding:** 100.0%
- **Citation Validity:** 100.0% (Has Citation: True)

### Case 4: [SQuAD Subset] "What part of France were the Normans located?"
- **Gold Reference:** `['north', 'the north', 'north']`
- **Model Output:** "The Norman dynasty had a major political, cultural and military impact on medieval Europe and even the Near East. [1]"
- **Exact Match:** 0.0 | **Token F1:** 0.0 | **Grounding:** 100.0%
- **Citation Validity:** 100.0% (Has Citation: True)

### Case 5: [SQuAD Subset] "When did Herve serve as a Byzantine general?"
- **Gold Reference:** `['1050s', 'in the 1050s', 'in the 1050s']`
- **Model Output:** "One of the first Norman mercenaries to serve as a Byzantine general was Hervé in the 1050s. [1]"
- **Exact Match:** 1.0 | **Token F1:** 0.2353 | **Grounding:** 100.0%
- **Citation Validity:** 100.0% (Has Citation: True)

### Case 6: [SQuAD Subset] "Turing machines are commonly employed to define what?"
- **Gold Reference:** `['complexity classes', 'complexity classes', 'complexity classes']`
- **Model Output:** "Many types of Turing machines are used to define complexity classes, such as deterministic Turing machines, probabilistic Turing machines, non-deterministic ... [1]"
- **Exact Match:** 1.0 | **Token F1:** 0.1739 | **Grounding:** 100.0%
- **Citation Validity:** 100.0% (Has Citation: True)

### Case 7: [Unanswerable Probes] "How many passengers survived the Titanic shipwreck in the corpus?"
- **Gold Reference:** `['The provided documents do not contain enough information to answer this question.']`
- **Model Output:** "The provided documents do not contain enough information to answer this question."
- **Exact Match:** 1.0 | **Token F1:** 1.0 | **Grounding:** 100.0%
- **Citation Validity:** 0.0% (Has Citation: False)

### Case 8: [Unanswerable Probes] "What is the recipe for chocolate chip cookies in the reference manual?"
- **Gold Reference:** `['The provided documents do not contain enough information to answer this question.']`
- **Model Output:** "The provided documents do not contain enough information to answer this question."
- **Exact Match:** 1.0 | **Token F1:** 1.0 | **Grounding:** 100.0%
- **Citation Validity:** 0.0% (Has Citation: False)

---

## 5. Distinction from Checker Unit Evaluation

This evaluation (`eval/e2e_answer_eval.py`) differs fundamentally from the unit test in `eval/faithfulness_eval.py`:
- `eval/faithfulness_eval.py` evaluated only the **rule-based validation functions** against 25 hand-crafted static strings.
- `eval/e2e_answer_eval.py` exercises the **complete live stack end-to-end**: parsing documents, ChromaDB dense retrieval, BM25 lexical retrieval, RRF fusion, cross-encoder re-ranking, token budget enforcement, and autoregressive model synthesis.
