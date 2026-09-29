# Intent Router Evaluation Report

**Date:** 2026-09-30 02:50:44
**Model:** `BAAI/bge-small-en-v1.5` (sentence-transformers)
**Scoring Strategy:** Mean of top-3 cosine similarities per intent
**Configured Threshold:** `0.55` (Fallback intent: `general`)
**Evaluation Dataset:** `router_eval.jsonl` (64 queries, zero exemplar leakage)

---

## 1. Executive Summary

- **Overall Accuracy:** `93.75%` (60 / 64 correct)
- **Mean Classification Latency:** `88.79 ms` (P50: `82.35 ms`, P95: `104.03 ms`)
- **Exemplar Leakage:** `0 duplicates` verified between evaluation set and training exemplars.

---

## 2. Per-Intent Performance Metrics

| Intent | Support | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **`general`** | 16 | 100.0% | 93.8% | 96.8% |
| **`technical`** | 16 | 88.2% | 93.8% | 90.9% |
| **`structured_json`** | 16 | 88.9% | 100.0% | 94.1% |
| **`rag`** | 16 | 100.0% | 87.5% | 93.3% |

---

## 3. Confusion Matrix

Rows represent ground-truth labels; columns represent router predictions at threshold `0.55`.

| Ground Truth \ Predicted | `general` | `technical` | `structured_json` | `rag` |
| :--- | :---: | :---: | :---: | :---: |
| **`general`** | **15** | 1 | 0 | 0 |
| **`technical`** | 0 | **15** | 1 | 0 |
| **`structured_json`** | 0 | 0 | **16** | 0 |
| **`rag`** | 0 | 1 | 1 | **14** |

---

## 4. Threshold Sweep (0.30 - 0.80)

A threshold sweep assesses router sensitivity: scores below threshold trigger a fallback to `general`.

| Threshold | Accuracy | Correct / Total | Fallbacks to `general` |
| :---: | :---: | :---: | :---: |
| `0.30` | 92.2% | 59/64 | 0 |
| `0.35` | 92.2% | 59/64 | 0 |
| `0.40` | 92.2% | 59/64 | 0 |
| `0.45` | 92.2% | 59/64 | 0 |
| `0.50` | 92.2% | 59/64 | 1 |
| `0.55` (Default) | 93.8% | 60/64 | 7 |
| `0.60` | 92.2% | 59/64 | 12 |
| `0.65` | 89.1% | 57/64 | 21 |
| `0.70` | 64.1% | 41/64 | 38 |
| `0.75` | 39.1% | 25/64 | 55 |
| `0.80` | 26.6% | 17/64 | 63 |

### Threshold Selection Rationale
- **Selected Operating Threshold:** `0.55`
- At `0.55`, the model maintains high discriminatory confidence across all distinct intents (`93.8%` accuracy) while preventing out-of-domain conversational noise from misrouting into specialized routes like `rag` or `structured_json`.
- Thresholds above `0.70` become overly conservative, causing legitimate borderline queries to collapse into `general` fallback.
- Thresholds below `0.45` risk routing ambiguous queries into specialized handlers without sufficient semantic alignment.

---

## 5. Tricky Edge-Case Analysis

The evaluation suite intentionally tested ambiguous and overlapping edge cases:

1. **Technical query mentioning 'document':**
   - Query: *'How does MongoDB index and query nested document structures inside collections?'*
   - Correctly classified as: `technical` (high similarity to database/storage exemplars rather than PDF QA).

2. **Technical query requesting JSON output:**
   - Query: *'Provide the TCP connection states as a valid JSON list of strings without commentary.'*
   - Correctly classified as: `structured_json` (structural constraint overrides domain context).

3. **RAG query referencing technical components:**
   - Query: *'According to the uploaded system design document, which port does the Redis cluster listen on?'*
   - Correctly classified as: `rag` (explicit document grounding correctly routes to retrieval pipeline).

4. **General knowledge query with numbers and coding history:**
   - Query: *'Who was Ada Lovelace and why is she celebrated as the earliest computer pioneer?'*
   - Correctly classified as: `general` (biographical historical query, avoiding false technical classification).
