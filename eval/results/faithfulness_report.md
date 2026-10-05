# Grounded Answer Quality, Faithfulness & Citation Evaluation Report

- **Evaluation Run Date:** 2026-10-05 23:44:59
- **Evaluated Pipeline:** Two-stage RAG generation (`SYSTEM_PROMPT` with numbered block citation format)
- **Evaluation Methodology:** Rule-based citation extraction + factual token containment verification + refusal assertion analysis
- **Sample Size:** 25 test cases (15 fully grounded + 5 hallucinated adversarial + 5 insufficient context refusals)

---

## 1. Quantitative Performance Matrix

| Evaluation Dimension | Metric | Measured Value | Standard / Target | Description |
|:---|:---|:---:|:---:|:---|
| **Citation Compliance** | Citation Presence Rate | **100.0%** | $\ge 95\%$ | Percentage of non-refusal answers citing context blocks via `[N]` |
| **Citation Accuracy** | Citation Precision | **100.0%** | $100\%$ | Percentage of cited block indices that map to valid retrieved chunks |
| **Factual Groundedness** | Mean Grounding Ratio | **69.48%** | $\ge 85\%$ | Average percentage of factual/alphanumeric tokens corroborated by source context |
| **Hallucination Detection** | Detection Sensitivity | **100.0%** | $100\%$ | Ability to flag answers containing fabricated entities or numbers absent from context |
| **Refusal Integrity** | Out-of-Domain Refusal Rate | **100.0%** | $100\%$ | Accurate emission of standard refusal string when context is insufficient |

---

## 2. Test Set Breakdown & Diagnostic Results

### 2.1 Grounded QA Samples with Citations
- **Query:** Where were the Normans settled in France?
  - Citations Extracted: `[1]`
  - Grounding Score: **83.3%**
  - Ungrounded Tokens: `['settled']`

- **Query:** What is the primary function of the UPF in 5G standalone architecture?
  - Citations Extracted: `[1]`
  - Grounding Score: **85.7%**
  - Ungrounded Tokens: `['handles']`

- **Query:** What is the Raft consensus algorithm designed to do?
  - Citations Extracted: `[1, 2]`
  - Grounding Score: **87.5%**
  - Ungrounded Tokens: `['providing']`

### 2.2 Adversarial Hallucination Catch Cases
- **Query:** What port does the SLM Gateway listen on?
  - Grounding Score: **28.6%** (Anomaly Detected)
  - Flagged Ungrounded Tokens: `['tlsv1', 'encryption', 'utilizes', '9090', 'listens']`

- **Query:** What is the maximum context length of Phi-3 Mini?
  - Grounding Score: **28.6%** (Anomaly Detected)
  - Flagged Ungrounded Tokens: `['128000', 'mini', 'phi-3', 'using', 'flashattention-3']`

- **Query:** Who is the author of the Raft consensus paper?
  - Grounding Score: **28.6%** (Anomaly Detected)
  - Flagged Ungrounded Tokens: `['satoshi', 'lamport', 'leslie', 'written', 'nakamoto']`

### 2.3 Insufficient Context Refusal Honesty
- **Query:** What is the quantum state coherence time of the D-Wave processor?
  - Refusal Emitted: **True** (`The provided documents do not contain enough information to answer this question.`)
  - Hallucination Avoided: **Yes**

- **Query:** Who won the FIFA World Cup in 1994?
  - Refusal Emitted: **True** (`The provided documents do not contain enough information to answer this question.`)
  - Hallucination Avoided: **Yes**

- **Query:** What is the formula for calculating Black-Scholes option pricing?
  - Refusal Emitted: **True** (`The provided documents do not contain enough information to answer this question.`)
  - Hallucination Avoided: **Yes**

---

## 3. Explicit Methodological Limitations & Honest Disclosure

1. **Lexical / Entity Overlap vs. Deep NLI Entailment:** This rule-based evaluator computes token and named entity overlap between the generated answer and the cited context blocks. While this reliably catches hallucinated numbers, dates, and proper nouns (e.g. port numbers, false authors, fabricated metrics), it does not detect subtler semantic contradictions (such as inverted logic or false causal attributions) that an NLI cross-encoder model would catch.
2. **Citation Formatting Drift:** While strict system prompt instructions enforce `[1]`, `[2]` bracketed notation, conversational SLMs may occasionally use parenthetical `(Block 1)` or combined `[1, 2]` formatting. The extraction regex accommodates standard variants, but format compliance should be monitored over continuous production queries.
3. **Corpus Scope:** The test set evaluates 25 representative scenarios across telecom, distributed systems, and historical topics. Enterprise deployments should integrate this evaluation into automated CI telemetry across larger domain-specific datasets.
