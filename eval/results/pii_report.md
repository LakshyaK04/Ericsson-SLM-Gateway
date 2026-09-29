# PII Redaction Engine Evaluation Report

**Date:** 2026-09-30 02:53:00
**NLP Engine:** Microsoft Presidio Analyzer + spaCy `en_core_web_sm`
**Policy:** Fail-Closed (`PII_FAIL_MODE=closed`), Typed Placeholders (`<ENTITY_NAME>`)
**Evaluation Dataset:** `pii_eval.jsonl` (45 test cases)

---

## 1. Executive Summary

- **Overall PII Entity Recall:** `100.00%` (35 / 35 entities detected)
- **False Positive Rate on Clean Text:** `0.00%` (0 / 10 clean queries redacted)
- **Mean Processing Latency:** `42.40 ms` per query
- **Restricted Entity List Protection:** Verified that `LOCATION` and `DATE_TIME` redactions are disabled, avoiding false positive corruption of geographic or temporal queries.

---

## 2. Per-Entity Recall Breakdown

| Entity Type | Category | Support | Detected | Recall | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`CREDIT_CARD`** | Standard Presidio | 5 | 5 | 100.0% | PASSED |
| **`EMAIL_ADDRESS`** | Standard Presidio | 5 | 5 | 100.0% | PASSED |
| **`EMPLOYEE_ID`** | Custom Enterprise | 5 | 5 | 100.0% | PASSED |
| **`IP_ADDRESS`** | Standard Presidio | 5 | 5 | 100.0% | PASSED |
| **`PERSON`** | Standard Presidio | 5 | 5 | 100.0% | PASSED |
| **`PHONE_NUMBER`** | Standard Presidio | 5 | 5 | 100.0% | PASSED |
| **`PROJECT_CODENAME`** | Custom Enterprise | 5 | 5 | 100.0% | PASSED |

---

## 3. False Positive Protection Analysis

A core requirement of Section 5.2 is ensuring that non-sensitive queries (e.g. 'What is the capital of Germany?', 'The summit will take place on Monday at 3 PM in Stockholm') are **not** redacted as `<LOCATION>` or `<DATE_TIME>`.

- **Clean Queries Tested:** 10
- **False Positives Observed:** 0
- **FPR:** `0.00%`

| Test Query | Redacted Output | Expected | Result |
| :--- | :--- | :---: | :---: |
| What is the capital of Germany? | What is the capital of Germany? | Clean (0 redactions) | **PASSED** |
| The annual summit will be held on Monday at 3 PM in Stockholm. | The annual summit will be held on Monday at 3 PM in Stockholm. | Clean (0 redactions) | **PASSED** |
| Explain photosynthesis and how chlorophyll absorbs solar radiation. | Explain photosynthesis and how chlorophyll absorbs solar radiation. | Clean (0 redactions) | **PASSED** |
| What does an HTTP 404 Not Found status code indicate? | What does an HTTP 404 Not Found status code indicate? | Clean (0 redactions) | **PASSED** |
| How do I implement binary search in Python without recursion? | How do I implement binary search in Python without recursion? | Clean (0 redactions) | **PASSED** |
| The flight lands in Tokyo on October 15 at 18:30. | The flight lands in Tokyo on October 15 at 18:30. | Clean (0 redactions) | **PASSED** |
| Describe the difference between synchronous and asynchronous operations. | Describe the difference between synchronous and asynchronous operations. | Clean (0 redactions) | **PASSED** |
| Why do leaves change color during the autumn season? | Why do leaves change color during the autumn season? | Clean (0 redactions) | **PASSED** |
| The speed of light in vacuum is approximately 299,792,458 meters per second. | The speed of light in vacuum is approximately 299,792,458 meters per second. | Clean (0 redactions) | **PASSED** |
| Explain the architectural components of a 5G core network. | Explain the architectural components of a 5G core network. | Clean (0 redactions) | **PASSED** |

---

## 4. Custom Enterprise Recognizers Evaluation

1. **`EMPLOYEE_ID` Pattern (`EMP-\d{5,7}`):**
   - Successfully captures corporate employee numbers (e.g., `EMP-12345`, `EMP-987654`, `EMP-5555555`).
   - High regex score (0.85) ensures reliable detection without matching arbitrary hyphenated words.

2. **`PROJECT_CODENAME` Deny-List (`PII_PROJECT_CODENAMES`):**
   - Matches all internal confidential code names (`Project-Titan`, `Project-Apollo`, `Project-Odin`, `Project-Thor`, `Project-Aegis`).
   - Protects internal confidential initiatives from being exposed to downstream LLMs.

3. **`PHONE_NUMBER` Robust Pattern:**
   - Extends Presidio's default NANP validation to capture diverse international formats (`+46-8-555-1234`, `(212) 555-0199`, `+44 20 7946 0912`).
