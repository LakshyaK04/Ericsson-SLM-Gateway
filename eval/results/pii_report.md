# PII Redaction Evaluation Report

## Summary

| Metric | Value |
|---|---|
| Total Cases | 33 |
| Recall Rate | 100.00% |
| False-Positive Rate | 0.00% |
| Overall Accuracy | 100.00% |
| Avg Redaction Latency | 37.5 ms |

## Per-Entity Recall

| Entity Type | Total | Detected | Recall (%) |
|---|:---:|:---:|:---:|
| `CREDIT_CARD` | 2 | 2 | 100.0% |
| `EMAIL_ADDRESS` | 2 | 2 | 100.0% |
| `EMPLOYEE_ID` | 3 | 3 | 100.0% |
| `IP_ADDRESS` | 2 | 2 | 100.0% |
| `PERSON` | 3 | 3 | 100.0% |
| `PHONE_NUMBER` | 2 | 2 | 100.0% |
| `PROJECT_CODENAME` | 2 | 2 | 100.0% |

## False-Positive Analysis

- **Clean queries tested:** 15
- **Incorrectly redacted:** 0
- **False-positive rate:** 0.00%

## Multi-Entity Detection

- **Multi-entity cases:** 2
- **Passed (>=2 redactions):** 2
- **Failed:** 0

