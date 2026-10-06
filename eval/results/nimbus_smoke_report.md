# Nimbus Alerting Handbook — Regression Smoke Test Report

> **Notice**: This is a standalone regression smoke test on a fictional test document (`eval/smoke/Nimbus_Alerting_Handbook_TEST.pdf`, 4 pages) evaluating 9 targeted operational questions. These results are isolated and intentionally separate from the project's headline benchmark tables.

## 1. Summary of Hit@3 Across Strategies and Retrieval Modes

| Strategy | Retrieval Mode | Re-ranker | Hit@3 (%) | Hit@1 (%) | MRR | Latency (ms) | Failed |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `character` | `dense` | **Off** | **77.8%** | 55.6% | 0.6667 | 72.7 | 2/9 |
| `character` | `dense` | **On** | **88.9%** | 66.7% | 0.7778 | 274.5 | 1/9 |
| `character` | `sparse` | **Off** | **88.9%** | 77.8% | 0.8333 | 1.2 | 1/9 |
| `character` | `sparse` | **On** | **100.0%** | 66.7% | 0.8148 | 115.4 | 0/9 |
| `character` | `hybrid` | **Off** | **100.0%** | 66.7% | 0.7963 | 72.2 | 0/9 |
| `character` | `hybrid` | **On** | **88.9%** | 66.7% | 0.7778 | 271.4 | 1/9 |
| `structure` | `dense` | **Off** | **77.8%** | 66.7% | 0.7037 | 103.1 | 2/9 |
| `structure` | `dense` | **On** | **88.9%** | 77.8% | 0.8333 | 305.8 | 1/9 |
| `structure` | `sparse` | **Off** | **100.0%** | 88.9% | 0.9444 | 1.4 | 0/9 |
| `structure` | `sparse` | **On** | **88.9%** | 77.8% | 0.8333 | 150.6 | 1/9 |
| `structure` | `hybrid` | **Off** | **88.9%** | 66.7% | 0.7593 | 88.7 | 1/9 |
| `structure` | `hybrid` | **On** | **88.9%** | 77.8% | 0.8333 | 305.1 | 1/9 |
| `semantic` | `dense` | **Off** | **77.8%** | 66.7% | 0.7222 | 91.3 | 2/9 |
| `semantic` | `dense` | **On** | **88.9%** | 66.7% | 0.7407 | 304.7 | 1/9 |
| `semantic` | `sparse` | **Off** | **88.9%** | 66.7% | 0.7778 | 1.9 | 1/9 |
| `semantic` | `sparse` | **On** | **100.0%** | 66.7% | 0.7963 | 165.9 | 0/9 |
| `semantic` | `hybrid` | **Off** | **100.0%** | 55.6% | 0.7593 | 84.5 | 0/9 |
| `semantic` | `hybrid` | **On** | **100.0%** | 66.7% | 0.7963 | 294.9 | 0/9 |

## 2. Failed Questions Breakdown

### Configuration: `character | dense | rerank=Off` (2 failed)

- **Question**: "How many days are alerts kept?"
  - **Expected Substring**: `45`
  - **Retrieved Top Chunks**:
    - [1] `Nimbus Alerting Service: Operations Handbook (test document)   |   Page 4 5. SEC`
    - [2] `4 to 6 so that more notifications are delivered in parallel. Pause all SEV3 rule`
    - [3] `r 10 minutes, Nimbus escalates it to the secondary on-call engineer. If it is st`
- **Question**: "Why did I get just one page for 40 broken servers?"
  - **Expected Substring**: `fingerprint`
  - **Retrieved Top Chunks**:
    - [1] `failing servers with the same problem produce one page, not forty. 3.2 Severity `
    - [2] `| Typical action: Check provider status and retry. Code: ERR-3310 | Meaning: The`
    - [3] `4 to 6 so that more notifications are delivered in parallel. Pause all SEV3 rule`

### Configuration: `character | dense | rerank=On` (1 failed)

- **Question**: "What is the default retention period?"
  - **Expected Substring**: `45`
  - **Retrieved Top Chunks**:
    - [1] `needs an approval token and cannot be longer than 8 hours.`
    - [2] `written to application logs. Processed alerts are deleted automatically once the`
    - [3] `3. ALERT PROCESSING 3.1 Deduplication When many systems report the same fault, N`

### Configuration: `character | sparse | rerank=Off` (1 failed)

- **Question**: "What is the default retention period?"
  - **Expected Substring**: `45`
  - **Retrieved Top Chunks**:
    - [1] `written to application logs. Processed alerts are deleted automatically once the`
    - [2] `ow long processed alerts are kept before deletion. Variable: NIMBUS_MAX_BATCH | `
    - [3] `3. ALERT PROCESSING 3.1 Deduplication When many systems report the same fault, N`

### Configuration: `character | hybrid | rerank=On` (1 failed)

- **Question**: "What is the default retention period?"
  - **Expected Substring**: `45`
  - **Retrieved Top Chunks**:
    - [1] `needs an approval token and cannot be longer than 8 hours.`
    - [2] `written to application logs. Processed alerts are deleted automatically once the`
    - [3] `3. ALERT PROCESSING 3.1 Deduplication When many systems report the same fault, N`

### Configuration: `structure | dense | rerank=Off` (2 failed)

- **Question**: "How many days are alerts kept?"
  - **Expected Substring**: `45`
  - **Retrieved Top Chunks**:
    - [1] `3.3 Escalation If a SEV1 alert is not acknowledged after 10 minutes, Nimbus esca`
    - [2] `6.2 What happens if nobody responds to a critical alert? Critical alerts are pas`
    - [3] `5. SECURITY AND PRIVACY  5.1 Access control Nimbus has three roles. Viewers can `
- **Question**: "Why did I get just one page for 40 broken servers?"
  - **Expected Substring**: `fingerprint`
  - **Retrieved Top Chunks**:
    - [1] `4.1 Error codes Code: ERR-4107 | Meaning: A rule contains invalid syntax and was`
    - [2] `4.2 Backlog recovery When ERR-3310 appears, first confirm that the rule engine i`
    - [3] `Nimbus Alerting Service: Operations Handbook (test document)   |   Page 4`

### Configuration: `structure | dense | rerank=On` (1 failed)

- **Question**: "What is the default retention period?"
  - **Expected Substring**: `45`
  - **Retrieved Top Chunks**:
    - [1] `5. SECURITY AND PRIVACY  5.1 Access control Nimbus has three roles. Viewers can `
    - [2] `3. ALERT PROCESSING  3.1 Deduplication When many systems report the same fault, `
    - [3] `3.3 Escalation If a SEV1 alert is not acknowledged after 10 minutes, Nimbus esca`

### Configuration: `structure | sparse | rerank=On` (1 failed)

- **Question**: "What is the default retention period?"
  - **Expected Substring**: `45`
  - **Retrieved Top Chunks**:
    - [1] `5. SECURITY AND PRIVACY  5.1 Access control Nimbus has three roles. Viewers can `
    - [2] `3. ALERT PROCESSING  3.1 Deduplication When many systems report the same fault, `
    - [3] `3.3 Escalation If a SEV1 alert is not acknowledged after 10 minutes, Nimbus esca`

### Configuration: `structure | hybrid | rerank=Off` (1 failed)

- **Question**: "How many days are alerts kept?"
  - **Expected Substring**: `45`
  - **Retrieved Top Chunks**:
    - [1] `6.2 What happens if nobody responds to a critical alert? Critical alerts are pas`
    - [2] `5. SECURITY AND PRIVACY  5.1 Access control Nimbus has three roles. Viewers can `
    - [3] `3.3 Escalation If a SEV1 alert is not acknowledged after 10 minutes, Nimbus esca`

### Configuration: `structure | hybrid | rerank=On` (1 failed)

- **Question**: "What is the default retention period?"
  - **Expected Substring**: `45`
  - **Retrieved Top Chunks**:
    - [1] `5. SECURITY AND PRIVACY  5.1 Access control Nimbus has three roles. Viewers can `
    - [2] `3. ALERT PROCESSING  3.1 Deduplication When many systems report the same fault, `
    - [3] `3.3 Escalation If a SEV1 alert is not acknowledged after 10 minutes, Nimbus esca`

### Configuration: `semantic | dense | rerank=Off` (2 failed)

- **Question**: "How many days are alerts kept?"
  - **Expected Substring**: `45`
  - **Retrieved Top Chunks**:
    - [1] `6.2 What happens if nobody responds to a critical alert? Critical alerts are pas`
    - [2] `If it is still unacknowledged after 25 minutes, Nimbus escalates to the engineer`
    - [3] `Nimbus Alerting Service: Operations Handbook (test document)   |   Page 4 5. SEC`
- **Question**: "Why did I get just one page for 40 broken servers?"
  - **Expected Substring**: `fingerprint`
  - **Retrieved Top Chunks**:
    - [1] `It increases a counter on the existing alert instead. This is why forty failing `
    - [2] `Code: ERR-5203 | Meaning: The notifier timed out after 12 seconds while contacti`
    - [3] `A window can last at most 8 hours and requires an approval token issued by an ad`

### Configuration: `semantic | dense | rerank=On` (1 failed)

- **Question**: "Why did I get just one page for 40 broken servers?"
  - **Expected Substring**: `fingerprint`
  - **Retrieved Top Chunks**:
    - [1] `It increases a counter on the existing alert instead. This is why forty failing `
    - [2] `Code: ERR-5203 | Meaning: The notifier timed out after 12 seconds while contacti`
    - [3] `A window can last at most 8 hours and requires an approval token issued by an ad`

### Configuration: `semantic | sparse | rerank=Off` (1 failed)

- **Question**: "What happens if nobody answers a critical alert?"
  - **Expected Substring**: `10 minutes`
  - **Retrieved Top Chunks**:
    - [1] `6.2 What happens if nobody responds to a critical alert? Critical alerts are pas`
    - [2] `If it is still unacknowledged after 25 minutes, Nimbus escalates to the engineer`
    - [3] `3. ALERT PROCESSING 3.1 Deduplication When many systems report the same fault, N`

