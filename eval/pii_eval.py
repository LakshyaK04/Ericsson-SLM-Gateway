"""PII Redaction Evaluation Pipeline.

Evaluates PII redaction recall per entity type and false-positive rate
on normal conversational text using eval/datasets/pii_eval.jsonl.

Generates eval/results/pii_report.md with per-entity metrics.
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

# Resolve paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
EVAL_DATASET = SCRIPT_DIR / "datasets" / "pii_eval.jsonl"
RESULTS_DIR = SCRIPT_DIR / "results"
REPORT_PATH = RESULTS_DIR / "pii_report.md"

# Ensure gateway package is importable
sys.path.insert(0, str(ROOT_DIR / "gateway" / "src"))

from slm_gateway.config import Settings
from slm_gateway.pii import PIIRedactor


def load_eval_cases():
    """Load evaluation cases from JSONL file."""
    cases = []
    with open(EVAL_DATASET, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                case = json.loads(line)
                case["line_num"] = line_num
                cases.append(case)
            except json.JSONDecodeError as e:
                print(f"WARNING: Skipping malformed line {line_num}: {e}")
    return cases


def run_evaluation():
    """Execute PII evaluation and generate report."""
    print("=" * 60)
    print("PII Redaction Evaluation Pipeline")
    print("=" * 60)

    # Load evaluation dataset
    cases = load_eval_cases()
    print(f"\nLoaded {len(cases)} evaluation cases from {EVAL_DATASET.name}")

    recall_cases = [c for c in cases if c["category"] == "recall"]
    fp_cases = [c for c in cases if c["category"] == "false_positive"]
    multi_cases = [c for c in cases if c["category"] == "multi_entity"]

    print(f"  Recall cases:         {len(recall_cases)}")
    print(f"  False-positive cases: {len(fp_cases)}")
    print(f"  Multi-entity cases:   {len(multi_cases)}")

    # Initialize redactor
    print("\nInitializing PIIRedactor...")
    cfg = Settings(PII_FAIL_MODE="closed")
    redactor = PIIRedactor(cfg)
    print("PIIRedactor ready.\n")

    # --- Recall Evaluation ---
    print("-" * 40)
    print("RECALL EVALUATION")
    print("-" * 40)

    entity_stats = defaultdict(lambda: {"total": 0, "detected": 0, "failures": []})
    total_recall_pass = 0
    total_recall_fail = 0
    latencies = []

    for case in recall_cases:
        text = case["text"]
        entity = case["expected_entity"]
        placeholder = case["expected_placeholder"]

        entity_stats[entity]["total"] += 1

        start = time.perf_counter()
        redacted, count = redactor.redact(text)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

        if count >= 1 and placeholder in redacted:
            entity_stats[entity]["detected"] += 1
            total_recall_pass += 1
            status = "PASS"
        else:
            total_recall_fail += 1
            entity_stats[entity]["failures"].append(text)
            status = "FAIL"

        print(f"  [{status}] {entity:20s} | redactions={count} | {text[:60]}...")

    # --- False-Positive Evaluation ---
    print("\n" + "-" * 40)
    print("FALSE-POSITIVE EVALUATION")
    print("-" * 40)

    fp_pass = 0
    fp_fail = 0
    fp_failures = []

    for case in fp_cases:
        text = case["text"]

        start = time.perf_counter()
        redacted, count = redactor.redact(text)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

        if count == 0 and redacted == text:
            fp_pass += 1
            status = "PASS"
        else:
            fp_fail += 1
            fp_failures.append({"text": text, "redacted": redacted, "count": count})
            status = "FAIL"

        print(f"  [{status}] redactions={count} | {text[:70]}")

    # --- Multi-Entity Evaluation ---
    print("\n" + "-" * 40)
    print("MULTI-ENTITY EVALUATION")
    print("-" * 40)

    multi_pass = 0
    multi_fail = 0

    for case in multi_cases:
        text = case["text"]

        start = time.perf_counter()
        redacted, count = redactor.redact(text)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

        if count >= 2:
            multi_pass += 1
            status = "PASS"
        else:
            multi_fail += 1
            status = "FAIL"

        print(f"  [{status}] redactions={count} | {text[:70]}")

    # --- Summary ---
    total_cases = len(cases)
    total_pass = total_recall_pass + fp_pass + multi_pass
    total_fail = total_recall_fail + fp_fail + multi_fail
    overall_accuracy = (total_pass / total_cases * 100) if total_cases > 0 else 0.0
    recall_rate = (total_recall_pass / len(recall_cases) * 100) if recall_cases else 0.0
    fp_rate = (fp_fail / len(fp_cases) * 100) if fp_cases else 0.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total Cases:          {total_cases}")
    print(f"  Passed:               {total_pass}")
    print(f"  Failed:               {total_fail}")
    print(f"  Overall Accuracy:     {overall_accuracy:.2f}%")
    print(f"  Recall Rate:          {recall_rate:.2f}%")
    print(f"  False-Positive Rate:  {fp_rate:.2f}%")
    print(f"  Avg Latency:          {avg_latency:.1f} ms")

    # --- Generate Report ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    generate_report(
        entity_stats=entity_stats,
        recall_cases=recall_cases,
        fp_cases=fp_cases,
        multi_cases=multi_cases,
        total_recall_pass=total_recall_pass,
        total_recall_fail=total_recall_fail,
        fp_pass=fp_pass,
        fp_fail=fp_fail,
        fp_failures=fp_failures,
        multi_pass=multi_pass,
        multi_fail=multi_fail,
        overall_accuracy=overall_accuracy,
        recall_rate=recall_rate,
        fp_rate=fp_rate,
        avg_latency=avg_latency,
    )
    print(f"\nReport saved to: {REPORT_PATH}")


def generate_report(**kwargs):
    """Generate Markdown evaluation report."""
    entity_stats = kwargs["entity_stats"]

    lines = [
        "# PII Redaction Evaluation Report",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Total Cases | {len(kwargs['recall_cases']) + len(kwargs['fp_cases']) + len(kwargs['multi_cases'])} |",
        f"| Recall Rate | {kwargs['recall_rate']:.2f}% |",
        f"| False-Positive Rate | {kwargs['fp_rate']:.2f}% |",
        f"| Overall Accuracy | {kwargs['overall_accuracy']:.2f}% |",
        f"| Avg Redaction Latency | {kwargs['avg_latency']:.1f} ms |",
        "",
        "## Per-Entity Recall",
        "",
        "| Entity Type | Total | Detected | Recall (%) |",
        "|---|:---:|:---:|:---:|",
    ]

    for entity in sorted(entity_stats.keys()):
        stats = entity_stats[entity]
        recall_pct = (stats["detected"] / stats["total"] * 100) if stats["total"] > 0 else 0.0
        lines.append(f"| `{entity}` | {stats['total']} | {stats['detected']} | {recall_pct:.1f}% |")

    lines.extend([
        "",
        "## False-Positive Analysis",
        "",
        f"- **Clean queries tested:** {len(kwargs['fp_cases'])}",
        f"- **Incorrectly redacted:** {kwargs['fp_fail']}",
        f"- **False-positive rate:** {kwargs['fp_rate']:.2f}%",
        "",
    ])

    if kwargs["fp_failures"]:
        lines.append("### False-Positive Failures")
        lines.append("")
        for fp in kwargs["fp_failures"]:
            lines.append(f"- Input: `{fp['text']}`")
            lines.append(f"  - Redacted: `{fp['redacted']}`")
            lines.append(f"  - Redaction count: {fp['count']}")
            lines.append("")

    lines.extend([
        "## Multi-Entity Detection",
        "",
        f"- **Multi-entity cases:** {len(kwargs['multi_cases'])}",
        f"- **Passed (>=2 redactions):** {kwargs['multi_pass']}",
        f"- **Failed:** {kwargs['multi_fail']}",
        "",
    ])

    # Entity failures
    failed_entities = {e: s for e, s in entity_stats.items() if s["failures"]}
    if failed_entities:
        lines.extend([
            "## Recall Failures",
            "",
        ])
        for entity, stats in failed_entities.items():
            lines.append(f"### `{entity}`")
            for fail_text in stats["failures"]:
                lines.append(f"- `{fail_text}`")
            lines.append("")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    run_evaluation()
