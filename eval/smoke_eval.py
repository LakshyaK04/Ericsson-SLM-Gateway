"""Smoke Test Evaluation Suite for Nimbus Alerting Handbook.

Evaluates retrieval efficacy on a fictional test document (eval/smoke/Nimbus_Alerting_Handbook_TEST.pdf)
across 9 targeted QA pairs with exact substring matching in top-3 retrieved chunks.

Evaluates:
- Strategies: character, structure, semantic
- Modes: dense, sparse, hybrid
- Re-ranker: Off, On

Outputs:
- eval/results/nimbus_smoke_report.csv
- eval/results/nimbus_smoke_report.md
"""

import json
import logging
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "rag" / "src"))

from rag_service.chunking import chunk_document
from rag_service.config import Settings
from rag_service.embeddings import EmbeddingModel
from rag_service.parsers import parse_document
from rag_service.reranker import Reranker
from rag_service.retriever import Retriever
from rag_service.store import ChromaStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("smoke_eval")


def normalize_for_matching(text: str) -> str:
    """Normalize text for invariant substring matching."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def run_smoke_evaluation():
    smoke_pdf = REPO_ROOT / "eval" / "smoke" / "Nimbus_Alerting_Handbook_TEST.pdf"
    qa_file = REPO_ROOT / "eval" / "smoke" / "nimbus_qa.jsonl"
    results_dir = REPO_ROOT / "eval" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if not smoke_pdf.exists():
        raise FileNotFoundError(f"Nimbus PDF smoke document not found: {smoke_pdf}")
    if not qa_file.exists():
        raise FileNotFoundError(f"Nimbus QA dataset not found: {qa_file}")

    with open(qa_file, "r", encoding="utf-8") as f:
        qa_pairs = [json.loads(line) for line in f if line.strip()]

    logger.info("Loaded %d smoke QA pairs for Nimbus document.", len(qa_pairs))

    # Parse document
    pages = parse_document(smoke_pdf, filename=smoke_pdf.name)
    logger.info("Parsed %s: %d pages.", smoke_pdf.name, len(pages))

    # Isolated temporary persistence directory for ChromaDB
    temp_dir = tempfile.mkdtemp(prefix="chroma_nimbus_smoke_")
    logger.info("Using temporary ChromaDB at %s", temp_dir)

    try:
        cfg = Settings(CHROMA_PERSIST_DIR=temp_dir)
        store = ChromaStore(temp_dir)
        emb_model = EmbeddingModel()
        reranker = Reranker()
        retriever = Retriever(store=store, embedding_model=emb_model, reranker=reranker, config=cfg)

        strategies = ["character", "structure", "semantic"]
        retrieval_modes = ["dense", "sparse", "hybrid"]
        reranker_options = [False, True]

        # Index all strategies into Chroma
        for strat in strategies:
            chunked = chunk_document(
                pages=pages,
                source=smoke_pdf.name,
                strategies=[strat],
                embedding_model=emb_model,
            )
            chunks = chunked.get(strat, [])
            if chunks:
                texts = [c.text for c in chunks]
                embeddings = emb_model.encode_documents(texts)
                store.add_chunks(
                    strategy=strat,
                    chunks=chunks,
                    embeddings=embeddings,
                    doc_id=smoke_pdf.name,
                )
                logger.info("Indexed strategy '%s': %d chunks.", strat, len(chunks))

        report_rows = []
        failures_by_config = {}

        for strat in strategies:
            for mode in retrieval_modes:
                for use_rerank in reranker_options:
                    rerank_label = "On" if use_rerank else "Off"
                    config_key = f"{strat} | {mode} | rerank={rerank_label}"
                    hit_1_count = 0
                    hit_3_count = 0
                    rr_sum = 0.0
                    latencies = []
                    failed_questions = []

                    for qa in qa_pairs:
                        q = qa["question"]
                        expected = normalize_for_matching(qa["expected_substring"])

                        t0 = time.perf_counter()
                        results = retriever.retrieve(
                            query=q,
                            strategy=strat,
                            retrieve_k=20,
                            final_k=3,
                            use_reranker=use_rerank,
                            retrieval_mode=mode,
                        )
                        latency_ms = (time.perf_counter() - t0) * 1000.0
                        latencies.append(latency_ms)

                        match_rank = 0
                        for r_idx, r in enumerate(results, start=1):
                            norm_text = normalize_for_matching(r.text)
                            if expected in norm_text:
                                match_rank = r_idx
                                break

                        if match_rank == 1:
                            hit_1_count += 1
                            hit_3_count += 1
                            rr_sum += 1.0
                        elif match_rank == 2:
                            hit_3_count += 1
                            rr_sum += 0.5
                        elif match_rank == 3:
                            hit_3_count += 1
                            rr_sum += 1.0 / 3.0
                        else:
                            failed_questions.append(
                                {
                                    "question": q,
                                    "expected": qa["expected_substring"],
                                    "top_chunks": [r.text[:80].replace("\n", " ") for r in results],
                                }
                            )

                    n_q = len(qa_pairs)
                    hit_1 = (hit_1_count / n_q) * 100.0
                    hit_3 = (hit_3_count / n_q) * 100.0
                    mrr = rr_sum / n_q
                    avg_latency = sum(latencies) / len(latencies)

                    report_rows.append(
                        {
                            "strategy": strat,
                            "mode": mode,
                            "reranker": rerank_label,
                            "hit_3": round(hit_3, 2),
                            "hit_1": round(hit_1, 2),
                            "mrr": round(mrr, 4),
                            "avg_latency_ms": round(avg_latency, 2),
                            "failed_count": len(failed_questions),
                        }
                    )

                    failures_by_config[config_key] = failed_questions

                    logger.info(
                        "Config [%s]: Hit@3=%.1f%%, Hit@1=%.1f%%, MRR=%.4f (Failed: %d)",
                        config_key,
                        hit_3,
                        hit_1,
                        mrr,
                        len(failed_questions),
                    )

        # Write CSV report
        csv_path = results_dir / "nimbus_smoke_report.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(
                "Strategy,Mode,Reranker,Hit@3 (%),Hit@1 (%),MRR,Avg Latency (ms),Failed Questions\n"
            )
            for r in report_rows:
                f.write(
                    f"{r['strategy']},{r['mode']},{r['reranker']},"
                    f"{r['hit_3']:.2f},{r['hit_1']:.2f},{r['mrr']:.4f},{r['avg_latency_ms']:.2f},{r['failed_count']}\n"
                )
        logger.info("Saved smoke test CSV to %s", csv_path)

        # Write Markdown report
        md_path = results_dir / "nimbus_smoke_report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Nimbus Alerting Handbook — Regression Smoke Test Report\n\n")
            f.write(
                "> **Notice**: This is a standalone regression smoke test on a fictional test document "
                "(`eval/smoke/Nimbus_Alerting_Handbook_TEST.pdf`, 4 pages) evaluating 9 targeted operational questions. "
                "These results are isolated and intentionally separate from the project's headline benchmark tables.\n\n"
            )

            f.write("## 1. Summary of Hit@3 Across Strategies and Retrieval Modes\n\n")
            f.write(
                "| Strategy | Retrieval Mode | Re-ranker | Hit@3 (%) | Hit@1 (%) | MRR | Latency (ms) | Failed |\n"
            )
            f.write("|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
            for r in report_rows:
                f.write(
                    f"| `{r['strategy']}` | `{r['mode']}` | **{r['reranker']}** | "
                    f"**{r['hit_3']:.1f}%** | {r['hit_1']:.1f}% | {r['mrr']:.4f} | {r['avg_latency_ms']:.1f} | {r['failed_count']}/9 |\n"
                )
            f.write("\n")

            f.write("## 2. Failed Questions Breakdown\n\n")
            has_any_failures = any(len(fails) > 0 for fails in failures_by_config.values())
            if not has_any_failures:
                f.write("All 9 questions passed across all evaluated configurations.\n")
            else:
                for cfg_name, fails in failures_by_config.items():
                    if fails:
                        f.write(f"### Configuration: `{cfg_name}` ({len(fails)} failed)\n\n")
                        for item in fails:
                            f.write(f'- **Question**: "{item["question"]}"\n')
                            f.write(f"  - **Expected Substring**: `{item['expected']}`\n")
                            f.write("  - **Retrieved Top Chunks**:\n")
                            for idx, snippet in enumerate(item["top_chunks"], start=1):
                                f.write(f"    - [{idx}] `{snippet}`\n")
                        f.write("\n")

        logger.info("Saved smoke test Markdown to %s", md_path)
        print("\n" + "=" * 80)
        print("NIMBUS SMOKE TEST COMPLETED")
        print("=" * 80)
        with open(csv_path) as cf:
            print(cf.read())

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.info("Cleaned up temporary ChromaDB directory.")


if __name__ == "__main__":
    run_smoke_evaluation()
