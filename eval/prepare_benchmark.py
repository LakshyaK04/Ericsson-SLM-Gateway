"""
Reproducible Benchmark Dataset Preparation Script.

Downloads and extracts a slice of the public Stanford Question Answering Dataset (SQuAD v2.0)
development set into:
1. eval/datasets/squad_retrieval_corpus.jsonl (500 distinct passage contexts)
2. eval/datasets/squad_retrieval_queries.jsonl (150 realistic queries with gold doc mappings)

Source: https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json
"""

import json
import logging
from pathlib import Path
import random
import urllib.request

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_benchmark")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = REPO_ROOT / "eval" / "datasets"
SQUAD_RAW_FILE = DATASETS_DIR / "squad_dev_v2_raw.json"
CORPUS_OUTPUT = DATASETS_DIR / "squad_retrieval_corpus.jsonl"
QUERIES_OUTPUT = DATASETS_DIR / "squad_retrieval_queries.jsonl"
SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"


def download_squad_raw(target_path: Path) -> None:
    """Download raw SQuAD v2.0 dev dataset if not already present locally."""
    if target_path.exists() and target_path.stat().st_size > 100000:
        logger.info("Found cached SQuAD raw dataset at %s (%d bytes).", target_path, target_path.stat().st_size)
        return

    target_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading SQuAD v2.0 dev dataset from %s...", SQUAD_URL)
    req = urllib.request.Request(
        SQUAD_URL,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp, open(target_path, "wb") as f:
        data = resp.read()
        f.write(data)
    logger.info("Downloaded %d bytes to %s.", len(data), target_path)


def prepare_benchmark(
    max_passages: int = 500,
    max_queries: int = 150,
    seed: int = 42,
) -> None:
    """Extract a reproducible retrieval benchmark from SQuAD v2.0."""
    download_squad_raw(SQUAD_RAW_FILE)

    with open(SQUAD_RAW_FILE, "r", encoding="utf-8") as f:
        squad_json = json.load(f)

    articles = squad_json.get("data", [])
    logger.info("Loaded %d articles from SQuAD dev set.", len(articles))

    # Deterministic sampling
    rng = random.Random(seed)

    corpus_entries = []
    query_entries = []
    doc_id_counter = 0

    # Collect passages and their answerable questions
    for article_idx, article in enumerate(articles):
        title = article.get("title", f"Article_{article_idx}")
        paragraphs = article.get("paragraphs", [])

        for p_idx, para in enumerate(paragraphs):
            context = para.get("context", "").strip()
            if len(context) < 100:  # skip trivial stubs
                continue

            doc_id = f"squad_doc_{doc_id_counter:04d}"
            doc_id_counter += 1

            corpus_entries.append({
                "doc_id": doc_id,
                "title": title,
                "text": context,
            })

            # Check for answerable questions in this passage
            qas = para.get("qas", [])
            for qa in qas:
                is_impossible = qa.get("is_impossible", False)
                if is_impossible:
                    continue  # We need answerable queries with valid gold documents

                q_text = qa.get("question", "").strip()
                answers = [a["text"] for a in qa.get("answers", []) if a.get("text")]
                if not q_text or not answers:
                    continue

                query_entries.append({
                    "query_id": f"squad_q_{len(query_entries):04d}",
                    "query": q_text,
                    "gold_doc_id": doc_id,
                    "title": title,
                    "answers": answers,
                })

            if len(corpus_entries) >= max_passages * 2 and len(query_entries) >= max_queries * 3:
                break
        if len(corpus_entries) >= max_passages * 2:
            break

    logger.info("Collected %d candidate passages and %d candidate queries.", len(corpus_entries), len(query_entries))

    # Downsample deterministically to target sizes
    rng.shuffle(query_entries)
    selected_queries = query_entries[:max_queries]

    # Ensure all gold documents for selected queries are in the corpus
    needed_doc_ids = {q["gold_doc_id"] for q in selected_queries}
    selected_corpus_map = {c["doc_id"]: c for c in corpus_entries if c["doc_id"] in needed_doc_ids}

    # Fill remaining corpus passages up to max_passages
    remaining_corpus = [c for c in corpus_entries if c["doc_id"] not in selected_corpus_map]
    rng.shuffle(remaining_corpus)

    for c in remaining_corpus:
        if len(selected_corpus_map) >= max_passages:
            break
        selected_corpus_map[c["doc_id"]] = c

    final_corpus = list(selected_corpus_map.values())
    final_corpus.sort(key=lambda x: x["doc_id"])
    selected_queries.sort(key=lambda x: x["query_id"])

    # Write outputs
    CORPUS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(CORPUS_OUTPUT, "w", encoding="utf-8") as f:
        for c in final_corpus:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    with open(QUERIES_OUTPUT, "w", encoding="utf-8") as f:
        for q in selected_queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    logger.info(
        "Successfully prepared benchmark: %d corpus passages written to %s, %d queries written to %s.",
        len(final_corpus), CORPUS_OUTPUT, len(selected_queries), QUERIES_OUTPUT
    )


if __name__ == "__main__":
    prepare_benchmark(max_passages=500, max_queries=150, seed=42)
