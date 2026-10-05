"""
Large-Scale Retrieval Benchmark Script for Hybrid RAG Stack.

Evaluates retrieval configurations across a 500-passage / 150-query slice of SQuAD v2.0:
1. BM25 Sparse Lexical Only
2. BGE Dense Vector Only (bge-small-en-v1.5)
3. Hybrid RRF (standard k=60)
4. Hybrid RRF (sensitivity k=20)
5. Hybrid RRF (sensitivity k=100)
6. Hybrid Weighted RRF (Dense-Heavy: 0.7 Dense / 0.3 Sparse)
7. Hybrid Weighted RRF (Sparse-Heavy: 0.3 Dense / 0.7 Sparse)
8. Hybrid (RRF k=60) + Cross-Encoder Re-Ranking (bge-reranker-base)

Calculates:
- Hit@1 (%)
- Hit@3 (%)
- Hit@10 (%)
- MRR (Mean Reciprocal Rank)
- nDCG@10
- Mean Latency (ms)
- P95 Latency (ms)
- Hardware metadata (CPU, GPU, Device)
"""

import csv
import json
import logging
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from typing import Any, Dict, List, Optional
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark_retrieval")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "rag" / "src"))

from rag_service.bm25 import BM25Index, reciprocal_rank_fusion
from rag_service.chunking.base import Chunk
from rag_service.config import Settings
from rag_service.embeddings import EmbeddingModel
from rag_service.reranker import Reranker
from rag_service.retriever import Retriever
from rag_service.store import ChromaStore

DATASETS_DIR = REPO_ROOT / "eval" / "datasets"
RESULTS_DIR = REPO_ROOT / "eval" / "results"
CORPUS_FILE = DATASETS_DIR / "squad_retrieval_corpus.jsonl"
QUERIES_FILE = DATASETS_DIR / "squad_retrieval_queries.jsonl"


def get_hardware_info() -> Dict[str, str]:
    """Capture precise runtime hardware details."""
    cpu_info = platform.processor() or platform.machine()
    os_info = f"{platform.system()} {platform.release()} (v{platform.version()})"
    cuda_avail = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "None (CPU only)"
    device = "cuda" if cuda_avail else "cpu"
    return {
        "platform": os_info,
        "cpu": cpu_info,
        "device": device,
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
    }


def compute_ndcg_at_k(found_rank: Optional[int], k: int = 10) -> float:
    """Compute nDCG@k for a single relevant document."""
    if found_rank is None or found_rank > k:
        return 0.0
    # DCG = 1.0 / log2(rank + 1), IDCG for 1 relevant document is 1.0 / log2(1 + 1) = 1.0
    return 1.0 / math.log2(found_rank + 1)


def run_benchmark():
    if not CORPUS_FILE.exists() or not QUERIES_FILE.exists():
        logger.info("Corpus or queries file missing, running prepare_benchmark.py...")
        from eval.prepare_benchmark import prepare_benchmark
        prepare_benchmark(max_passages=500, max_queries=150, seed=42)

    # 1. Load data
    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        corpus = [json.loads(line) for line in f if line.strip()]

    with open(QUERIES_FILE, "r", encoding="utf-8") as f:
        queries = [json.loads(line) for line in f if line.strip()]

    hw = get_hardware_info()
    logger.info("Loaded %d corpus passages and %d queries.", len(corpus), len(queries))
    logger.info("Hardware detected: %s | GPU: %s | Device: %s", hw["cpu"], hw["gpu_name"], hw["device"])

    # 2. Setup isolated Chroma directory
    eval_db_dir = REPO_ROOT / "data" / "chroma_squad_benchmark"
    if eval_db_dir.exists():
        shutil.rmtree(eval_db_dir, ignore_errors=True)

    cfg = Settings(
        CHROMA_PERSIST_DIR=str(eval_db_dir),
        EMBEDDING_MODEL_NAME="BAAI/bge-small-en-v1.5",
        RERANKER_MODEL_NAME="BAAI/bge-reranker-base",
        DEFAULT_STRATEGY="structure",
    )

    logger.info("Loading models...")
    emb_model = EmbeddingModel(cfg.EMBEDDING_MODEL_NAME)
    reranker = Reranker(cfg.RERANKER_MODEL_NAME)
    store = ChromaStore(persist_dir=str(eval_db_dir))
    retriever = Retriever(store=store, embedding_model=emb_model, reranker=reranker, config=cfg)

    # 3. Index corpus into isolated store
    logger.info("Encoding and indexing %d passages into ChromaDB...", len(corpus))
    chunks = [
        Chunk(
            chunk_id=c["doc_id"],
            text=c["text"],
            source=c.get("title", "squad"),
            page=1,
            strategy="structure",
            metadata={"doc_id": c["doc_id"], "title": c.get("title", "")},
        )
        for c in corpus
    ]

    t0_index = time.perf_counter()
    batch_size = 64
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_texts = [c.text for c in batch_chunks]
        batch_embeddings = emb_model.encode_documents(batch_texts)
        store.add_chunks(
            strategy="structure",
            chunks=batch_chunks,
            embeddings=batch_embeddings,
            doc_id="squad_benchmark_corpus",
        )
    index_time = round(time.perf_counter() - t0_index, 2)
    logger.info("Indexed %d passages in %.2fs.", len(chunks), index_time)

    # 4. Benchmark Configurations
    configs = [
        {
            "name": "BM25 Sparse Lexical Only",
            "mode": "sparse",
            "use_reranker": False,
            "rrf_k": 60,
            "dense_weight": 1.0,
            "sparse_weight": 1.0,
        },
        {
            "name": "BGE Dense Vector Only",
            "mode": "dense",
            "use_reranker": False,
            "rrf_k": 60,
            "dense_weight": 1.0,
            "sparse_weight": 1.0,
        },
        {
            "name": "Hybrid RRF (k=60)",
            "mode": "hybrid",
            "use_reranker": False,
            "rrf_k": 60,
            "dense_weight": 1.0,
            "sparse_weight": 1.0,
        },
        {
            "name": "Hybrid RRF (k=20)",
            "mode": "hybrid",
            "use_reranker": False,
            "rrf_k": 20,
            "dense_weight": 1.0,
            "sparse_weight": 1.0,
        },
        {
            "name": "Hybrid RRF (k=100)",
            "mode": "hybrid",
            "use_reranker": False,
            "rrf_k": 100,
            "dense_weight": 1.0,
            "sparse_weight": 1.0,
        },
        {
            "name": "Hybrid Weighted RRF (Dense 0.7 / Sparse 0.3)",
            "mode": "hybrid",
            "use_reranker": False,
            "rrf_k": 60,
            "dense_weight": 0.7,
            "sparse_weight": 0.3,
        },
        {
            "name": "Hybrid Weighted RRF (Dense 0.3 / Sparse 0.7)",
            "mode": "hybrid",
            "use_reranker": False,
            "rrf_k": 60,
            "dense_weight": 0.3,
            "sparse_weight": 0.7,
        },
        {
            "name": "Hybrid + Cross-Encoder Re-Ranking",
            "mode": "hybrid",
            "use_reranker": True,
            "rrf_k": 60,
            "dense_weight": 1.0,
            "sparse_weight": 1.0,
        },
    ]

    benchmark_rows = []
    n_queries = len(queries)

    for conf in configs:
        conf_name = conf["name"]
        mode = conf["mode"]
        use_reranker = conf["use_reranker"]
        rrf_k = conf["rrf_k"]
        dw = conf["dense_weight"]
        sw = conf["sparse_weight"]

        hits_at_1 = 0
        hits_at_3 = 0
        hits_at_10 = 0
        reciprocal_ranks = []
        ndcg_scores = []
        latencies = []

        logger.info("Benchmarking configuration: %s...", conf_name)

        for q_item in queries:
            q_text = q_item["query"]
            gold_id = q_item["gold_doc_id"]

            t0 = time.perf_counter()
            results = retriever.retrieve(
                query=q_text,
                strategy="structure",
                retrieve_k=20,
                final_k=10,
                use_reranker=use_reranker,
                retrieval_mode=mode,
                rrf_k=rrf_k,
                dense_weight=dw,
                sparse_weight=sw,
            )
            lat_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat_ms)

            # Determine rank of gold_doc_id
            found_rank = None
            for idx, r in enumerate(results):
                if r.chunk_id == gold_id:
                    found_rank = idx + 1
                    break

            if found_rank == 1:
                hits_at_1 += 1
                hits_at_3 += 1
                hits_at_10 += 1
                reciprocal_ranks.append(1.0)
            elif found_rank and found_rank <= 3:
                hits_at_3 += 1
                hits_at_10 += 1
                reciprocal_ranks.append(1.0 / found_rank)
            elif found_rank and found_rank <= 10:
                hits_at_10 += 1
                reciprocal_ranks.append(1.0 / found_rank)
            else:
                reciprocal_ranks.append(0.0)

            ndcg_scores.append(compute_ndcg_at_k(found_rank, k=10))

        hit1_pct = round((hits_at_1 / n_queries) * 100.0, 2)
        hit3_pct = round((hits_at_3 / n_queries) * 100.0, 2)
        hit10_pct = round((hits_at_10 / n_queries) * 100.0, 2)
        mrr = round(float(np.mean(reciprocal_ranks)), 4)
        ndcg10 = round(float(np.mean(ndcg_scores)), 4)
        mean_lat = round(float(np.mean(latencies)), 2)
        p95_lat = round(float(np.percentile(latencies, 95)), 2)

        logger.info(
            "[%s] Hit@1: %.2f%% | Hit@3: %.2f%% | Hit@10: %.2f%% | MRR: %.4f | nDCG@10: %.4f | Lat: %.2fms (p95: %.2fms)",
            conf_name, hit1_pct, hit3_pct, hit10_pct, mrr, ndcg10, mean_lat, p95_lat
        )

        benchmark_rows.append({
            "Configuration": conf_name,
            "Mode": mode,
            "Re-Ranker": "On (bge-reranker-base)" if use_reranker else "Off",
            "RRF k": rrf_k,
            "Weights (Dense/Sparse)": f"{dw:.1f} / {sw:.1f}" if mode == "hybrid" else "N/A",
            "Hit@1 (%)": hit1_pct,
            "Hit@3 (%)": hit3_pct,
            "Hit@10 (%)": hit10_pct,
            "MRR": mrr,
            "nDCG@10": ndcg10,
            "Mean Latency (ms)": mean_lat,
            "P95 Latency (ms)": p95_lat,
        })

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "retrieval_benchmark.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(benchmark_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(benchmark_rows)

    md_path = RESULTS_DIR / "retrieval_benchmark.md"
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Empirical Retrieval & Re-Ranking Benchmark Report\n\n")
        f.write(f"- **Benchmark Run Date:** {timestamp}\n")
        f.write(f"- **Dataset:** SQuAD v2.0 Public Slice ({len(corpus)} passages, {n_queries} queries)\n")
        f.write(f"- **Dense Bi-Encoder:** `BAAI/bge-small-en-v1.5` (384-dim, normalized)\n")
        f.write(f"- **Sparse Lexical Engine:** Okapi BM25 ($k_1=1.5, b=0.75$)\n")
        f.write(f"- **Neural Cross-Encoder:** `BAAI/bge-reranker-base`\n")
        f.write(f"- **Hardware Platform:** {hw['platform']}\n")
        f.write(f"- **CPU:** {hw['cpu']}\n")
        f.write(f"- **GPU / Device:** {hw['gpu_name']} (`{hw['device']}`)\n")
        f.write(f"- **PyTorch Version:** {hw['torch_version']}\n\n")

        f.write("---\n\n## 1. Retrieval Performance Matrix\n\n")
        f.write("| Configuration | Re-Ranker | RRF $k$ | Dense / Sparse Weights | Hit@1 (%) | Hit@3 (%) | Hit@10 (%) | MRR | nDCG@10 | Mean Latency (ms) | P95 Latency (ms) |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for r in benchmark_rows:
            f.write(
                f"| **{r['Configuration']}** | {r['Re-Ranker']} | {r['RRF k']} | {r['Weights (Dense/Sparse)']} | "
                f"**{r['Hit@1 (%)']}%** | {r['Hit@3 (%)']}% | {r['Hit@10 (%)']}% | "
                f"**{r['MRR']}** | **{r['nDCG@10']}** | {r['Mean Latency (ms)']} ms | {r['P95 Latency (ms)']} ms |\n"
            )

        f.write("\n---\n\n## 2. In-Depth Empirical Analysis & Honest Insights\n\n")

        best_hit1 = max(benchmark_rows, key=lambda x: x["Hit@1 (%)"])
        bm25_row = next(r for r in benchmark_rows if "BM25" in r["Configuration"])
        dense_row = next(r for r in benchmark_rows if "Dense" in r["Configuration"] and "Hybrid" not in r["Configuration"])
        hybrid_row = next(r for r in benchmark_rows if r["Configuration"] == "Hybrid RRF (k=60)")
        rerank_row = next(r for r in benchmark_rows if "Cross-Encoder" in r["Configuration"])

        f.write("### 2.1 Dense vs. Sparse Modality Comparison\n")
        f.write(
            f"- **BGE Dense Vector Search** achieved **{dense_row['Hit@1 (%)']}% Hit@1** and **{dense_row['MRR']} MRR**, "
            f"compared to **{bm25_row['Hit@1 (%)']}% Hit@1** and **{bm25_row['MRR']} MRR** for **BM25 Sparse Search**.\n"
            "- Dense embeddings excel on conceptual questions where queries do not share verbatim tokens with passages, "
            "whereas BM25 performs strongly on exact keyword, entity name, and numeric constraints.\n\n"
        )

        f.write("### 2.2 Impact of Reciprocal Rank Fusion (RRF) & Parameter Sensitivity\n")
        f.write(
            f"- Standard Hybrid RRF ($k=60$) achieved **{hybrid_row['Hit@1 (%)']}% Hit@1** and **{hybrid_row['MRR']} MRR**.\n"
            "- Comparing $k=20$, $k=60$, and $k=100$: On this 500-passage corpus, varying $k$ produces subtle rank changes. "
            "Lower $k=20$ concentrates fusion score on rank-1/rank-2 positions, while higher $k=100$ dampens rank decay.\n"
            "- **Weighted RRF**: Tuning dense weight to 0.7 and sparse to 0.3 demonstrates how prioritizing dense semantics "
            "impacts balance across varied question styles.\n\n"
        )

        f.write("### 2.3 Cross-Encoder Re-Ranking Tradeoff\n")
        f.write(
            f"- Hybrid + Cross-Encoder Re-Ranking achieved **{rerank_row['Hit@1 (%)']}% Hit@1**, "
            f"**{rerank_row['Hit@3 (%)']}% Hit@3**, and **{rerank_row['MRR']} MRR** with **nDCG@10 of {rerank_row['nDCG@10']}**.\n"
            f"- **Latency Tradeoff:** Re-ranking adds neural cross-attention overhead: "
            f"{rerank_row['Mean Latency (ms)']} ms mean vs {hybrid_row['Mean Latency (ms)']} ms for pure hybrid search. "
            "In latency-critical SLAs (<20ms), pure Hybrid RRF is often preferred; in high-accuracy applications, cross-encoder re-ranking provides highest precision.\n"
        )

    # Cleanup temporary Chroma directory
    if eval_db_dir.exists():
        shutil.rmtree(eval_db_dir, ignore_errors=True)

    logger.info("Benchmark complete! Reports saved to %s and %s.", md_path, csv_path)
    return benchmark_rows


if __name__ == "__main__":
    run_benchmark()
