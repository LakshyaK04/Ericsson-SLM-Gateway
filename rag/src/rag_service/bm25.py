"""Okapi BM25 sparse keyword retriever and Reciprocal Rank Fusion (RRF).

Implements hybrid retrieval combining dense vector search (BGE-small bi-encoder)
with sparse lexical search (Okapi BM25) fused via Reciprocal Rank Fusion (RRF).
"""

from collections import Counter, defaultdict
import math
import re
from typing import Any, Dict, List, Optional, Tuple


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric words."""
    return re.findall(r"\w+", text.lower())


class BM25Index:
    """In-memory Okapi BM25 index over document chunks."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size: int = 0
        self.avgdl: float = 0.0
        self.doc_lengths: List[int] = []
        self.doc_ids: List[str] = []
        self.chunk_data: List[Dict[str, Any]] = []
        self.df: Counter = Counter()  # Document frequency for terms
        self.idf: Dict[str, float] = {}
        self.doc_term_freqs: List[Counter] = []

    def index_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Build or replace the BM25 index from a list of chunk dictionaries."""
        self.chunk_data = list(chunks)
        self.corpus_size = len(chunks)
        self.doc_ids = [c["chunk_id"] for c in chunks]
        self.doc_lengths = []
        self.doc_term_freqs = []
        self.df = Counter()
        self.idf = {}

        if self.corpus_size == 0:
            self.avgdl = 0.0
            return

        total_tokens = 0
        for chunk in chunks:
            tokens = tokenize(chunk.get("text", ""))
            length = len(tokens)
            self.doc_lengths.append(length)
            total_tokens += length

            term_freq = Counter(tokens)
            self.doc_term_freqs.append(term_freq)
            for term in term_freq.keys():
                self.df[term] += 1

        self.avgdl = total_tokens / self.corpus_size

        # Compute Robertson-Spärck Jones IDF with smoothing
        for term, freq in self.df.items():
            self.idf[term] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

    def search(
        self,
        query: str,
        top_k: int = 20,
        doc_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search the BM25 index for query terms and return scored candidates.

        Args:
            query: The search query string.
            top_k: Maximum candidate chunks to return.
            doc_ids: Optional document ID filter.

        Returns:
            List of chunk dicts augmented with 'bm25_score'.
        """
        if self.corpus_size == 0 or not query.strip():
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        allowed_doc_ids = set(doc_ids) if doc_ids else None
        scores: List[Tuple[int, float]] = []

        for i, chunk in enumerate(self.chunk_data):
            if allowed_doc_ids and chunk.get("doc_id") not in allowed_doc_ids:
                continue

            doc_len = self.doc_lengths[i]
            tf = self.doc_term_freqs[i]
            score = 0.0

            for q_term in query_tokens:
                if q_term not in tf:
                    continue
                term_count = tf[q_term]
                term_idf = self.idf.get(q_term, 0.0)
                numerator = term_count * (self.k1 + 1.0)
                denominator = term_count + self.k1 * (1.0 - self.b + self.b * (doc_len / (self.avgdl or 1.0)))
                score += term_idf * (numerator / denominator)

            if score > 0:
                scores.append((i, score))

        # Sort descending by BM25 score
        scores.sort(key=lambda x: x[1], reverse=True)
        top_results = scores[:top_k]

        results = []
        for idx, score in top_results:
            item = dict(self.chunk_data[idx])
            item["bm25_score"] = round(float(score), 4)
            results.append(item)

        return results


def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]],
    lexical_results: List[Dict[str, Any]],
    rrf_k: int = 60,
    top_k: int = 20,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
) -> List[Dict[str, Any]]:
    """Fuse dense vector rankings and sparse BM25 rankings using Reciprocal Rank Fusion (RRF).

    Formula: RRF_score(d) = dense_weight / (rrf_k + rank_dense(d)) + sparse_weight / (rrf_k + rank_sparse(d))

    Args:
        dense_results: Ranked chunk list from dense bi-encoder.
        lexical_results: Ranked chunk list from BM25 sparse search.
        rrf_k: Constant smoothing parameter (standard default 60).
        top_k: Number of fused results to return.
        dense_weight: Relative weight multiplier for dense ranks (default 1.0).
        sparse_weight: Relative weight multiplier for sparse ranks (default 1.0).

    Returns:
        List of deduplicated chunk dictionaries ordered by fused RRF score.
    """
    chunk_map: Dict[str, Dict[str, Any]] = {}
    rrf_scores: Dict[str, float] = defaultdict(float)

    # 1. Rank contributions from dense bi-encoder
    for rank, chunk in enumerate(dense_results, start=1):
        cid = chunk["chunk_id"]
        rrf_scores[cid] += float(dense_weight) / (rrf_k + rank)
        if cid not in chunk_map:
            chunk_map[cid] = dict(chunk)
        chunk_map[cid]["dense_rank"] = rank

    # 2. Rank contributions from lexical BM25
    for rank, chunk in enumerate(lexical_results, start=1):
        cid = chunk["chunk_id"]
        rrf_scores[cid] += float(sparse_weight) / (rrf_k + rank)
        if cid not in chunk_map:
            chunk_map[cid] = dict(chunk)
        chunk_map[cid]["bm25_score"] = chunk.get("bm25_score", 0.0)
        chunk_map[cid]["bm25_rank"] = rank

    # 3. Sort by fused score
    sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

    fused = []
    for cid in sorted_cids[:top_k]:
        item = chunk_map[cid]
        item["rrf_score"] = round(float(rrf_scores[cid]), 6)
        if "dense_score" not in item:
            item["dense_score"] = 0.0
        if "bm25_score" not in item:
            item["bm25_score"] = 0.0
        fused.append(item)

    return fused
