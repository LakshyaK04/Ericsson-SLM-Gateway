"""Persistent ChromaDB vector store with one collection per chunking strategy.

Per Section 5.5:
- One collection per strategy: chunks_character, chunks_structure, chunks_semantic
- Metadata: doc_id, source, page, chunk_id, strategy
- Persisted on disk so service restarts retain all index data.
"""

from collections import defaultdict
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import chromadb
from chromadb.config import Settings as ChromaSettings

from .chunking.base import Chunk
from .config import Settings, settings

logger = logging.getLogger(__name__)

STRATEGY_COLLECTION_MAP = {
    "character": "chunks_character",
    "structure": "chunks_structure",
    "semantic": "chunks_semantic",
}


class ChromaStore:
    """Enterprise vector store managing per-strategy ChromaDB collections."""

    def __init__(self, persist_dir: Optional[Union[str, Path]] = None):
        self.persist_dir = Path(persist_dir or settings.CHROMA_PERSIST_DIR)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Initializing ChromaDB PersistentClient at %s...", self.persist_dir)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collections: Dict[str, chromadb.Collection] = {}

        # Initialize collections for each strategy with cosine distance
        for strategy, col_name in STRATEGY_COLLECTION_MAP.items():
            self.collections[strategy] = self.client.get_or_create_collection(
                name=col_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "Chroma collection '%s' ready with %d items.",
                col_name,
                self.collections[strategy].count(),
            )

    def add_chunks(
        self,
        strategy: str,
        chunks: List[Chunk],
        embeddings: List[List[float]],
        doc_id: str,
    ) -> int:
        """Add chunks and embeddings to the collection for the specified strategy."""
        if strategy not in self.collections:
            raise ValueError(f"Unknown strategy: '{strategy}'. Supported: {list(self.collections.keys())}")

        if not chunks:
            return 0

        col = self.collections[strategy]
        ids = [c.chunk_id for c in chunks]
        texts = [c.text for c in chunks]
        metadatas = [
            {
                "doc_id": doc_id,
                "source": c.source,
                "page": int(c.page),
                "chunk_id": c.chunk_id,
                "strategy": strategy,
            }
            for c in chunks
        ]

        col.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return len(chunks)

    def query(
        self,
        strategy: str,
        query_embedding: List[float],
        n_results: int = 20,
        doc_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Query the vector collection for the given strategy.

        Returns list of matching chunks with dense cosine similarity scores (1 - distance).
        """
        if strategy not in self.collections:
            raise ValueError(f"Unknown strategy: '{strategy}'. Supported: {list(self.collections.keys())}")

        col = self.collections[strategy]
        col_count = col.count()
        if col_count == 0:
            return []

        limit = min(n_results, col_count)
        where_filter = None
        if doc_ids:
            if len(doc_ids) == 1:
                where_filter = {"doc_id": doc_ids[0]}
            else:
                where_filter = {"doc_id": {"$in": doc_ids}}

        results = col.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        matches = []
        if results and results["ids"] and results["ids"][0]:
            ids = results["ids"][0]
            docs = results["documents"][0] if results["documents"] else [""] * len(ids)
            metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(ids)
            dists = results["distances"][0] if results["distances"] else [0.0] * len(ids)

            for c_id, doc, meta, dist in zip(ids, docs, metas, dists):
                # Cosine space: distance in [0, 2], similarity = 1 - (dist / 2) or 1 - dist
                dense_score = max(0.0, float(1.0 - dist))
                matches.append({
                    "chunk_id": c_id,
                    "text": doc,
                    "source": meta.get("source", ""),
                    "page": meta.get("page", 1),
                    "strategy": meta.get("strategy", strategy),
                    "dense_score": round(dense_score, 4),
                    "doc_id": meta.get("doc_id", ""),
                })

        return matches

    def list_documents(self) -> List[Dict[str, Any]]:
        """Return summary of all indexed documents across collections."""
        doc_summary: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"source": "", "strategies": set(), "total_chunks": 0}
        )

        for strat, col in self.collections.items():
            col_data = col.get(include=["metadatas"])
            if col_data and col_data["metadatas"]:
                for m in col_data["metadatas"]:
                    d_id = m.get("doc_id")
                    if d_id:
                        doc_summary[d_id]["source"] = m.get("source", "")
                        doc_summary[d_id]["strategies"].add(strat)
                        doc_summary[d_id]["total_chunks"] += 1

        docs = []
        for d_id, data in doc_summary.items():
            docs.append({
                "doc_id": d_id,
                "source": data["source"],
                "strategies": sorted(list(data["strategies"])),
                "total_chunks": data["total_chunks"],
            })
        return docs

    def delete_document(self, doc_id: str) -> int:
        """Delete all chunks for a document from all collections."""
        total_deleted = 0
        for strat, col in self.collections.items():
            before_count = col.count()
            col.delete(where={"doc_id": doc_id})
            after_count = col.count()
            total_deleted += (before_count - after_count)
        return total_deleted

    def get_all_chunks(self, strategy: str) -> List[Dict[str, Any]]:
        """Retrieve all indexed chunks and metadata for a specific strategy."""
        if strategy not in self.collections:
            return []
        col = self.collections[strategy]
        if col.count() == 0:
            return []
        data = col.get(include=["documents", "metadatas"])
        chunks = []
        if data and data["ids"]:
            for cid, doc, meta in zip(data["ids"], data["documents"], data["metadatas"]):
                chunks.append({
                    "chunk_id": cid,
                    "text": doc,
                    "source": meta.get("source", ""),
                    "page": meta.get("page", 1),
                    "strategy": meta.get("strategy", strategy),
                    "doc_id": meta.get("doc_id", ""),
                })
        return chunks

    def count(self, strategy: Optional[str] = None) -> int:
        """Return total chunks in specified strategy or all collections."""
        if strategy:
            return self.collections[strategy].count() if strategy in self.collections else 0
        return sum(c.count() for c in self.collections.values())


_store: Optional[ChromaStore] = None


def get_chroma_store(cfg: Optional[Settings] = None) -> ChromaStore:
    """Return or initialize the singleton ChromaStore."""
    global _store
    persist_dir = cfg.CHROMA_PERSIST_DIR if cfg else settings.CHROMA_PERSIST_DIR
    if _store is None or str(_store.persist_dir) != str(persist_dir):
        _store = ChromaStore(persist_dir)
    return _store