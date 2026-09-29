import faiss
import numpy as np
from sentence_transformers import CrossEncoder


class VectorStore:
    def __init__(self, dimension: int):
        self.index = faiss.IndexFlatIP(dimension)
        self.documents = []

        self.reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    def add(self, embeddings, documents):
        embeddings = np.asarray(
            embeddings,
            dtype="float32",
        )

        self.index.add(embeddings)
        self.documents.extend(documents)

    def search(
        self,
        query_embedding,
        query: str,
        top_k: int = 3,
        candidate_k: int = 5,
    ):
        query_embedding = np.asarray(
            query_embedding,
            dtype="float32",
        )

        # Step 1: Fast FAISS retrieval
        scores, indices = self.index.search(
            query_embedding,
            min(candidate_k, len(self.documents)),
        )

        candidates = []

        for score, index in zip(scores[0], indices[0]):
            if index != -1:
                candidates.append({
                    "text": self.documents[index],
                    "faiss_score": float(score),
                })

        # Step 2: Cross-encoder reranking
        pairs = [
            (query, candidate["text"])
            for candidate in candidates
        ]

        reranker_scores = self.reranker.predict(pairs)

        for candidate, score in zip(
            candidates,
            reranker_scores,
        ):
            candidate["reranker_score"] = float(score)

        # Step 3: Sort by reranker score
        candidates.sort(
            key=lambda x: x["reranker_score"],
            reverse=True,
        )

        # Step 4: Return top-k
        return candidates[:top_k]