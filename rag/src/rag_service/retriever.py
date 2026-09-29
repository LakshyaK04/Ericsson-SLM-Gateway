"""RAG retrieval pipeline.

Loads a PDF, chunks it, embeds the chunks, stores them in a
vector store, and provides retrieval with cross-encoder reranking.
"""

import logging

from rag_service.parsers import extract_text_from_pdf, clean_text
from rag_service.chunking import chunk_text
from rag_service.embeddings import EmbeddingModel
from rag_service.store import VectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    """End-to-end retrieval pipeline for a single PDF document."""

    def __init__(
        self,
        pdf_path: str,
        chunking_strategy: str = "character",
    ):
        # 1. Load and clean the PDF text
        text = extract_text_from_pdf(pdf_path)
        text = clean_text(text)

        # 2. Create chunks using the selected strategy
        self.chunks = chunk_text(text, chunking_strategy)

        # 3. Create embeddings for all chunks
        self.embedding_model = EmbeddingModel()
        embeddings = self.embedding_model.encode(self.chunks)

        # 4. Build the vector store and index the chunks
        self.vector_store = VectorStore(
            dimension=embeddings.shape[1]
        )
        self.vector_store.add(embeddings, self.chunks)

    def retrieve(self, query: str, top_k: int = 3):
        """Retrieve the top-k most relevant chunks for a query."""
        query_embedding = self.embedding_model.encode([query])

        return self.vector_store.search(
            query_embedding,
            query,
            top_k=top_k,
            candidate_k=5,
        )