import httpx

from src.phi3_project.ingestion import (
    extract_text_from_pdf,
    clean_text,
)
from src.phi3_project.chunking import chunk_text
from src.phi3_project.embeddings import EmbeddingModel
from src.phi3_project.vector_store import VectorStore


class RAGPipeline:
    def __init__(self, pdf_path: str):
        # 1. Load PDF
        text = extract_text_from_pdf(pdf_path)
        text = clean_text(text)

        # 2. Create chunks
        self.chunks = chunk_text(text)

        # 3. Create embeddings
        self.embedding_model = EmbeddingModel()
        embeddings = self.embedding_model.encode(self.chunks)

        # 4. Create vector store
        self.vector_store = VectorStore(
            dimension=embeddings.shape[1]
        )

        # 5. Store embeddings + chunks
        self.vector_store.add(
            embeddings,
            self.chunks,
        )

    def retrieve(self, query: str, top_k: int = 3):
        query_embedding = self.embedding_model.encode([query])

        return self.vector_store.search(
            query_embedding,
            query,
            top_k=top_k,
            candidate_k=5,
        )

    async def generate(self, query: str):
        # Retrieve relevant document chunks
        results = self.retrieve(query)

        # Build context
        context = "\n\n".join(
            result["text"]
            for result in results
        )

        prompt = f"""
Answer the user's question using the provided document context.

Document context:
{context}

User question:
{query}

Answer based on the document context. If the answer
cannot be found in the context, say that the document
does not provide enough information.
"""

        request = {
            "model": "phi-3-mini",
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": 0.2,
            "max_tokens": 300,
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                "http://localhost:8000/v1/chat/completions",
                json=request,
            )

        response.raise_for_status()

        result = response.json()

        return {
            "answer": result["choices"][0]["message"]["content"],
            "sources": results,
        }