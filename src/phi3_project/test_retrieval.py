from src.phi3_project.ingestion import (
    extract_text_from_pdf,
    clean_text,
)
from src.phi3_project.chunking import chunk_text
from src.phi3_project.embeddings import EmbeddingModel
from src.phi3_project.vector_store import VectorStore


# 1. Load and clean PDF
text = extract_text_from_pdf("data/ericsson_rag_sample.pdf")
text = clean_text(text)

# 2. Split into chunks
chunks = chunk_text(text)

print("Number of chunks:", len(chunks))

# 3. Create embeddings
embedding_model = EmbeddingModel()
chunk_embeddings = embedding_model.encode(chunks)

print("Embedding shape:", chunk_embeddings.shape)

# 4. Create vector store
vector_store = VectorStore(
    dimension=chunk_embeddings.shape[1]
)

# 5. Store chunks and embeddings
vector_store.add(
    chunk_embeddings,
    chunks,
)

# 6. Search
query = "How does RAG retrieve information from documents?"

query_embedding = embedding_model.encode([query])

results = vector_store.search(
    query_embedding,
    top_k=2,
)

# 7. Display results
print("\nQuery:", query)

for i, result in enumerate(results, start=1):
    print(f"\n--- RESULT {i} ---")
    print("Score:", result["score"])
    print("Text:", result["text"])