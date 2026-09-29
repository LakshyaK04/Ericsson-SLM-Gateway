import asyncio

from src.phi3_project.rag import RAGPipeline


async def main():
    rag = RAGPipeline(
        "data/ericsson_rag_sample.pdf"
    )

    query = "How does RAG retrieve information from documents?"

    result = await rag.generate(query)

    print("\n========== ANSWER ==========\n")
    print(result["answer"])

    print("\n========== SOURCES ==========\n")

    for i, source in enumerate(result["sources"], start=1):
        print(f"\n--- SOURCE {i} ---")
        print("Score:", source["reranker_score"])
        print(source["text"])


if __name__ == "__main__":
    asyncio.run(main())