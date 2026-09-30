# Merged System Architecture: Local GenAI Stack

This document specifies the merged end-to-end architecture connecting the **SLM Gateway** (`slm_gateway`, port 8000) and the **Hybrid RAG Service** (`rag_service`, port 8001).

---

## 1. System Architecture Diagram

```mermaid
graph TD
    Client[Client / OpenAI SDK] -->|POST /v1/chat/completions| Gateway[SLM Gateway :8000]

    subgraph Gateway Subsystems
        PII[Presidio PII Redaction<br/>PERSON, EMAIL, PHONE, CC, IP, EMP-ID]
        Router[BGE Semantic Intent Router<br/>general | technical | rag]
        LocalModel[Phi-3-Mini 4k Model Backend<br/>In-process 4-bit NF4 / Semaphore]
    end

    Gateway --> PII
    PII --> Router
    Router -->|general / technical| LocalModel
    Router -->|rag intent| RAGClient[Gateway RAG Client]

    subgraph RAG Service :8001
        Ingestion[Document Parsers<br/>PyMuPDF PDF + python-docx DOCX]
        Chunking[Chunking Strategies<br/>Character | Structure | Semantic]
        VectorStore[(ChromaDB Persistent Collections)]
        DenseRetriever[BGE-Small Dense Retrieval<br/>Top-20 Candidates]
        Reranker[BGE-Reranker Cross-Encoder<br/>Top-3 Chunks]
        PromptBuilder[Grounded Context Builder<br/>[1], [2], [3] Citations]
    end

    RAGClient -->|POST /answer| PromptBuilder
    PromptBuilder --> DenseRetriever
    DenseRetriever --> VectorStore
    DenseRetriever --> Reranker
    Reranker --> PromptBuilder

    PromptBuilder -->|POST /v1/chat/completions<br/>X-Bypass-Router: true| LocalModel
    LocalModel -->|Tokens & Usage| Gateway
    Gateway -->|OpenAI-Compatible Chat Completion<br/>+ x_routing, x_pii, x_sources| Client
```

---

## 2. End-to-End Request Flow

1. **Client Request**: The client sends a standard OpenAI `POST /v1/chat/completions` request to the Gateway on port 8000.
2. **PII Masking**: The Gateway's Presidio pipeline intercepts incoming `user` messages, masking sensitive entities (`PERSON`, `EMAIL_ADDRESS`, `PHONE_NUMBER`, `CREDIT_CARD`, `IP_ADDRESS`, and custom `EMPLOYEE_ID`) into typed placeholders (e.g., `<EMAIL_ADDRESS>`), while intentionally preserving location and date nouns.
3. **Intent Classification**: Using `BAAI/bge-small-en-v1.5` embeddings, the Intent Router classifies the sanitized query against exemplar banks into `general`, `technical`, or `rag` (operating threshold: `0.55`).
4. **Branch A — Direct In-Process Generation (`general`, `technical`)**:
   - The Gateway invokes the local `microsoft/Phi-3-mini-4k-instruct` model (4-bit NF4 in PyTorch) behind an `asyncio.Semaphore(1)` to protect GPU memory.
   - The Gateway returns standard OpenAI JSON with `x_routing` and `x_pii` metadata.
5. **Branch B — Grounded RAG Query (`rag`)**:
   - The Gateway checks if documents are indexed via `GET http://rag:8001/documents`.
   - If indexed, the Gateway calls `POST http://rag:8001/answer`.
   - The RAG service performs two-stage retrieval: ChromaDB retrieves the top 20 dense candidates, and `BAAI/bge-reranker-base` re-ranks them to the top 3 chunks.
   - The RAG service formats a grounded prompt with bracketed context citations (`[1]`, `[2]`, `[3]`) and sends it back to the Gateway `/v1/chat/completions` with header `X-Bypass-Router: true`.
   - The Gateway sees `X-Bypass-Router: true`, bypasses the router, runs local model generation, and returns the response to the RAG service.
   - The Gateway packages the final response to the client, attaching `x_sources` containing chunk IDs, source document names, page numbers, and both dense and re-rank scores.
6. **Graceful Fallback**: If zero documents are indexed or the RAG service is unreachable, the Gateway automatically falls back to local Phi-3 generation without error, setting a clear notice in `x_routing["warning"]`.

---

## 3. Service Responsibilities Summary

| Capability | SLM Gateway (:8000) | RAG Service (:8001) | Rationale |
|---|:---:|:---:|---|
| **API Entry Point** | Yes (Single Front Door) | No | Clients interact with one standard OpenAI endpoint. |
| **LLM Weight Hosting** | Yes (Phi-3-mini 4-bit NF4) | No | Prevents duplicating model weights in GPU VRAM. |
| **Privacy / PII Masking** | Yes (Presidio + spaCy) | No | Scrubs sensitive text before search or generation. |
| **Intent Routing** | Yes (BGE embeddings) | No | Decides if document retrieval is required. |
| **Document Parsers** | No | Yes (PDF & DOCX) | Handles digital text parsing and validation. |
| **Chunking & Indexing** | No | Yes (3 strategies) | Character, Structure, and Semantic chunking. |
| **Vector Storage** | No | Yes (ChromaDB) | Persistent HNSW collections on disk. |
| **Two-Stage Retrieval** | No | Yes (Dense + Cross-Encoder) | Top-20 candidate search followed by Top-3 re-ranking. |
