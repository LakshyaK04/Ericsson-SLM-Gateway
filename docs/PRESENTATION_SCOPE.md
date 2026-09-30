# Presentation Scope & Study Boundaries

This document defines the strict conceptual boundary for presenting this project to an internship mentor. Focus exclusively on the concepts in **Section A** and **Section B**. Do not bring up items in **Section C** or **Section D** unless specifically asked.

---

## A. MUST EXPLAIN (Core Concepts to Master)

You should be able to explain each of these clearly in 1–2 minutes:

1. **FastAPI Gateway (`gateway/src/slm_gateway/main.py`)**:
   - The central "front door" for all client requests.
   - Decouples client applications from model and RAG internals.
2. **OpenAI-Compatible Endpoint (`/v1/chat/completions`)**:
   - Follows standard OpenAI request/response schemas (`id`, `choices`, `usage`).
   - Any standard OpenAI client or tool can use this gateway simply by pointing `base_url="http://localhost:8000/v1"`.
3. **Phi-3 Mini In-Process (`microsoft/Phi-3-mini-4k-instruct`)**:
   - Small language model (3.8B parameters) loaded directly in Python memory via HuggingFace Transformers.
   - Avoids external cloud APIs or third-party server processes.
4. **4-Bit NF4 Quantization (`bitsandbytes`)**:
   - NormalFloat4 quantization reduces Phi-3's memory footprint from ~7.6 GB (FP16) down to ~2.6 GB VRAM.
   - Allows high-quality local inference on standard consumer/laptop GPUs.
5. **PII Redaction (`gateway/src/slm_gateway/pii.py`)**:
   - Scans incoming prompts using Microsoft Presidio and spaCy before any model sees them.
   - Replaces sensitive entities (`PERSON`, `EMAIL_ADDRESS`, `PHONE_NUMBER`, `CREDIT_CARD`, `IP_ADDRESS`, `EMPLOYEE_ID`) with typed placeholders (`<EMAIL_ADDRESS>`).
   - Intentionally excludes `LOCATION` and `DATE_TIME` to avoid corrupting normal queries ("What is the capital of Germany?").
6. **Dense Sentence Embeddings (`BAAI/bge-small-en-v1.5`)**:
   - 384-dimensional dense vectors used for both intent routing and document search.
   - Asymmetric query instruction prefixing aligns short search queries with document content.
7. **Semantic Intent Routing (`gateway/src/slm_gateway/router.py`)**:
   - Classifies query intent without an expensive LLM call or keyword matching.
   - Compares the query's embedding against curated exemplar sentences using cosine similarity.
   - Scores each intent by averaging its top-3 closest matches.
   - Three core intents: `general`, `technical`, and `rag`.
8. **PDF and DOCX Parsing (`rag/src/rag_service/parsers/`)**:
   - PyMuPDF (`fitz`) extracts clean digital text page by page from PDFs.
   - `python-docx` extracts headings, paragraphs, and tables from Word documents.
   - Early validation: Rejects empty or scanned image-only PDFs with a 400 error (OCR is not supported).
9. **Three Chunking Strategies (`rag/src/rag_service/chunking/`)**:
   - **Character**: Fixed-size sliding window (500 chars, 50-char overlap) snapping backwards to whitespace.
   - **Structure**: Heading- and paragraph-aware; preserves natural section boundaries and merges small paragraphs.
   - **Semantic**: Calculates sentence embeddings and splits when cosine similarity between consecutive sentences drops below a threshold.
10. **ChromaDB Vector Store (`rag/src/rag_service/store.py`)**:
    - Embedded, persistent vector database running locally on disk.
    - Maintains three isolated collections (`chunks_character`, `chunks_structure`, `chunks_semantic`) to evaluate strategies without cross-contamination.
11. **Two-Stage Retrieval (Top 20 Dense + Top 3 Re-ranked)**:
    - **Stage 1 (Dense Search)**: Fast vector search retrieves top 20 candidate chunks from ChromaDB (~10–25ms).
    - **Stage 2 (Cross-Encoder Re-Ranking)**: `BAAI/bge-reranker-base` analyzes `(query, passage)` pairs with deep cross-attention to accurately select the top 3 chunks (~140–200ms).
12. **Grounded Generation (`rag/src/rag_service/generation.py`)**:
    - Formats retrieved top-3 chunks into numbered context blocks (`[1]`, `[2]`, `[3]`).
    - Uses a strict system prompt directing Phi-3 to answer *only* from the context and cite sources.
13. **Loop Prevention (`X-Bypass-Router: true`)**:
    - When RAG calls Gateway `/v1/chat/completions` for final text generation, it passes `X-Bypass-Router: true`.
    - Gateway sees this header, skips intent classification, and generates directly via local Phi-3, preventing infinite recursive routing.

---

## B. SHOULD UNDERSTAND (Conceptual Context & Mechanics)

- **The Main Story**: "A local Phi-3 gateway with PII protection and semantic routing, connected to a document RAG pipeline with three chunking strategies, dense retrieval, and cross-encoder reranking."
- **Evaluation Takeaways**:
  - Router accuracy: 93.75% across 48 balanced test queries.
  - Chunking evaluation: Structure chunking with re-ranking performed best (100% Hit@1 on the 36-question test set).
  - Re-ranking did *not* improve character chunking Hit@1 (86.1% without vs 83.3% with), because severed sentences lack full context for cross-attention.
  - Honest dataset note: The evaluation corpus is small and synthetic (3 PDFs, 5 pages total from `scripts/create_eval_docs.py`).
- **Concurrency Guard**:
  - HuggingFace generation runs behind `asyncio.Semaphore(1)` so concurrent requests queue sequentially rather than crashing GPU VRAM.
- **Graceful Fallbacks**:
  - If a user asks a RAG query but no documents are uploaded, Gateway falls back to local Phi-3 generation with a warning in `x_routing["warning"]`.

---

## C. ONLY IF ASKED (Optional / Extension Material)

- **Docker Packaging**: Both services have Dockerfiles and can be launched together with `docker compose up --build`.
- **Model / Backend Swapping**: An optional `openai_compatible` backend exists in `backends/openai_compatible.py` to allow running against external mock servers or CPU development environments via `BACKEND=openai_compatible`.
- **Custom PII Pattern**: The regex for `EMPLOYEE_ID` is `\bEMP-\d{5,7}\b`.
- **Operating Threshold (0.55)**: Chosen from an empirical sweep between 0.30 and 0.80 where 0.55 yielded the highest classification accuracy.

---

## D. IGNORE FOR PRESENTATION (Implementation Details)

Do NOT bring up or spend presentation time explaining:
- Docker internal networking, named volumes, and health check retry parameters.
- GPU resource reservation syntax in Compose files.
- `pydantic-settings` environment loading precedence.
- Pytest fixture lifespans, tmp_path_factory, or mock setup.
- Internal schema definitions beyond standard OpenAI request/response.
- BM25 or sparse retrieval (this project implements dense vector retrieval + cross-encoder reranking).
- Removed features (API-key Bearer auth, `structured_json` intent, `PROJECT_CODENAME`).
