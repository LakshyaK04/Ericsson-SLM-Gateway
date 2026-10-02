# 10-Minute Mentor Presentation Script

This script provides an exact 12-step presentation walkthrough for your internship mentor. It connects the live software demonstration with clean, plain-English talking points.

---

## Presentation Checklist
- **Terminal 1**: `uv run uvicorn rag_service.main:app --host 0.0.0.0 --port 8001`
- **Terminal 2**: `uv run uvicorn slm_gateway.main:app --host 0.0.0.0 --port 8000`
- **Terminal 3**: For running curl commands or `uv run python scripts/demo.py`

---

## 1. Problem (30 Seconds)
**Talking Point**:
> *"Enterprise teams want to use generative AI for internal documentation and developer assistance. However, sending proprietary documents and sensitive employee data to external cloud APIs creates severe privacy and compliance risks. Our objective was to build a 100% local, self-hosted GenAI stack that runs open-weights models in-process on local hardware, strips sensitive PII before inference, and grounds answers in internal technical documents."*

---

## 2. Architecture (1 Minute)
**Talking Point** (Refer to the simple architecture diagram):
```
Client  -->  FastAPI Gateway  -->  PII Redaction  -->  Semantic Router
                                                          /          \
                                              Normal query            RAG query
                                                   │                      │
                                             Phi-3 Mini              RAG Pipeline
                                                                     (Chroma + BGE)
                                                                          │
                                                                     Phi-3 Mini
                                                                          │
                                                                       Answer
```
> *"The system is split into two focused services. The SLM Gateway on port 8000 is our front door. It exposes an OpenAI-compatible `/v1/chat/completions` endpoint, scrubs incoming prompts of PII using Microsoft Presidio and spaCy, and routes queries using BGE embeddings. If the query is conversational or technical, it generates an answer directly using a local 4-bit quantized Phi-3 Mini. If the user asks about an uploaded document, it delegates to our RAG service on port 8001."*

---

## 3. Normal Query Demo (1 Minute)
**Command**:
```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is the capital of Sweden?"}],
    "temperature": 0.7,
    "max_tokens": 64
  }'
```
**Talking Point**:
> *"Here we send a standard chat completion request. The Gateway responds with the standard OpenAI format (`id`, `choices`, `usage`). Any OpenAI Python client or tool works out of the box. In the response metadata, `x_routing` shows that the query was classified as `general` with high confidence, routing directly to local Phi-3."*

---

## 4. PII Redaction Example (1 Minute)
**Command**:
```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{
      "role": "user",
      "content": "Hello, I am Alice Smith (EMP-84920) working on Project Phoenix. My email is alice.smith@ericsson.com and phone is +46-8-555-1234. I work at the headquarters in Stockholm."
    }],
    "max_tokens": 64
  }'
```
**Talking Point**:
> *"Notice that before Phi-3 sees the prompt, Presidio masks Alice Smith as `<PERSON>`, the email as `<EMAIL_ADDRESS>`, the phone number as `<PHONE_NUMBER>`, our custom employee ID regex as `<EMPLOYEE_ID>`, and our configurable project deny-list as `<PROJECT_CODENAME>`. The response metadata confirms `x_pii: {"redactions": 5}`. Notice also that 'Stockholm' remains unmasked: we deliberately excluded `LOCATION` and `DATE_TIME` so factual geographical queries aren't mutilated."*

---

## 5. Router Example (1 Minute)
**Talking Point**:
> *"Instead of using an expensive LLM call or brittle keyword matching, our Intent Router uses `BAAI/bge-small-en-v1.5` embeddings. We have a bank of curated example queries across three classes: `general`, `technical`, and `rag`. For each incoming query, we compute its cosine similarity against the exemplars and average the top 3 matches. If similarity is above our tuned threshold of 0.55, it routes to that intent; otherwise it safely defaults to `general`."*

---

## 6. Upload Document (1 Minute)
**Command**:
```bash
curl -s -X POST http://localhost:8001/documents \
  -F "file=@eval/docs/ericsson_rag_sample.pdf"
```
**Talking Point**:
> *"Now we upload a technical specification to the RAG service. The PDF parser uses PyMuPDF to extract clean text page by page. Notice that if someone uploads an empty or scanned image-only PDF, our parser immediately rejects it with HTTP 400 because OCR is not supported. The document is chunked across all three strategies and indexed into local ChromaDB collections."*

---

## 7. RAG Query Demo (1 Minute)
**Command**:
```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{
      "role": "user",
      "content": "According to the uploaded documentation, what capabilities does the Ericsson AI Platform provide?"
    }],
    "max_tokens": 128
  }'
```
**Talking Point**:
> *"We send this question to the Gateway front door. The semantic router identifies this as a `rag` intent and passes the request to the RAG service."*

---

## 8. Explain Retrieval (1 Minute)
**Talking Point**:
> *"To retrieve relevant context efficiently, we use a two-stage retrieval pipeline. Stage 1 is fast dense vector retrieval. We embed the search query using BGE-small with an asymmetric task prompt and query ChromaDB for the top 20 candidate chunks. This casts a wide net over the document in just ~10 to 25 milliseconds."*

---

## 9. Explain Re-Ranking (1 Minute)
**Talking Point**:
> *"Dense retrieval can return chunks that share keywords or topics but don't actually answer the question. In Stage 2, we pass those 20 candidate chunks through a cross-encoder model, `BAAI/bge-reranker-base`. Unlike bi-encoders that compare precomputed vectors, the cross-encoder feeds the query and candidate chunk together through full transformer cross-attention, selecting the top 3 highest-precision chunks."*

---

## 10. Show Final Grounded Answer & Citations (1 Minute)
**Talking Point**:
> *"The RAG service formats the top 3 chunks into a numbered context block (`[1]`, `[2]`, `[3]`) with strict instructions: answer only using this context and cite sources. It calls Gateway `/v1/chat/completions` with the header `X-Bypass-Router: true`. This header tells the Gateway: 'This is an internal RAG generation call, do not route it back to RAG.' The model answers the question with bracketed citations, and the Gateway attaches `x_sources` to the response so the user can verify the exact document and page."*

---

## 11. Show Evaluation Results (1 Minute)
**Command**:
```bash
uv run python scripts/demo.py --benchmark-only
```
**Talking Point**:
> *"We evaluated both subsystems rigorously:
> 1. **Intent Router**: Achieved 93.75% classification accuracy across 48 out-of-distribution queries with zero overlap with training examples.
> 2. **Chunking & Re-ranking**: We evaluated Character, Structure, and Semantic chunking across 36 ground-truth questions. Structure chunking with re-ranking performed best, reaching 100% Hit@1. Interestingly, re-ranking did not improve character chunking on Hit@1 (86.1% vs 83.3%), because fixed character cuts sever sentences mid-phrase, giving the cross-encoder incomplete context."*

---

## 12. Mention Limitations (30 Seconds)
**Talking Point**:
> *"To keep our engineering honest, we documented key constraints:
> 1. Our evaluation corpus is small and synthetic (3 PDFs, 5 pages total).
> 2. Re-ranking adds ~140–200ms latency over pure dense search.
> 3. Scanned image-only PDFs require OCR, which is not supported.
> 4. In-process GPU generation is serialized behind a semaphore of size 1 to prevent CUDA memory conflicts."*
