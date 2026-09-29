# Learning Notes

This document records what was built in each phase, the design decisions made, and questions a mentor might ask. It is updated at the end of every phase.

---

## Phase 0: Baseline and Restructure

### What was built

We took the existing working prototype (a single `phi3_project` package with gateway, router, PII redaction, RAG pipeline, chunking, embeddings, and vector store all mixed together) and restructured it into the target architecture: two independent services (`gateway/` and `rag/`) plus an `eval/` folder for evaluation.

Each service now has its own `pyproject.toml`, `README.md`, source package under `src/`, and `tests/` directory. The chunking module was split from a single file into one file per strategy (`character.py`, `structure.py`, `semantic.py`) with a dispatcher in `__init__.py`. Files were moved using `git mv` to preserve history.

All imports were converted from the old `src.phi3_project.xxx` absolute paths and `.xxx` relative paths to their new package-relative forms (`slm_gateway.xxx` for gateway, `rag_service.xxx` for RAG). The sample PDF was moved to `eval/docs/`.

### Glossary

| Term | Meaning |
|------|---------|
| **uv workspace** | A uv feature that lets multiple Python packages live in one repo and share a single lockfile. We defined `gateway/` and `rag/` as workspace members in the root `pyproject.toml`. |
| **`git mv`** | Git command to rename/move a file while preserving its history in the commit log. |
| **pyproject.toml** | The standard Python project configuration file (PEP 621). It declares the project name, version, dependencies, and build system. |
| **src layout** | A project structure where the importable package is inside `src/` (e.g. `src/slm_gateway/`). This prevents accidental imports from the project root. |

### Why we did it this way

- **Two separate services** instead of one monolith, because the build plan's architecture diagram shows them communicating over HTTP. This means each can be deployed, tested, and Dockerised independently.
- **uv workspace** rather than two completely separate repos, because we're one intern working locally — sharing a lockfile avoids dependency version conflicts and makes `pytest` discovery easy across both.
- **src layout** because it forces you to install the package before importing it. This catches import errors early and mirrors how the code runs in production.
- **One chunking file per strategy** instead of everything in one file, because the plan explicitly asks for `chunking/ {base.py, character.py, structure.py, semantic.py}` and each strategy is independent, making it easier to test and compare.
- **Kept FAISS for now** even though the plan mentions ChromaDB. We will switch in Phase 4 when building the full RAG service endpoints. Changing too many things in one phase risks breaking the working tests.

### Mentor questions

**Q1: Why not just keep everything in one package?**
A: The build plan requires two services that talk over HTTP. Separate packages make the boundary explicit — the gateway can't accidentally import RAG internals. It also means each service gets its own Dockerfile and can scale independently.

**Q2: Why use `git mv` instead of just moving files?**
A: `git mv` is how git tracks renames. Without it, git sees a delete + create and we lose the file's blame history. With it, `git log --follow` can trace a file back to its original location.

**Q3: What happens if `uv sync` fails on a dependency?**
A: We pinned versions that already work on this machine. If a new dependency fails, we stop and investigate rather than guessing — the build plan says "if a pin causes a real error, stop and ask."

**Q4: Why keep the old test_phi3.py as a reference file?**
A: It contains the working model-loading code (quantization config, tokenizer template, generation loop) that Phase 1 will reuse for the `hf_local` backend. It's not a test in the pytest sense, so it was renamed to `test_phi3_reference.py`.

**Q5: How can you verify this phase is working?**
A: Run `uv run pytest -m "not slow" -v` from the repo root to run the fast tests, or `uv run pytest -v` to run all tests including slow model tests. Both services collect and pass their tests.

### Verification command

```bash
uv run pytest -m "not slow" -v
```

---

## Phase 1: Gateway Core

### What was built

We implemented the core SLM Gateway server with full OpenAI API compatibility. We built `config.py` using `pydantic-settings` to load all configurations from environment variables, and `schemas.py` defining OpenAI-compatible request, response, error, and model listing models.

We implemented an extensible backend architecture under `backends/` with an abstract base class `LLMBackend`, an in-process HuggingFace backend `HFLocalBackend` loading `microsoft/Phi-3-mini-4k-instruct` with 4-bit NF4 quantization via bitsandbytes protected by a single-GPU semaphore, and an `OpenAICompatibleBackend` forwarding requests to any OpenAI-compatible HTTP server.

We wired these into `main.py` with FastAPI lifespan model loading, `/health` and `/ready` probes, `/v1/models`, `/v1/chat/completions`, and optional Bearer token authentication in `security.py`. All error responses conform to OpenAI's `{"error": {"message", "type", "code"}}` format.

### Glossary

| Term | Meaning |
|------|---------|
| **NF4 (NormalFloat4)** | An information-theoretically optimal 4-bit quantization data type provided by bitsandbytes for normally distributed neural network weights. |
| **FastAPI Lifespan** | An asynchronous context manager hook that runs setup (loading the model) on startup and teardown on shutdown. |
| **Semaphore** | A concurrency synchronization primitive. We use `asyncio.Semaphore(1)` to ensure only one thread generates on the GPU at a time, preventing Out-Of-Memory errors. |
| **OpenAI Compatibility** | Implementing the exact JSON schemas and endpoint paths (`/v1/chat/completions`, `/v1/models`) used by OpenAI, allowing any OpenAI SDK client to work by just changing `base_url`. |
| **Readiness vs Liveness Probe** | `/health` (liveness) checks if the HTTP server process is running; `/ready` (readiness) returns 200 only when the heavy AI model is loaded and ready for inference. |

### Why we did it this way

- **In-process HuggingFace with 4-bit quantization** rather than full fp16/bf16, because 4-bit NF4 reduces Phi-3-mini's VRAM footprint to ~2.3GB, allowing it to run smoothly on a laptop GPU (RTX 3050 6GB) while retaining high reasoning quality.
- **FastAPI lifespan loading** rather than loading per request, because loading weights from disk takes ~25-30 seconds. Lifespan loads once when the process starts.
- **Asyncio semaphore behind asyncio.to_thread** because HuggingFace PyTorch generation is CPU/GPU blocking. `asyncio.to_thread` runs generation in a worker thread so the event loop remains responsive for health checks, while the semaphore serializes GPU execution.
- **OpenAI Compatible backend abstraction** so developers without a dedicated CUDA GPU can develop and test against external endpoints (or vLLM/Ollama) with a single environment flag `BACKEND=openai_compatible`.
- **Standardized OpenAI error shapes** so client libraries like the official `openai` Python SDK handle errors (such as 401 Unauthorized or 422 Validation Error) gracefully as standard API errors.

### Mentor questions

**Q1: Why do we need `asyncio.to_thread` and an `asyncio.Semaphore(1)` around generation?**
A: PyTorch's `model.generate()` is a blocking synchronous call. Without `asyncio.to_thread`, running it would block the Python event loop, causing all concurrent requests (including `/health` and `/ready` probes) to freeze. The semaphore of size 1 ensures that multiple requests don't attempt simultaneous generation on a single GPU, avoiding CUDA Out-of-Memory crashes.

**Q2: How does authentication work, and why is it optional?**
A: In `security.py`, `verify_api_key` inspects `GATEWAY_API_KEY`. If unset, the gateway acts as an open local proxy for easy development. If configured, every request to protected routes must include `Authorization: Bearer <key>`, or it receives a 401 `authentication_error`.

**Q3: How does prompt context truncation work in `HFLocalBackend`?**
A: Phi-3-mini has a 4096 token context window. If the prompt tokens plus `max_tokens` exceeds 4096, `HFLocalBackend` preserves the system prompt at index 0 and progressively trims the oldest conversational turns, logging a warning, instead of failing silently or crashing.

**Q4: How can third-party tools use this gateway?**
A: Any application or SDK designed for OpenAI can interact with this service simply by configuring `base_url="http://localhost:8000/v1"` and setting any arbitrary API key if `GATEWAY_API_KEY` is disabled.

**Q5: How do we verify Phase 1 without waiting for heavy model loads during CI?**
A: Unit and API tests mock the backend, executing 16 tests in less than a second. A dedicated `@pytest.mark.slow` test (`test_phi3_slow.py`) tests the real in-process model inference on the GPU when desired.

### Verification command

```bash
# 1. Run unit test suite
uv run pytest gateway/tests/test_api.py gateway/tests/test_schemas.py -v

# 2. Test official OpenAI Python client against running gateway
uv run python -c "from openai import OpenAI; client = OpenAI(base_url='http://127.0.0.1:8000/v1', api_key='not-needed'); resp = client.chat.completions.create(model='microsoft/Phi-3-mini-4k-instruct', messages=[{'role': 'user', 'content': 'What is the capital of Sweden?'}], temperature=0.0); print(resp.choices[0].message.content)"
```

