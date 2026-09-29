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
