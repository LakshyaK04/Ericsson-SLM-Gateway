.PHONY: help up down test test-cov lint format eval eval-squad eval-faithfulness clean

help:
	@echo "Available commands:"
	@echo "  make up                 Start services using Docker Compose"
	@echo "  make down               Stop Docker Compose services"
	@echo "  make test               Run unit and integration tests (non-slow)"
	@echo "  make test-cov           Run test suite with test coverage reporting"
	@echo "  make lint               Run ruff linting, format check, and mypy"
	@echo "  make format             Auto-format codebase with ruff"
	@echo "  make eval               Run baseline benchmark suites"
	@echo "  make eval-squad         Run empirical SQuAD retrieval benchmark (8 configs)"
	@echo "  make eval-faithfulness  Run RAG answer quality & faithfulness evaluation"
	@echo "  make clean              Remove build artifacts, caches, and pycache"

up:
	docker compose up -d

down:
	docker compose down

test:
	uv run pytest gateway/tests rag/tests -m "not slow" -v --tb=short

test-cov:
	uv run pytest gateway/tests rag/tests -m "not slow" --cov=slm_gateway --cov=rag_service --cov-report=term-missing

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy gateway/src rag/src

format:
	uv run ruff check --fix .
	uv run ruff format .

eval:
	uv run python scripts/run_benchmarks.py

eval-squad:
	uv run python eval/benchmark_retrieval.py

eval-faithfulness:
	uv run python eval/faithfulness_eval.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.py[cod]" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage htmlcov .mypy_cache
