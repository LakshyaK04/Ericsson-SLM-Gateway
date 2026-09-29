# ==============================================================================
# SLM Gateway Dockerfile
#
# Enterprise Container Specification:
# - Pinned Python 3.10 slim base image
# - Non-root execution (appuser:10001) for container hardening
# - Healthcheck on /health endpoint
# - Supports GPU inference (NVIDIA Container Toolkit) or CPU fallback
# ==============================================================================

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install minimal OS dependencies for healthcheck and native wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast, reliable package installation
RUN pip install --no-cache-dir uv

# Copy dependency specifications
COPY pyproject.toml ./

# Install project dependencies
RUN uv pip install --system --no-cache -e .

# Ensure spaCy language pipeline is installed
RUN python -m spacy download en_core_web_sm || true

# Copy gateway source code and configuration
COPY src/ ./src/
COPY README.md ./

# Create unprivileged system user
RUN useradd -m -u 10001 -s /bin/bash appuser && \
    chown -R appuser:appuser /app

USER appuser

# Expose Gateway HTTP service port
EXPOSE 8000

# Docker healthcheck testing liveness
HEALTHCHECK --interval=20s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start Uvicorn ASGI server
CMD ["uvicorn", "slm_gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]
