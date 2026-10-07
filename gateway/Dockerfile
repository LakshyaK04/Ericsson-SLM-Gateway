# ==============================================================================
# SLM Gateway Dockerfile (Multi-Stage Build)
#
# Container Specification:
# - Pinned base image: python:3.10.16-slim-bookworm
# - Multi-stage build: builder builds virtualenv; runtime keeps image minimal
# - Non-root execution: appuser (UID 10001) for container hardening
# - Healthcheck on /health endpoint
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Builder Stage
# ------------------------------------------------------------------------------
FROM python:3.10.16-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# Install build dependencies for compiling native C/C++ wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

# Create isolated virtual environment
RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml README.md ./
RUN uv pip install --no-cache -e .
RUN python -m spacy download en_core_web_sm || true

# ------------------------------------------------------------------------------
# Stage 2: Runtime Stage
# ------------------------------------------------------------------------------
FROM python:3.10.16-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Install minimal runtime dependencies (curl for container healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtualenv from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy gateway source code, static assets (Playground UI), and configuration
COPY src/ ./src/
COPY pyproject.toml README.md ./

# Create unprivileged system user and set ownership
RUN useradd -m -u 10001 -s /bin/bash appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Docker healthcheck testing liveness
HEALTHCHECK --interval=20s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start Uvicorn ASGI server
CMD ["uvicorn", "slm_gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]
