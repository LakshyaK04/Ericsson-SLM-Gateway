#!/usr/bin/env bash
# ==============================================================================
# Ericsson Local GenAI Stack: Demonstration Launcher
# ==============================================================================
set -e

GATEWAY_URL=${GATEWAY_URL:-"http://localhost:8000"}
RAG_URL=${RAG_URL:-"http://localhost:8001"}

echo "Running live demonstration script against Gateway ($GATEWAY_URL) and RAG ($RAG_URL)..."
python scripts/demo.py --gateway-url "$GATEWAY_URL" --rag-url "$RAG_URL" "$@"
