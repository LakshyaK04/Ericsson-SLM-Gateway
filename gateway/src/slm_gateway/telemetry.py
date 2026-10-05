"""Lightweight, production-grade Prometheus metrics collector and exporter for SLM Gateway.

Exports standard Prometheus / OpenMetrics plain-text format (0.0.4) on /metrics:
- gateway_requests_total{intent, status}: Counter for API completions
- gateway_pii_redactions_total: Counter for detected & redacted PII items
- gateway_tokens_total{type}: Counter for prompt and generated tokens
- gateway_request_duration_seconds: Latency summary and averages
- gateway_active_requests: Gauge of in-flight inference requests
"""

import threading
from collections import defaultdict
from typing import Dict, Tuple


class MetricsCollector:
    """Thread-safe Prometheus metrics collector for GenAI gateway observability."""

    def __init__(self):
        self._lock = threading.Lock()
        # gateway_requests_total{intent="...", status="..."}
        self._requests: Dict[Tuple[str, int], int] = defaultdict(int)
        # gateway_pii_redactions_total
        self._pii_redactions: int = 0
        # gateway_tokens_total{type="prompt|completion"}
        self._tokens: Dict[str, int] = defaultdict(int)
        # Latency tracking
        self._request_count: int = 0
        self._total_duration_seconds: float = 0.0
        # Active requests gauge
        self._active_requests: int = 0

    def inc_active_requests(self) -> None:
        with self._lock:
            self._active_requests += 1

    def dec_active_requests(self) -> None:
        with self._lock:
            self._active_requests = max(0, self._active_requests - 1)

    def record_request(self, intent: str, status_code: int, duration_seconds: float) -> None:
        with self._lock:
            self._requests[(intent, status_code)] += 1
            self._request_count += 1
            self._total_duration_seconds += duration_seconds

    def record_pii_redactions(self, count: int) -> None:
        with self._lock:
            self._pii_redactions += count

    def record_tokens(self, prompt_tokens: int, completion_tokens: int) -> None:
        with self._lock:
            self._tokens["prompt"] += prompt_tokens
            self._tokens["completion"] += completion_tokens

    def export_prometheus_text(self) -> str:
        """Render metrics in official Prometheus text format 0.0.4."""
        lines = []

        with self._lock:
            # 1. Active requests gauge
            lines.append("# HELP gateway_active_requests Current in-flight inference requests.")
            lines.append("# TYPE gateway_active_requests gauge")
            lines.append(f"gateway_active_requests {self._active_requests}")

            # 2. Total requests counter
            lines.append(
                "# HELP gateway_requests_total Total number of chat completions processed."
            )
            lines.append("# TYPE gateway_requests_total counter")
            if not self._requests:
                lines.append('gateway_requests_total{intent="none",status="200"} 0')
            else:
                for (intent, status), count in sorted(self._requests.items()):
                    lines.append(
                        f'gateway_requests_total{{intent="{intent}",status="{status}"}} {count}'
                    )

            # 3. PII redactions counter
            lines.append(
                "# HELP gateway_pii_redactions_total Total number of sensitive PII entities redacted."
            )
            lines.append("# TYPE gateway_pii_redactions_total counter")
            lines.append(f"gateway_pii_redactions_total {self._pii_redactions}")

            # 4. Token counters
            lines.append("# HELP gateway_tokens_total Total tokens processed and generated.")
            lines.append("# TYPE gateway_tokens_total counter")
            lines.append(f'gateway_tokens_total{{type="prompt"}} {self._tokens.get("prompt", 0)}')
            lines.append(
                f'gateway_tokens_total{{type="completion"}} {self._tokens.get("completion", 0)}'
            )

            # 5. Request duration summary
            lines.append(
                "# HELP gateway_request_duration_seconds Total execution time for chat completion requests."
            )
            lines.append("# TYPE gateway_request_duration_seconds summary")
            lines.append(f"gateway_request_duration_seconds_sum {self._total_duration_seconds:.6f}")
            lines.append(f"gateway_request_duration_seconds_count {self._request_count}")

        return "\n".join(lines) + "\n"


# Singleton instance
telemetry = MetricsCollector()
