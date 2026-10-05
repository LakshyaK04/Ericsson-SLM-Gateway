"""In-memory sliding-window rate limiter per client IP for RAG Service."""

import time
from collections import defaultdict
from typing import Dict, List, Tuple


class InMemoryRateLimiter:
    """Sliding-window rate limiter tracking request timestamps per client key."""

    def __init__(self, requests_limit: int = 120, window_seconds: int = 60, enabled: bool = False):
        self.requests_limit = requests_limit
        self.window_seconds = window_seconds
        self.enabled = enabled
        self._history: Dict[str, List[float]] = defaultdict(list)

    def check(self, key: str) -> Tuple[bool, int]:
        """Check if a request under `key` (client IP) is allowed.

        Args:
            key: Client identifier, e.g. IP address.

        Returns:
            Tuple of (allowed: bool, retry_after_seconds: int).
        """
        if not self.enabled:
            return True, 0

        now = time.time()
        window_start = now - self.window_seconds

        active = [t for t in self._history[key] if t > window_start]
        self._history[key] = active

        if len(active) >= self.requests_limit:
            oldest = active[0]
            retry_after = max(1, int(oldest + self.window_seconds - now))
            return False, retry_after

        self._history[key].append(now)
        return True, 0

    def reset(self) -> None:
        """Clear all in-memory tracking state (for testing)."""
        self._history.clear()
