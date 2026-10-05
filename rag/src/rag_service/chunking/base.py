"""Base definitions for document chunking."""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Chunk:
    """Standardized chunk representation across all chunking strategies per Section 5.4."""

    chunk_id: str
    text: str
    source: str
    page: int
    strategy: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to a dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source": self.source,
            "page": self.page,
            "strategy": self.strategy,
            "metadata": self.metadata,
        }
