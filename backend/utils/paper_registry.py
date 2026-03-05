"""Paper deduplication registry."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional


class PaperRegistry:
    """Thread-safe registry for tracking discovered papers and avoiding duplicates."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._papers: dict[str, dict] = {}  # paper_id -> metadata

    def register(self, paper_id: str, metadata: dict) -> bool:
        """Register a paper. Returns False if already registered."""
        with self._lock:
            if paper_id in self._papers:
                return False
            self._papers[paper_id] = metadata
            return True

    def is_registered(self, paper_id: str) -> bool:
        """Check if a paper is already registered."""
        with self._lock:
            return paper_id in self._papers

    def get_by_id(self, paper_id: str) -> Optional[dict]:
        """Get paper metadata by ID."""
        with self._lock:
            return self._papers.get(paper_id)

    def get_all(self) -> dict[str, dict]:
        """Return a copy of all registered papers."""
        with self._lock:
            return dict(self._papers)

    def count(self) -> int:
        """Return the number of registered papers."""
        with self._lock:
            return len(self._papers)

    def to_index_file(self, path: str | Path) -> None:
        """Serialize the registry to an index.json file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            data = self._papers.copy()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_index_file(cls, path: str | Path) -> "PaperRegistry":
        """Deserialize a registry from an index.json file."""
        path = Path(path)
        registry = cls()
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            registry._papers = data
        return registry
