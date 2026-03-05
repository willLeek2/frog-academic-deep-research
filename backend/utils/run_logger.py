"""JSONL run logger for tracking pipeline execution events."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path


class RunLogger:
    """Append-only JSONL logger for a single run.

    Each log entry is a single JSON line containing:
    - timestamp (ISO-8601)
    - stage
    - agent
    - event_type
    - detail (arbitrary dict)
    """

    def __init__(self, log_path: str | Path) -> None:
        self._lock = threading.Lock()
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        stage: str,
        agent: str,
        event_type: str,
        detail: dict | None = None,
    ) -> None:
        """Append a log entry."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "agent": agent,
            "event_type": event_type,
            "detail": detail or {},
        }
        line = json.dumps(entry, ensure_ascii=False)
        with self._lock:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def read_all(self) -> list[dict]:
        """Read all log entries."""
        if not self._log_path.exists():
            return []
        entries: list[dict] = []
        with open(self._log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries
