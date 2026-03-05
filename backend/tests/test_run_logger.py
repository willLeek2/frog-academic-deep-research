"""Tests for RunLogger."""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.run_logger import RunLogger


def test_log_and_read():
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "test.jsonl"
        logger = RunLogger(log_path)

        logger.log("broad_survey", "agent-1", "search", {"query": "test"})
        logger.log("broad_survey", "agent-1", "fetch", {"url": "http://example.com"})

        entries = logger.read_all()
        assert len(entries) == 2
        assert entries[0]["stage"] == "broad_survey"
        assert entries[0]["event_type"] == "search"
        assert "timestamp" in entries[0]
        assert entries[1]["event_type"] == "fetch"


def test_log_creates_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "subdir" / "test.jsonl"
        logger = RunLogger(log_path)
        logger.log("init", "system", "start", {})
        assert log_path.exists()


def test_read_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "empty.jsonl"
        logger = RunLogger(log_path)
        assert logger.read_all() == []


def test_jsonl_format():
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "test.jsonl"
        logger = RunLogger(log_path)
        logger.log("test", "agent", "event", {"key": "value"})

        with open(log_path) as f:
            lines = f.readlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["detail"]["key"] == "value"
