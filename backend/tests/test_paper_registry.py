"""Tests for PaperRegistry."""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paper_registry import PaperRegistry


def test_register_and_lookup():
    reg = PaperRegistry()
    assert reg.register("p1", {"title": "Paper 1"}) is True
    assert reg.is_registered("p1") is True
    assert reg.get_by_id("p1") == {"title": "Paper 1"}


def test_register_duplicate():
    reg = PaperRegistry()
    reg.register("p1", {"title": "Paper 1"})
    assert reg.register("p1", {"title": "Duplicate"}) is False
    assert reg.get_by_id("p1") == {"title": "Paper 1"}


def test_count():
    reg = PaperRegistry()
    assert reg.count() == 0
    reg.register("p1", {"title": "Paper 1"})
    reg.register("p2", {"title": "Paper 2"})
    assert reg.count() == 2


def test_serialize_and_deserialize():
    reg = PaperRegistry()
    reg.register("p1", {"title": "Paper 1"})
    reg.register("p2", {"title": "Paper 2"})

    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = Path(tmpdir) / "index.json"
        reg.to_index_file(index_path)

        assert index_path.exists()
        with open(index_path) as f:
            data = json.load(f)
        assert "p1" in data
        assert "p2" in data

        restored = PaperRegistry.from_index_file(index_path)
        assert restored.is_registered("p1")
        assert restored.is_registered("p2")
        assert restored.count() == 2


def test_from_index_file_missing():
    reg = PaperRegistry.from_index_file("/nonexistent/path/index.json")
    assert reg.count() == 0


def test_get_all():
    reg = PaperRegistry()
    reg.register("p1", {"title": "A"})
    reg.register("p2", {"title": "B"})
    all_papers = reg.get_all()
    assert len(all_papers) == 2
    assert all_papers["p1"]["title"] == "A"
