"""Integration test for graph construction with mock LLM.

Verifies that the graph can be built and compiles without errors when
mock LLMs are injected.
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config_loader import AppConfig, load_config
from core.graph import build_graph
from utils.mcp_caller import MCPCaller
from utils.paper_registry import PaperRegistry
from utils.quota_manager import QuotaManager
from utils.run_logger import RunLogger
from utils.stop_controller import StopController


class _FakeAIMessage:
    def __init__(self, content: str):
        self.content = content
        self.tool_calls = []


def _mock_llm():
    """Return a mock LLM that returns a generic JSON response."""
    llm = MagicMock()
    llm.invoke.return_value = _FakeAIMessage(json.dumps({
        "topic": "test", "questions": ["q1"], "keywords": ["k1"],
        "constraints": "", "expected_scope": "test",
    }))
    return llm


def test_build_graph_compiles():
    """Graph builds and compiles without error."""
    cfg = load_config(str(Path(__file__).resolve().parent.parent / "config.yaml"))
    # Disable human intervention so the graph can run end-to-end
    cfg.human_intervention.after_path_evaluation = False
    cfg.human_intervention.after_outline_planning = False

    with tempfile.TemporaryDirectory() as td:
        logger = RunLogger(Path(td) / "log.jsonl")
        registry = PaperRegistry()
        qm = QuotaManager(cfg)
        stop = StopController()
        caller = MCPCaller(qm, logger, registry)

        graph_app = build_graph(
            "test-run-001",
            cfg,
            heavy_llm=_mock_llm(),
            light_llm=_mock_llm(),
            mcp_caller=caller,
            quota_manager=qm,
            run_logger=logger,
            paper_registry=registry,
            stop_controller=stop,
        )
        assert graph_app is not None


def test_graph_topology_has_expected_nodes():
    """Verify the compiled graph has all expected node names."""
    cfg = load_config(str(Path(__file__).resolve().parent.parent / "config.yaml"))
    cfg.human_intervention.after_path_evaluation = False
    cfg.human_intervention.after_outline_planning = False

    with tempfile.TemporaryDirectory() as td:
        logger = RunLogger(Path(td) / "log.jsonl")
        registry = PaperRegistry()
        qm = QuotaManager(cfg)
        stop = StopController()
        caller = MCPCaller(qm, logger, registry)

        graph_app = build_graph(
            "test-run-002",
            cfg,
            heavy_llm=_mock_llm(),
            light_llm=_mock_llm(),
            mcp_caller=caller,
            quota_manager=qm,
            run_logger=logger,
            paper_registry=registry,
            stop_controller=stop,
        )

        # The graph should have all the expected nodes
        graph = graph_app.get_graph()
        node_ids = set()
        for n in graph.nodes:
            if isinstance(n, str):
                node_ids.add(n)
            elif hasattr(n, "id"):
                node_ids.add(n.id)
            elif hasattr(n, "name"):
                node_ids.add(n.name)
        expected = {
            "input_preprocessing", "broad_survey", "path_evaluation",
            "human_review_paths", "deep_research_dispatch",
            "deep_research_valuable", "deep_research_suboptimal",
            "post_deep_research_review", "outline_planning",
            "human_review_outline", "context_assembly",
            "sequential_writing", "post_processing",
        }
        for name in expected:
            assert name in node_ids, f"Missing node: {name}"
