"""Tests for agent modules using a mock LLM."""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config_loader import load_config
from utils.mcp_caller import MCPCaller
from utils.paper_registry import PaperRegistry
from utils.quota_manager import QuotaManager
from utils.run_logger import RunLogger
from utils.stop_controller import StopController


# ---------------------------------------------------------------------------
# Mock LLM helper
# ---------------------------------------------------------------------------

class _FakeAIMessage:
    def __init__(self, content: str):
        self.content = content


def _mock_llm(response_text: str):
    """Return a mock LLM that always produces *response_text*."""
    llm = MagicMock()
    llm.invoke.return_value = _FakeAIMessage(response_text)
    return llm


def _make_deps(tmpdir: str):
    cfg = load_config(str(Path(__file__).resolve().parent.parent / "config.yaml"))
    logger = RunLogger(Path(tmpdir) / "log.jsonl")
    registry = PaperRegistry()
    qm = QuotaManager(cfg)
    caller = MCPCaller(qm, logger, registry)
    stop = StopController()
    return cfg, logger, registry, qm, caller, stop


# ---------------------------------------------------------------------------
# InputPreprocessor
# ---------------------------------------------------------------------------

def test_input_preprocessor_with_file():
    from agents.input_preprocessor import InputPreprocessor

    with tempfile.TemporaryDirectory() as td:
        # Create input file
        inp = Path(td) / "input.md"
        inp.write_text("# Test topic\n\nResearch about AI safety.", encoding="utf-8")

        llm = _mock_llm(json.dumps({
            "topic": "AI safety",
            "questions": ["What is AI alignment?"],
            "keywords": ["AI", "safety", "alignment"],
            "constraints": "",
            "expected_scope": "broad survey",
        }))
        agent = InputPreprocessor(llm, td)
        state = {"raw_input_path": str(inp)}
        result = agent(state)

        assert result["current_stage"] == "input_preprocessing_done"
        assert result["extracted_context"]["topic"] == "AI safety"
        assert "AI" in result["extracted_context"]["keywords"]


def test_input_preprocessor_no_file():
    from agents.input_preprocessor import InputPreprocessor

    with tempfile.TemporaryDirectory() as td:
        llm = _mock_llm("{}")
        agent = InputPreprocessor(llm, td)
        state = {"raw_input_path": ""}
        result = agent(state)
        assert result["current_stage"] == "input_preprocessing_done"
        assert result["extracted_context"]["topic"] == "Unknown topic"


# ---------------------------------------------------------------------------
# BroadSurveyAgent
# ---------------------------------------------------------------------------

def test_broad_survey_agent():
    from agents.broad_survey import BroadSurveyAgent

    with tempfile.TemporaryDirectory() as td:
        cfg, logger, registry, qm, caller, stop = _make_deps(td)

        # LLM returns queries, then paths
        call_count = [0]
        def _side_effect(messages):
            call_count[0] += 1
            if call_count[0] == 1:
                return _FakeAIMessage(json.dumps({
                    "queries": [{"query": "test AI", "type": "perplexity"}]
                }))
            else:
                return _FakeAIMessage(json.dumps([
                    {"id": "path-1", "title": "AI Safety", "description": "About AI safety", "status": "proposed"},
                    {"id": "path-2", "title": "AI Alignment", "description": "About alignment", "status": "proposed"},
                ]))

        llm = MagicMock()
        llm.invoke.side_effect = _side_effect

        agent = BroadSurveyAgent(llm, caller, logger, stop, td)
        state = {
            "extracted_context": {"topic": "AI safety", "questions": ["What is AI safety?"]},
        }
        result = agent(state)

        assert result["current_stage"] == "broad_survey_done"
        assert len(result["research_paths"]) >= 2


# ---------------------------------------------------------------------------
# PathEvaluatorAgent
# ---------------------------------------------------------------------------

def test_path_evaluator_agent():
    from agents.path_evaluator import PathEvaluatorAgent

    with tempfile.TemporaryDirectory() as td:
        cfg, logger, registry, qm, caller, stop = _make_deps(td)

        llm = _mock_llm(json.dumps([
            {"path_id": "path-1", "score": 0.9, "category": "valuable", "reason": "Very relevant"},
            {"path_id": "path-2", "score": 0.4, "category": "suboptimal", "reason": "Less relevant"},
        ]))

        agent = PathEvaluatorAgent(llm, caller, logger, td)
        state = {
            "research_paths": [
                {"id": "path-1", "title": "AI Safety", "description": "About safety"},
                {"id": "path-2", "title": "AI History", "description": "About history"},
            ],
            "extracted_context": {"topic": "AI safety"},
        }
        result = agent(state)

        assert result["current_stage"] == "path_evaluation_done"
        assert len(result["path_evaluations"]) == 2
        assert result["path_evaluations"][0]["category"] == "valuable"
        assert result["path_evaluations"][1]["category"] == "suboptimal"


# ---------------------------------------------------------------------------
# OutlinePlannerAgent
# ---------------------------------------------------------------------------

def test_outline_planner_agent():
    from agents.outline_planner import OutlinePlannerAgent

    with tempfile.TemporaryDirectory() as td:
        cfg, logger, registry, qm, caller, stop = _make_deps(td)

        llm = _mock_llm(json.dumps({
            "title": "AI Safety Report",
            "sections": [
                {"id": "sec-1", "title": "Introduction", "level": 1, "target_words": 500,
                 "related_paths": [], "description": "Overview"},
                {"id": "sec-2", "title": "Main Findings", "level": 1, "target_words": 1500,
                 "related_paths": ["path-1"], "description": "Core findings"},
            ],
        }))

        agent = OutlinePlannerAgent(llm, logger, td)
        state = {
            "research_notes": {"path-1": "Some notes"},
            "path_evaluations": [{"path_id": "path-1", "score": 0.9, "category": "valuable"}],
            "extracted_context": {"topic": "AI safety"},
            "research_paths": [{"id": "path-1"}],
        }
        result = agent(state)

        assert result["current_stage"] == "outline_planning_done"
        assert result["outline"]["title"] == "AI Safety Report"
        assert len(result["outline"]["sections"]) == 2


# ---------------------------------------------------------------------------
# PostProcessorAgent
# ---------------------------------------------------------------------------

def test_post_processor_agent():
    from agents.post_processor import PostProcessorAgent

    with tempfile.TemporaryDirectory() as td:
        cfg, logger, registry, qm, caller, stop = _make_deps(td)

        llm = _mock_llm(json.dumps({
            "neural network": "A computing system inspired by biological neural networks.",
            "gradient descent": "An optimization algorithm.",
        }))

        agent = PostProcessorAgent(llm, logger, td)
        state = {
            "outline": {
                "title": "Test Report",
                "sections": [{"id": "sec-1", "title": "Intro", "level": 1}],
            },
            "drafts": {"sec-1": "This uses [[TERM:neural network]] and [[TERM:gradient descent]]."},
            "terminology": ["neural network", "gradient descent"],
            "path_evaluations": [],
            "research_notes": {},
            "run_id": "test-run",
        }
        result = agent(state)

        assert result["current_stage"] == "completed"
        # Terms should be replaced
        assert "[[TERM:" not in result["drafts"]["sec-1"]
        assert "**neural network**" in result["drafts"]["sec-1"]
        # Report file should exist
        assert (Path(td) / "output" / "report_final.md").exists()


# ---------------------------------------------------------------------------
# ContextAssembler
# ---------------------------------------------------------------------------

def test_context_assembler():
    from utils.context_assembler import ContextAssembler
    from utils.summary_generator import SummaryGenerator

    with tempfile.TemporaryDirectory() as td:
        llm = _mock_llm("A brief summary.")
        sg = SummaryGenerator(llm, Path(td) / "summaries")

        assembler = ContextAssembler(sg, td)
        outline = {
            "sections": [
                {"id": "sec-1", "title": "Intro", "level": 1, "related_paths": ["path-1"]},
                {"id": "sec-2", "title": "Body", "level": 1, "related_paths": ["path-1"]},
            ]
        }
        notes = {"path-1": "Research notes about AI safety."}
        evals = [{"path_id": "path-1", "score": 0.9, "category": "valuable"}]

        packs = assembler.assemble(outline, notes, evals)
        assert "sec-1" in packs
        assert "sec-2" in packs
        assert packs["sec-1"]["token_count"] > 0


# ---------------------------------------------------------------------------
# RelevanceEvaluator
# ---------------------------------------------------------------------------

def test_relevance_evaluator():
    from utils.relevance_evaluator import RelevanceEvaluator

    llm = _mock_llm('{"score": 8, "reason": "Very relevant to the topic."}')
    ev = RelevanceEvaluator(llm)
    result = ev.evaluate("AI safety", "This paper discusses AI safety risks.")
    assert result["score"] == 8
    assert "relevant" in result["reason"].lower()


def test_relevance_evaluator_fallback():
    from utils.relevance_evaluator import RelevanceEvaluator

    llm = _mock_llm("I think the score is 7 out of 10")
    ev = RelevanceEvaluator(llm)
    result = ev.evaluate("AI safety", "Some content")
    assert 0 <= result["score"] <= 10


# ---------------------------------------------------------------------------
# SummaryGenerator
# ---------------------------------------------------------------------------

def test_summary_generator_caching():
    from utils.summary_generator import SummaryGenerator

    with tempfile.TemporaryDirectory() as td:
        llm = _mock_llm("This is a summary.")
        sg = SummaryGenerator(llm, td)

        # First call should invoke LLM
        s1 = sg.generate("Some content", level="L1")
        assert s1 == "This is a summary."
        assert llm.invoke.call_count == 1

        # Second call should use cache
        s2 = sg.generate("Some content", level="L1")
        assert s2 == "This is a summary."
        assert llm.invoke.call_count == 1  # Still 1, cached
