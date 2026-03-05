"""LangGraph graph construction and compilation.

Builds the full research pipeline topology with real agent implementations.
Each node is a callable agent class instance that captures its dependencies
via the constructor, making the graph both testable and self-contained.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from agents.broad_survey import BroadSurveyAgent
from agents.deep_researcher import DeepResearcherAgent
from agents.input_preprocessor import InputPreprocessor
from agents.outline_planner import OutlinePlannerAgent
from agents.path_evaluator import PathEvaluatorAgent
from agents.post_processor import PostProcessorAgent
from agents.writer import WriterAgent
from core.config_loader import AppConfig
from core.state import ResearchState
from models.llm_factory import create_heavy_llm, create_light_llm
from utils.context_assembler import ContextAssembler
from utils.mcp_caller import MCPCaller
from utils.paper_registry import PaperRegistry
from utils.quota_manager import QuotaManager
from utils.relevance_evaluator import RelevanceEvaluator
from utils.run_logger import RunLogger
from utils.stop_controller import StopController
from utils.summary_generator import SummaryGenerator


# ---------------------------------------------------------------------------
# Human-in-the-loop nodes
# ---------------------------------------------------------------------------

def human_review_paths_node(state: ResearchState) -> dict[str, Any]:
    """Interrupt for human review of path evaluations."""
    decision = interrupt({
        "type": "path_evaluation_review",
        "paths": state.get("research_paths", []),
        "evaluations": state.get("path_evaluations", []),
        "message": "请审核研究路径评估结果，确认后继续。",
    })
    # When resumed, *decision* contains the user's confirmed evaluations
    if isinstance(decision, dict):
        confirmed = decision.get("confirmed_evaluations")
        if confirmed and isinstance(confirmed, list):
            return {
                "path_evaluations": confirmed,
                "current_stage": "human_review_paths_done",
            }
    return {"current_stage": "human_review_paths_done"}


def human_review_outline_node(state: ResearchState) -> dict[str, Any]:
    """Interrupt for human review of the report outline."""
    decision = interrupt({
        "type": "outline_review",
        "outline": state.get("outline"),
        "message": "请审核报告大纲，确认后继续。",
    })
    if isinstance(decision, dict):
        confirmed_outline = decision.get("confirmed_outline")
        if confirmed_outline and isinstance(confirmed_outline, dict):
            return {
                "outline": confirmed_outline,
                "current_stage": "human_review_outline_done",
            }
    return {"current_stage": "human_review_outline_done"}


# ---------------------------------------------------------------------------
# Dispatch / review helper nodes
# ---------------------------------------------------------------------------

def _make_dispatch_node():
    """Return a simple dispatch node."""
    def deep_research_dispatch_node(state: ResearchState) -> dict[str, Any]:
        return {"current_stage": "deep_research_dispatch_done"}
    return deep_research_dispatch_node


def _make_post_deep_research_review():
    """Return a node that checks for path status changes / new proposals."""
    def post_deep_research_review_node(state: ResearchState) -> dict[str, Any]:
        changes = state.get("path_status_changes", [])
        proposals = state.get("new_path_proposals", [])
        backtrack = state.get("backtrack_round", 0)

        needs_backtrack = False
        # Downgraded paths -> need re-evaluation
        if any(c.get("change") == "downgrade" for c in changes):
            needs_backtrack = True
        # New path proposals -> need evaluation
        if proposals:
            needs_backtrack = True

        if needs_backtrack and backtrack < 2:
            # Inject proposed paths into research_paths
            new_paths = list(state.get("research_paths", []))
            for prop in proposals:
                new_paths.append({
                    "id": f"path-new-{backtrack + 1}-{len(new_paths)}",
                    "title": prop.get("title", "New path"),
                    "description": prop.get("description", ""),
                    "status": "proposed",
                })
            return {
                "research_paths": new_paths,
                "new_path_proposals": [],
                "path_status_changes": [],
                "backtrack_round": backtrack + 1,
                "current_stage": "backtrack_to_evaluation",
            }
        return {"current_stage": "post_deep_research_review_done"}
    return post_deep_research_review_node


def _make_context_assembly_node(
    summary_gen: SummaryGenerator,
    run_dir: str | Path,
):
    assembler = ContextAssembler(summary_gen, run_dir)

    def context_assembly_node(state: ResearchState) -> dict[str, Any]:
        outline = state.get("outline") or {}
        notes = state.get("research_notes", {})
        evals = state.get("path_evaluations", [])
        drafts = state.get("drafts", {})
        packs = assembler.assemble(outline, notes, evals, drafts or None)

        # Persist context packs
        cp_dir = Path(run_dir) / "writing" / "context_packs"
        cp_dir.mkdir(parents=True, exist_ok=True)
        import json
        for sid, pack in packs.items():
            (cp_dir / f"{sid}.json").write_text(
                json.dumps(pack, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )

        return {"context_packs": packs, "current_stage": "context_assembly_done"}
    return context_assembly_node


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------

def _make_route_after_evaluation(config: AppConfig):
    def route_after_evaluation(state: ResearchState) -> Literal["human_review_paths", "deep_research_dispatch"]:
        if config.human_intervention.after_path_evaluation:
            return "human_review_paths"
        return "deep_research_dispatch"
    return route_after_evaluation


def _make_route_after_deep_research():
    def route_after_deep_research(state: ResearchState) -> Literal["post_deep_research_review", "outline_planning"]:
        # Always go through post-review to check for backtracks
        return "post_deep_research_review"
    return route_after_deep_research


def _make_route_after_post_review():
    def route_after_post_review(state: ResearchState) -> Literal["path_evaluation", "outline_planning"]:
        if state.get("current_stage") == "backtrack_to_evaluation":
            return "path_evaluation"
        return "outline_planning"
    return route_after_post_review


def _make_route_after_outline(config: AppConfig):
    def route_after_outline(state: ResearchState) -> Literal["human_review_outline", "context_assembly"]:
        if config.human_intervention.after_outline_planning:
            return "human_review_outline"
        return "context_assembly"
    return route_after_outline


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(
    run_id: str,
    config: AppConfig | None = None,
    *,
    heavy_llm: BaseChatModel | None = None,
    light_llm: BaseChatModel | None = None,
    mcp_caller: MCPCaller | None = None,
    quota_manager: QuotaManager | None = None,
    run_logger: RunLogger | None = None,
    paper_registry: PaperRegistry | None = None,
    stop_controller: StopController | None = None,
) -> Any:
    """Build and compile the full research pipeline graph.

    All optional dependencies default to production instances when not
    supplied, making the function usable both in production and in tests
    (where mocks can be injected).
    """
    if config is None:
        from core.config_loader import load_config
        config = load_config()

    runs_dir = Path(__file__).resolve().parent.parent / "runs" / run_id
    runs_dir.mkdir(parents=True, exist_ok=True)

    # -- Defaults for optional dependencies --
    if run_logger is None:
        run_logger = RunLogger(runs_dir / "run_log.jsonl")
    if paper_registry is None:
        paper_registry = PaperRegistry()
    if quota_manager is None:
        quota_manager = QuotaManager(config)
    if stop_controller is None:
        stop_controller = StopController()
    if mcp_caller is None:
        mcp_caller = MCPCaller(quota_manager, run_logger, paper_registry)
    if heavy_llm is None:
        heavy_llm = create_heavy_llm(config)
    if light_llm is None:
        light_llm = create_light_llm(config)

    summary_gen = SummaryGenerator(light_llm, runs_dir / "papers" / "summaries")
    relevance_eval = RelevanceEvaluator(light_llm)

    # -- Create agent instances --
    input_pre = InputPreprocessor(heavy_llm, runs_dir)
    broad_survey = BroadSurveyAgent(heavy_llm, mcp_caller, run_logger, stop_controller, runs_dir)
    path_eval = PathEvaluatorAgent(heavy_llm, mcp_caller, run_logger, runs_dir)
    deep_researcher = DeepResearcherAgent(
        heavy_llm, light_llm, mcp_caller, summary_gen, relevance_eval,
        run_logger, stop_controller, runs_dir,
    )
    outline_planner = OutlinePlannerAgent(heavy_llm, run_logger, runs_dir)
    writer = WriterAgent(heavy_llm, mcp_caller, run_logger, stop_controller, runs_dir)
    post_processor = PostProcessorAgent(light_llm, run_logger, runs_dir)

    # -- Build graph --
    graph = StateGraph(ResearchState)

    graph.add_node("input_preprocessing", input_pre)
    graph.add_node("broad_survey", broad_survey)
    graph.add_node("path_evaluation", path_eval)
    graph.add_node("human_review_paths", human_review_paths_node)
    graph.add_node("deep_research_dispatch", _make_dispatch_node())
    graph.add_node("deep_research_valuable", deep_researcher)
    graph.add_node("deep_research_suboptimal", lambda state: {"current_stage": "deep_research_suboptimal_done"})
    graph.add_node("post_deep_research_review", _make_post_deep_research_review())
    graph.add_node("outline_planning", outline_planner)
    graph.add_node("human_review_outline", human_review_outline_node)
    graph.add_node("context_assembly", _make_context_assembly_node(summary_gen, runs_dir))
    graph.add_node("sequential_writing", writer)
    graph.add_node("post_processing", post_processor)

    # -- Define edges --
    graph.add_edge(START, "input_preprocessing")
    graph.add_edge("input_preprocessing", "broad_survey")
    graph.add_edge("broad_survey", "path_evaluation")

    graph.add_conditional_edges(
        "path_evaluation",
        _make_route_after_evaluation(config),
        {"human_review_paths": "human_review_paths", "deep_research_dispatch": "deep_research_dispatch"},
    )
    graph.add_edge("human_review_paths", "deep_research_dispatch")

    graph.add_edge("deep_research_dispatch", "deep_research_valuable")
    graph.add_edge("deep_research_valuable", "deep_research_suboptimal")

    graph.add_conditional_edges(
        "deep_research_suboptimal",
        _make_route_after_deep_research(),
        {"post_deep_research_review": "post_deep_research_review", "outline_planning": "outline_planning"},
    )

    # Post-review: backtrack to path_evaluation or continue to outline
    graph.add_conditional_edges(
        "post_deep_research_review",
        _make_route_after_post_review(),
        {"path_evaluation": "path_evaluation", "outline_planning": "outline_planning"},
    )

    graph.add_conditional_edges(
        "outline_planning",
        _make_route_after_outline(config),
        {"human_review_outline": "human_review_outline", "context_assembly": "context_assembly"},
    )
    graph.add_edge("human_review_outline", "context_assembly")

    graph.add_edge("context_assembly", "sequential_writing")
    graph.add_edge("sequential_writing", "post_processing")
    graph.add_edge("post_processing", END)

    # -- Compile with SQLite checkpointer --
    db_path = str(runs_dir / "checkpoints.db")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    return graph.compile(checkpointer=checkpointer)
