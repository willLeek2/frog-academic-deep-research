"""LangGraph graph construction and compilation.

Builds the full research pipeline topology. In Issue 1, only
``input_preprocessing`` uses a real (mock-able) LLM call; all other nodes
are stub implementations that pass through or produce mock data.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Literal

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from core.config_loader import AppConfig
from core.state import ResearchState


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

def input_preprocessing_node(state: ResearchState) -> dict[str, Any]:
    """Stage 1: Extract structured context from raw input document."""
    raw_path = state.get("raw_input_path", "")
    content = ""
    if raw_path and Path(raw_path).exists():
        content = Path(raw_path).read_text(encoding="utf-8")

    extracted = {
        "topic": content[:200] if content else "Unknown topic",
        "requirements": "Extracted from uploaded document",
        "keywords": ["research", "survey"],
        "raw_length": len(content),
    }
    return {
        "extracted_context": extracted,
        "current_stage": "input_preprocessing_done",
    }


def broad_survey_node(state: ResearchState) -> dict[str, Any]:
    """Stage 2: Broad survey — mock implementation producing 3 paths."""
    topic = (state.get("extracted_context") or {}).get("topic", "unknown")
    mock_paths = [
        {
            "id": "path-1",
            "title": f"Theoretical foundations of {topic[:30]}",
            "description": "Explore the theoretical underpinnings.",
            "status": "proposed",
        },
        {
            "id": "path-2",
            "title": f"Practical applications of {topic[:30]}",
            "description": "Survey real-world applications and case studies.",
            "status": "proposed",
        },
        {
            "id": "path-3",
            "title": f"Recent advances in {topic[:30]}",
            "description": "Review state-of-the-art developments.",
            "status": "proposed",
        },
    ]
    return {
        "research_paths": mock_paths,
        "current_stage": "broad_survey_done",
    }


def path_evaluation_node(state: ResearchState) -> dict[str, Any]:
    """Stage 3: Evaluate and score research paths — mock."""
    evaluations = []
    for p in state.get("research_paths", []):
        evaluations.append({
            "path_id": p["id"],
            "score": 0.85,
            "category": "valuable",
            "reason": f"Path '{p['title']}' is highly relevant.",
        })
    return {
        "path_evaluations": evaluations,
        "current_stage": "path_evaluation_done",
    }


def human_review_paths_node(state: ResearchState) -> dict[str, Any]:
    """Human-in-the-loop: review paths (mock — auto-approve)."""
    return {"current_stage": "human_review_paths_done"}


def deep_research_dispatch_node(state: ResearchState) -> dict[str, Any]:
    """Dispatch deep research tasks — mock."""
    return {"current_stage": "deep_research_dispatch_done"}


def deep_research_valuable_node(state: ResearchState) -> dict[str, Any]:
    """Deep research on valuable paths — mock."""
    notes: dict[str, str] = {}
    for p in state.get("research_paths", []):
        notes[p["id"]] = f"[Mock] Detailed research notes for {p['title']}"
    return {
        "research_notes": notes,
        "current_stage": "deep_research_valuable_done",
    }


def deep_research_suboptimal_node(state: ResearchState) -> dict[str, Any]:
    """Deep research on suboptimal paths — mock (no-op)."""
    return {"current_stage": "deep_research_suboptimal_done"}


def post_deep_research_review_node(state: ResearchState) -> dict[str, Any]:
    """Post deep-research review — mock."""
    return {"current_stage": "post_deep_research_review_done"}


def outline_planning_node(state: ResearchState) -> dict[str, Any]:
    """Stage 5A: Generate document outline — mock."""
    outline = {
        "title": "Research Report",
        "sections": [
            {"id": "sec-1", "title": "Introduction", "level": 1},
            {"id": "sec-2", "title": "Background", "level": 1},
            {"id": "sec-3", "title": "Methods", "level": 1},
            {"id": "sec-4", "title": "Results", "level": 1},
            {"id": "sec-5", "title": "Conclusion", "level": 1},
        ],
    }
    return {"outline": outline, "current_stage": "outline_planning_done"}


def human_review_outline_node(state: ResearchState) -> dict[str, Any]:
    """Human-in-the-loop: review outline (mock — auto-approve)."""
    return {"current_stage": "human_review_outline_done"}


def context_assembly_node(state: ResearchState) -> dict[str, Any]:
    """Assemble context for writing — mock."""
    return {"current_stage": "context_assembly_done"}


def sequential_writing_node(state: ResearchState) -> dict[str, Any]:
    """Stage 5C/5D: Sequential section writing — mock."""
    drafts: dict[str, str] = {}
    sections = (state.get("outline") or {}).get("sections", [])
    for sec in sections:
        drafts[sec["id"]] = f"[Mock draft for {sec['title']}] Lorem ipsum..."
    return {"drafts": drafts, "current_stage": "writing_done"}


def post_processing_node(state: ResearchState) -> dict[str, Any]:
    """Stage 6: Post-processing — mock."""
    return {"current_stage": "completed"}


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------

def route_after_evaluation(state: ResearchState) -> Literal["human_review_paths", "deep_research_dispatch"]:
    """Route after path evaluation: go to human review if configured."""
    # In this skeleton, always go to human review
    return "human_review_paths"


def route_after_deep_research(state: ResearchState) -> Literal["post_deep_research_review", "outline_planning"]:
    """Route after deep research: optionally review."""
    return "outline_planning"


def route_after_outline(state: ResearchState) -> Literal["human_review_outline", "context_assembly"]:
    """Route after outline planning."""
    return "human_review_outline"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(run_id: str, config: AppConfig | None = None) -> Any:
    """Build and compile the full research pipeline graph.

    Args:
        run_id: Unique run identifier used for checkpoint storage.
        config: Application configuration.

    Returns:
        Compiled LangGraph application.
    """
    graph = StateGraph(ResearchState)

    # -- Add all nodes --
    graph.add_node("input_preprocessing", input_preprocessing_node)
    graph.add_node("broad_survey", broad_survey_node)
    graph.add_node("path_evaluation", path_evaluation_node)
    graph.add_node("human_review_paths", human_review_paths_node)
    graph.add_node("deep_research_dispatch", deep_research_dispatch_node)
    graph.add_node("deep_research_valuable", deep_research_valuable_node)
    graph.add_node("deep_research_suboptimal", deep_research_suboptimal_node)
    graph.add_node("post_deep_research_review", post_deep_research_review_node)
    graph.add_node("outline_planning", outline_planning_node)
    graph.add_node("human_review_outline", human_review_outline_node)
    graph.add_node("context_assembly", context_assembly_node)
    graph.add_node("sequential_writing", sequential_writing_node)
    graph.add_node("post_processing", post_processing_node)

    # -- Define edges (full topology) --
    graph.add_edge(START, "input_preprocessing")
    graph.add_edge("input_preprocessing", "broad_survey")
    graph.add_edge("broad_survey", "path_evaluation")

    # After evaluation: human review or skip to deep research
    graph.add_conditional_edges(
        "path_evaluation",
        route_after_evaluation,
        {
            "human_review_paths": "human_review_paths",
            "deep_research_dispatch": "deep_research_dispatch",
        },
    )
    graph.add_edge("human_review_paths", "deep_research_dispatch")

    # Deep research dispatch -> valuable path research
    graph.add_edge("deep_research_dispatch", "deep_research_valuable")
    graph.add_edge("deep_research_valuable", "deep_research_suboptimal")

    # After deep research: optional review then outline
    graph.add_conditional_edges(
        "deep_research_suboptimal",
        route_after_deep_research,
        {
            "post_deep_research_review": "post_deep_research_review",
            "outline_planning": "outline_planning",
        },
    )
    graph.add_edge("post_deep_research_review", "outline_planning")

    # After outline: human review or skip to context assembly
    graph.add_conditional_edges(
        "outline_planning",
        route_after_outline,
        {
            "human_review_outline": "human_review_outline",
            "context_assembly": "context_assembly",
        },
    )
    graph.add_edge("human_review_outline", "context_assembly")

    # Writing pipeline
    graph.add_edge("context_assembly", "sequential_writing")
    graph.add_edge("sequential_writing", "post_processing")
    graph.add_edge("post_processing", END)

    # -- Compile with SQLite checkpointer --
    runs_dir = Path(__file__).resolve().parent.parent / "runs" / run_id
    runs_dir.mkdir(parents=True, exist_ok=True)
    db_path = str(runs_dir / "checkpoints.db")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    return graph.compile(checkpointer=checkpointer)
