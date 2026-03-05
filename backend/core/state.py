"""LangGraph State definition for the research pipeline."""

from __future__ import annotations

from typing import Annotated, Optional, TypedDict

from langgraph.graph.message import add_messages


class ResearchState(TypedDict):
    """State schema shared across all LangGraph nodes."""

    # Input
    raw_input_path: str  # Path to the raw research requirement document
    extracted_context: Optional[dict]  # Structured context after preprocessing

    # Paths
    research_paths: list[dict]  # All research paths
    path_evaluations: list[dict]  # Path evaluation results

    # Research
    research_notes: dict[str, str]  # {path_id: notes_content}
    paper_ids: list[str]  # Registered paper IDs

    # Writing
    outline: Optional[dict]  # Document outline
    context_packs: dict[str, dict]  # {section_id: context_pack}
    drafts: dict[str, str]  # {section_id: draft_content}
    terminology: list[str]  # Marked terminology list

    # Backtracking
    path_status_changes: list[dict]  # Path status change records
    new_path_proposals: list[dict]  # New path proposals
    backtrack_round: int  # Current backtrack round

    # Writing supplement research
    supplement_requests: dict[str, int]  # {section_id: supplement count}

    # Meta
    run_id: str
    run_dir: str  # Absolute path to the run directory
    current_stage: str
    messages: Annotated[list, add_messages]
