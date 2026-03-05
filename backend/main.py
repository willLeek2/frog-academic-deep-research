"""FastAPI entry point for the deep research agent backend."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiofiles
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langgraph.types import Command
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

# Ensure the backend package is on sys.path
_backend_dir = Path(__file__).resolve().parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from core.config_loader import AppConfig, load_config
from core.graph import build_graph
from utils.mcp_caller import MCPCaller
from utils.paper_registry import PaperRegistry
from utils.quota_manager import QuotaManager
from utils.run_logger import RunLogger
from utils.stop_controller import StopController

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
load_dotenv(dotenv_path=_backend_dir / ".env")

app = FastAPI(title="Deep Research Agent API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

config: AppConfig = load_config()

# Global run state registry (keyed by run_id)
run_states: dict[str, dict] = {}
stop_controllers: dict[str, StopController] = {}
# Store compiled graph apps for resume support
_graph_apps: dict[str, object] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_run_id() -> str:
    """Generate a run ID in YYYYMMDD-HHMMSS format."""
    now = datetime.now(timezone.utc)
    return now.strftime("%Y%m%d-%H%M%S")


def _runs_base() -> Path:
    return _backend_dir / "runs"


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ResumeRequest(BaseModel):
    decision: dict


class RunSummary(BaseModel):
    run_id: str
    status: str
    stage: str
    created_at: Optional[str] = None


# ---------------------------------------------------------------------------
# Pipeline executor (runs in background)
# ---------------------------------------------------------------------------

async def execute_research_pipeline(run_id: str, input_path: str) -> None:
    """Execute the LangGraph pipeline for *run_id* in the background."""
    stop_ctrl = stop_controllers.setdefault(run_id, StopController())
    run_dir = _runs_base() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = RunLogger(run_dir / "run_log.jsonl")
    paper_registry = PaperRegistry()
    quota_manager = QuotaManager(config)
    mcp_caller = MCPCaller(quota_manager, logger, paper_registry)

    logger.log("system", "pipeline", "start", {"run_id": run_id})

    try:
        graph_app = build_graph(
            run_id,
            config,
            mcp_caller=mcp_caller,
            quota_manager=quota_manager,
            run_logger=logger,
            paper_registry=paper_registry,
            stop_controller=stop_ctrl,
        )
        _graph_apps[run_id] = graph_app

        initial_state = {
            "raw_input_path": input_path,
            "extracted_context": None,
            "research_paths": [],
            "path_evaluations": [],
            "research_notes": {},
            "paper_ids": [],
            "outline": None,
            "context_packs": {},
            "drafts": {},
            "terminology": [],
            "path_status_changes": [],
            "new_path_proposals": [],
            "backtrack_round": 0,
            "supplement_requests": {},
            "run_id": run_id,
            "run_dir": str(run_dir),
            "current_stage": "init",
            "messages": [],
        }

        thread_config = {"configurable": {"thread_id": run_id}}

        # Run graph synchronously in a thread to avoid blocking the event loop
        def _run_graph():
            return graph_app.invoke(initial_state, config=thread_config)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _run_graph)

        # Check if graph is interrupted (human-in-the-loop)
        graph_state = graph_app.get_state(thread_config)
        if graph_state.next:
            run_states[run_id]["status"] = "waiting_for_human"
            run_states[run_id]["stage"] = result.get("current_stage", "interrupted")
            run_states[run_id]["interrupt_data"] = _extract_interrupt_data(graph_state)
            logger.log("system", "pipeline", "interrupted", {
                "next_nodes": list(graph_state.next),
            })
        elif stop_ctrl.is_stop_requested():
            run_states[run_id]["status"] = "stopped"
            run_states[run_id]["stage"] = "stopped"
            logger.log("system", "pipeline", "stopped", {})
        else:
            run_states[run_id]["status"] = "completed"
            run_states[run_id]["stage"] = result.get("current_stage", "completed")
            logger.log("system", "pipeline", "completed", {})

        # Persist paper index
        paper_registry.to_index_file(run_dir / "index.json")

        # Write final report
        report_path = run_dir / "report.md"
        # Try to use the report from output/ if available
        final_report = run_dir / "output" / "report_final.md"
        if final_report.exists():
            shutil.copy2(final_report, report_path)
        else:
            drafts = result.get("drafts", {})
            report_parts = [f"# Research Report — {run_id}\n"]
            for sec_id, content in drafts.items():
                report_parts.append(f"\n## {sec_id}\n\n{content}\n")
            report_path.write_text("\n".join(report_parts), encoding="utf-8")

    except Exception as exc:
        run_states[run_id]["status"] = "error"
        run_states[run_id]["error"] = str(exc)
        logger.log("system", "pipeline", "error", {"error": str(exc)})


def _extract_interrupt_data(graph_state) -> dict | None:
    """Extract interrupt payload from a LangGraph state snapshot."""
    try:
        tasks = getattr(graph_state, "tasks", ())
        for task in tasks:
            interrupts = getattr(task, "interrupts", ())
            for intr in interrupts:
                return getattr(intr, "value", None)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Resume helper
# ---------------------------------------------------------------------------

async def resume_pipeline(run_id: str, decision: dict) -> None:
    """Resume a paused pipeline after human review."""
    graph_app = _graph_apps.get(run_id)
    if graph_app is None:
        run_states[run_id]["status"] = "error"
        run_states[run_id]["error"] = "Graph app not found for resume"
        return

    thread_config = {"configurable": {"thread_id": run_id}}

    def _resume():
        return graph_app.invoke(Command(resume=decision), config=thread_config)

    try:
        run_states[run_id]["status"] = "running"
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _resume)

        # Check again for interrupts
        graph_state = graph_app.get_state(thread_config)
        if graph_state.next:
            run_states[run_id]["status"] = "waiting_for_human"
            run_states[run_id]["stage"] = result.get("current_stage", "interrupted")
            run_states[run_id]["interrupt_data"] = _extract_interrupt_data(graph_state)
        else:
            run_states[run_id]["status"] = "completed"
            run_states[run_id]["stage"] = result.get("current_stage", "completed")

        # Copy final report
        run_dir = _runs_base() / run_id
        final_report = run_dir / "output" / "report_final.md"
        if final_report.exists():
            shutil.copy2(final_report, run_dir / "report.md")

    except Exception as exc:
        run_states[run_id]["status"] = "error"
        run_states[run_id]["error"] = str(exc)


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/runs")
async def create_run(file: UploadFile = File(...)):
    """Create a new research run by uploading a markdown file."""
    run_id = _generate_run_id()
    run_dir = _runs_base() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded file
    input_path = run_dir / "input.md"
    async with aiofiles.open(input_path, "wb") as f:
        content = await file.read()
        await f.write(content)

    run_states[run_id] = {
        "run_id": run_id,
        "status": "running",
        "stage": "init",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "progress": {},
        "error": None,
    }
    stop_controllers[run_id] = StopController()

    # Launch pipeline in background
    asyncio.create_task(execute_research_pipeline(run_id, str(input_path)))

    return {"run_id": run_id, "status": "running"}


@app.get("/api/runs/{run_id}/stream")
async def stream_progress(run_id: str):
    """SSE stream for real-time progress updates."""

    async def event_generator():
        while True:
            state = run_states.get(run_id)
            if state:
                yield {"event": "progress", "data": json.dumps(state, default=str)}
            if state and state.get("status") in ("completed", "error", "stopped"):
                yield {
                    "event": "done",
                    "data": json.dumps({"status": state.get("status")}),
                }
                break
            if state and state.get("status") == "waiting_for_human":
                yield {
                    "event": "interrupt",
                    "data": json.dumps({
                        "status": "waiting_for_human",
                        "interrupt_data": state.get("interrupt_data"),
                    }, default=str),
                }
                # Keep streaming but slower
                await asyncio.sleep(3)
                continue
            await asyncio.sleep(1)

    return EventSourceResponse(event_generator())


@app.post("/api/runs/{run_id}/stop")
async def stop_run(run_id: str):
    """Request emergency stop for a running pipeline."""
    ctrl = stop_controllers.get(run_id)
    if ctrl:
        ctrl.request_stop()
    if run_id in run_states:
        run_states[run_id]["status"] = "stopped"
        run_states[run_id]["stage"] = "stopped"
    return {"status": "stopping"}


@app.post("/api/runs/{run_id}/resume")
async def resume_run(run_id: str, request: ResumeRequest):
    """Resume a paused pipeline after human review."""
    if run_id not in run_states:
        return {"error": "Run not found", "run_id": run_id}
    if run_states[run_id].get("status") != "waiting_for_human":
        return {"error": "Run is not waiting for human input", "run_id": run_id}

    # Launch resume in background
    asyncio.create_task(resume_pipeline(run_id, request.decision))

    return {"status": "resumed", "run_id": run_id}


@app.get("/api/runs/{run_id}/report")
async def get_report(run_id: str):
    """Retrieve the final generated report."""
    report_path = _runs_base() / run_id / "report.md"
    if not report_path.exists():
        return {"error": "Report not found", "run_id": run_id}
    content = report_path.read_text(encoding="utf-8")
    return {"run_id": run_id, "report": content}


@app.get("/api/runs")
async def list_runs():
    """List all runs with their current status."""
    runs: list[dict] = []
    for rid, state in run_states.items():
        runs.append({
            "run_id": rid,
            "status": state.get("status", "unknown"),
            "stage": state.get("stage", "unknown"),
            "created_at": state.get("created_at"),
        })
    return {"runs": runs}


@app.get("/health")
async def health_check():
    return {"status": "ok"}
