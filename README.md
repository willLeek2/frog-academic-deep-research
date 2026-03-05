# 🐸 Deep Research Agent

AI-powered academic deep research agent built with **LangGraph** + **FastAPI** + **React**.

## Overview

This project implements a multi-stage research pipeline that:
1. **Input Preprocessing** — Extracts structured context from research requirements
2. **Broad Survey** — Discovers and proposes research paths
3. **Path Evaluation** — Scores and categorizes paths
4. **Deep Research** — In-depth investigation of selected paths
5. **Writing** — Generates a structured research report
6. **Post Processing** — Finalizes and formats the output

## Quick Start

### Prerequisites
- Python >= 3.11
- Node.js >= 20 LTS

### Backend

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env   # Edit and add your OPENROUTER_API_KEY
python -m uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev   # Opens at http://localhost:5173
```

### Run Tests

```bash
cd backend
python -m pytest tests/ -v
```

## Project Structure

```
├── backend/
│   ├── main.py                    # FastAPI entry point
│   ├── config.yaml                # Default configuration
│   ├── requirements.txt
│   ├── core/
│   │   ├── state.py               # LangGraph State definition
│   │   ├── graph.py               # LangGraph graph construction
│   │   └── config_loader.py       # Configuration loader
│   ├── agents/                    # Agent implementations (Issue 2)
│   ├── utils/
│   │   ├── quota_manager.py       # MCP call quota management
│   │   ├── paper_registry.py      # Paper deduplication
│   │   ├── run_logger.py          # JSONL run logging
│   │   ├── stop_controller.py     # Emergency stop control
│   │   ├── token_counter.py       # Token counting
│   │   └── mcp_caller.py          # MCP call wrapper (mock)
│   ├── tools/
│   │   └── mcp_tools.py           # LangChain tool definitions
│   ├── models/
│   │   └── llm_factory.py         # LLM instance factory
│   ├── runs/                      # Run data storage
│   └── tests/                     # Unit tests
├── frontend/
│   ├── src/
│   │   ├── App.tsx                # Main application layout
│   │   ├── components/
│   │   │   ├── RunList.tsx        # Run history list
│   │   │   ├── RunCreator.tsx     # New task creation
│   │   │   ├── ProgressPanel.tsx  # Real-time progress panel
│   │   │   ├── StageIndicator.tsx # Stage visualization
│   │   │   ├── PathReview.tsx     # Path review (placeholder)
│   │   │   ├── OutlineReview.tsx  # Outline review (placeholder)
│   │   │   └── ReportViewer.tsx   # Report display
│   │   ├── hooks/
│   │   │   └── useSSE.ts         # SSE connection hook
│   │   ├── types/
│   │   │   └── index.ts          # TypeScript type definitions
│   │   └── api/
│   │       └── client.ts         # Backend API client
│   └── ...
└── README.md
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/runs` | Create a new research run (file upload) |
| GET | `/api/runs` | List all runs |
| GET | `/api/runs/{id}/stream` | SSE progress stream |
| POST | `/api/runs/{id}/stop` | Emergency stop |
| POST | `/api/runs/{id}/resume` | Resume after human review |
| GET | `/api/runs/{id}/report` | Get final report |
| GET | `/health` | Health check |

## Architecture

- **LangGraph StateGraph** with 13 nodes and conditional routing
- **SQLite checkpointing** for run persistence
- **SSE (Server-Sent Events)** for real-time progress updates
- **Human-in-the-loop** support via LangGraph interrupt mechanism (placeholder)
- **Dual LLM strategy**: heavy model for reasoning, light model for extraction
- **OpenRouter integration** with automatic fallback (Scheme A → Scheme B)

## Current Status (Issue 1)

This is the **minimum viable skeleton**:
- ✅ All utils infrastructure (quota, paper registry, logging, stop control)
- ✅ Full LangGraph graph topology (mock node implementations)
- ✅ FastAPI server with all endpoints
- ✅ React frontend with SSE progress streaming
- ✅ Unit tests for all utility modules
- ⬜ Real LLM integration (Issue 2)
- ⬜ Real MCP tool integration (Issue 2)
- ⬜ Human-in-the-loop flow (Issue 2)
