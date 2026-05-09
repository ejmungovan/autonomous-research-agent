# 🔍 Autonomous Research Agent

[![CI](https://github.com/ejmungovan/autonomous-research-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/ejmungovan/autonomous-research-agent/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-style **autonomous multi-step research agent** that decomposes queries,
orchestrates tool use, applies self-reflection, and synthesises structured reports —
all without human-in-the-loop intervention.

---

## Problem Statement

Research tasks typically require multiple steps: breaking down a question,
gathering information from different sources, evaluating evidence quality,
and synthesising a coherent answer. Current LLM wrappers either do this in
a single shot (losing multi-step reasoning) or require brittle hand-coded
pipelines. This project implements a clean, extensible **agentic loop** that
handles the full cycle autonomously.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     ResearchAgent.run(query)                │
│                                                             │
│  ┌──────────┐    ┌────────────────────────────────────┐     │
│  │ Planner  │───▶│  ResearchPlan                       │     │
│  │  (LLM)   │    │  - goal                             │     │
│  └──────────┘    │  - strategy: web | synthesis | hybrid│    │
│                  │  - sub_questions: [q1, q2, ...]      │     │
│                  └────────────────┬───────────────────┘     │
│                                   │                         │
│              ┌────────────────────▼──────────────────┐      │
│              │         Execution Loop (per sub-q)     │      │
│              │                                        │      │
│              │   ToolRouter ──▶ SearchTool            │      │
│              │                ──▶ SummarizeTool       │      │
│              │                                        │      │
│              │   Reflection: if weak evidence →       │      │
│              │     retry with fallback tool           │      │
│              └────────────────┬───────────────────────┘      │
│                               │                             │
│              ┌────────────────▼──────────────────────┐      │
│              │            Scratchpad (memory)         │      │
│              │  accumulates evidence + sources        │      │
│              └────────────────┬──────────────────────┘      │
│                               │                             │
│              ┌────────────────▼──────────────────────┐      │
│              │         Reporter (LLM synthesis)       │      │
│              │  → structured markdown report          │      │
│              └────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

---

## Technical Stack

| Layer | Technology |
|---|---|
| LLM Backbone | Anthropic Claude (claude-sonnet-4) |
| Web Search | Tavily API (graceful stub fallback) |
| API Server | FastAPI + Uvicorn |
| Data Validation | Pydantic v2 |
| Testing | pytest + pytest-cov |
| Linting / Types | Ruff + mypy |
| CI | GitHub Actions |
| Packaging | pyproject.toml |

---

## Project Structure

```
autonomous-research-agent/
├── agent/
│   ├── core.py           # Main agent loop (Plan → Execute → Reflect → Report)
│   ├── planner.py        # LLM-based query decomposition
│   ├── reporter.py       # Evidence synthesis → structured report
│   ├── tools/
│   │   ├── base.py       # Abstract BaseTool + ToolResult dataclass
│   │   ├── search.py     # Web search (Tavily API + stub fallback)
│   │   ├── summarize.py  # LLM-based synthesis tool
│   │   └── registry.py   # Tool registration + fallback chain
│   └── memory/
│       └── scratchpad.py # Short-term evidence accumulator
├── api/
│   └── main.py           # FastAPI REST interface
├── tests/
│   ├── test_agent.py     # Core loop unit tests
│   ├── test_tools.py     # Tool-level tests with mocks
│   └── test_api.py       # API integration tests
├── .github/
│   └── workflows/ci.yml  # GitHub Actions CI
├── requirements.txt
├── requirements-dev.txt
└── pyproject.toml
```

---

## Installation

### 1. Clone & create virtualenv
```bash
git clone https://github.com/ejmungovan/autonomous-research-agent.git
cd autonomous-research-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env:
# ANTHROPIC_API_KEY=your_key_here
# TAVILY_API_KEY=your_key_here  (optional — stub mode works without it)
```

### 3. Run the API server
```bash
uvicorn api.main:app --reload
# → http://localhost:8000/docs
```

### 4. Run the agent directly (Python)
```python
from agent.core import ResearchAgent, AgentConfig
from agent.tools.registry import build_default_registry

agent = ResearchAgent(
    tool_registry=build_default_registry(),
    config=AgentConfig(verbose=True),
)
result = agent.run("What are the key challenges in multi-agent LLM systems?")
print(result.report)
```

---

## API Usage

### POST /research
```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key challenges in federated learning?", "max_iterations": 6}'
```

**Response:**
```json
{
  "query": "What are the key challenges in federated learning?",
  "report": "## Executive Summary\n...",
  "sources": ["https://..."],
  "steps_taken": 4,
  "elapsed_seconds": 3.21,
  "plan": {
    "goal": "Identify and explain key challenges in federated learning",
    "strategy": "hybrid",
    "sub_questions": ["What is federated learning?", "..."]
  }
}
```

### GET /health
```bash
curl http://localhost:8000/health
```

Interactive docs: `http://localhost:8000/docs`

---

## Running Tests

```bash
# All tests with coverage
pytest tests/ --cov=agent --cov=api --cov-report=term-missing

# Fast unit tests only
pytest tests/test_agent.py tests/test_tools.py -v

# Lint
ruff check .
mypy agent/ api/ --ignore-missing-imports
```

---

## Design Decisions

**Why a custom agent loop instead of LangChain?**
Explicit control over the execution graph makes the system easier to debug,
test, and extend. Every step is observable and individually testable.

**Why Pydantic v2 for tool results?**
Strong typing at the tool boundary prevents subtle data shape bugs as the
tool set grows. ToolResult is a dataclass for performance; request/response
schemas are Pydantic for automatic API validation.

**Why a stub mode for SearchTool?**
Tests should never require API keys or network access. The stub provides a
structurally identical response so the agent loop is fully testable offline.

---

## Roadmap

- [ ] Streaming API endpoint (Server-Sent Events)
- [ ] Persistent memory store (SQLite / Redis)
- [ ] ML-based tool router (replace keyword heuristic)
- [ ] Async tool execution (parallel sub-question resolution)
- [ ] Docker + docker-compose deployment
- [ ] Evaluation harness with benchmark queries
- [ ] Web UI (Streamlit or React)

---

## License

MIT — see [LICENSE](LICENSE)
