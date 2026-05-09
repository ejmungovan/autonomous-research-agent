# Changelog

All notable changes to this project will be documented in this file.
Format follows [Conventional Commits](https://www.conventionalcommits.org/).

---

## [1.0.0] — 2025-05-09

### feat: initial production scaffold

- `agent/core.py` — ResearchAgent main loop (Plan → Execute → Reflect → Report)
- `agent/planner.py` — LLM-based query decomposition into ResearchPlan
- `agent/reporter.py` — Evidence synthesis into structured markdown report
- `agent/tools/base.py` — Abstract BaseTool + ToolResult dataclass
- `agent/tools/search.py` — Tavily web search with offline stub fallback
- `agent/tools/summarize.py` — LLM-powered synthesis tool
- `agent/tools/registry.py` — Tool registry with fallback chain resolution
- `agent/memory/scratchpad.py` — Short-term evidence accumulator

### feat: FastAPI REST interface

- `api/main.py` — POST /research, GET /health, GET /tools endpoints
- Pydantic v2 request/response schemas with validation
- Lifespan-managed agent initialisation

### test: full test suite (28 tests)

- `tests/test_agent.py` — Core loop, registry, iteration limits, fallback behavior
- `tests/test_tools.py` — Tool-level tests with mocked APIs
- `tests/test_api.py` — API integration tests with FastAPI TestClient

### ci: GitHub Actions pipeline

- Multi-version matrix (Python 3.11, 3.12)
- Ruff linting, mypy type checks, pytest with coverage
- Codecov integration

### docs: production README

- Problem statement, architecture diagram, stack table
- Full installation guide, API usage examples
- Design decisions and roadmap
