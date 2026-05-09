"""
main.py — FastAPI service exposing the Research Agent via REST.

Endpoints:
  POST /research      — run a full research cycle
  GET  /health        — liveness probe
  GET  /tools         — list available tools
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from agent.core import AgentConfig, ResearchAgent
from agent.tools.registry import build_default_registry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App state ─────────────────────────────────────────────────────────────────
_registry = None
_agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _registry, _agent
    _registry = build_default_registry()
    _agent = ResearchAgent(tool_registry=_registry)
    logger.info("ResearchAgent initialised with tools: %s", _registry.available_tools())
    yield
    logger.info("ResearchAgent shutting down.")


app = FastAPI(
    title="Autonomous Research Agent",
    description=(
        "Multi-step LLM-powered research agent. "
        "Decomposes queries → searches → synthesises → reports."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request / Response schemas ─────────────────────────────────────────────────

class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=5, max_length=500, description="Research question")
    max_iterations: int = Field(default=6, ge=1, le=15)
    verbose: bool = Field(default=False)


class ResearchResponse(BaseModel):
    query: str
    report: str
    sources: list[str]
    steps_taken: int
    elapsed_seconds: float
    plan: dict


class HealthResponse(BaseModel):
    status: str
    version: str
    tools_available: list[str]


class ToolsResponse(BaseModel):
    tools: list[str]


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health() -> HealthResponse:
    """Liveness probe — confirms service is running and agent is loaded."""
    return HealthResponse(
        status="ok",
        version=app.version,
        tools_available=_registry.available_tools() if _registry else [],
    )


@app.get("/tools", response_model=ToolsResponse, tags=["System"])
async def list_tools() -> ToolsResponse:
    """List all tools registered in the agent."""
    if not _registry:
        raise HTTPException(status_code=503, detail="Agent not initialised")
    return ToolsResponse(tools=_registry.available_tools())


@app.post("/research", response_model=ResearchResponse, tags=["Agent"])
async def run_research(request: ResearchRequest) -> ResearchResponse:
    """
    Run a full autonomous research cycle.

    The agent will:
    1. Decompose your query into sub-questions
    2. Execute appropriate tools per sub-question
    3. Reflect and retry on weak evidence
    4. Synthesise findings into a structured report
    """
    if not _agent or not _registry:
        raise HTTPException(status_code=503, detail="Agent not initialised")

    config = AgentConfig(
        max_iterations=request.max_iterations,
        verbose=request.verbose,
    )
    agent = ResearchAgent(tool_registry=_registry, config=config)

    try:
        start = time.perf_counter()
        result = agent.run(query=request.query)
        elapsed = time.perf_counter() - start

        return ResearchResponse(
            query=result.query,
            report=result.report,
            sources=result.sources,
            steps_taken=result.steps_taken,
            elapsed_seconds=round(elapsed, 3),
            plan={
                "goal": result.plan.goal,
                "strategy": result.plan.strategy,
                "sub_questions": result.plan.sub_questions,
            },
        )
    except Exception as exc:
        logger.exception("Research cycle failed for query: %s", request.query)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
