"""
core.py — Autonomous Research Agent main loop.

Orchestrates: Plan → Tool Execution → Reflection → Report
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from agent.memory.scratchpad import Scratchpad
from agent.planner import Planner, ResearchPlan
from agent.reporter import Reporter
from agent.tools.base import ToolResult
from agent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Tunable parameters for the research agent."""

    max_iterations: int = 8
    max_reflection_retries: int = 2
    model: str = "claude-sonnet-4-20250514"
    verbose: bool = False


@dataclass
class ResearchResult:
    """Final structured output returned to callers."""

    query: str
    report: str
    sources: list[str]
    steps_taken: int
    plan: ResearchPlan
    scratchpad_summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "report": self.report,
            "sources": self.sources,
            "steps_taken": self.steps_taken,
            "plan": {
                "goal": self.plan.goal,
                "sub_questions": self.plan.sub_questions,
                "strategy": self.plan.strategy,
            },
            "scratchpad_summary": self.scratchpad_summary,
        }


class ResearchAgent:
    """
    Autonomous multi-step research agent.

    Architecture
    ------------
    1. Planner decomposes the query into sub-questions + strategy.
    2. Tool Router selects and executes the best tool per sub-question.
    3. Scratchpad accumulates evidence incrementally.
    4. Reflection loop evaluates sufficiency; retries if gaps remain.
    5. Reporter synthesises findings into a structured report.
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        config: AgentConfig | None = None,
    ) -> None:
        self.registry = tool_registry
        self.config = config or AgentConfig()
        self.planner = Planner(model=self.config.model)
        self.reporter = Reporter(model=self.config.model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, query: str) -> ResearchResult:
        """Execute a full research cycle for *query* and return the result."""
        logger.info("ResearchAgent starting: %s", query)
        memory = Scratchpad(query=query)

        # ── 1. Plan ────────────────────────────────────────────────────
        plan = self.planner.create_plan(query)
        memory.set_plan(plan)
        self._log(f"Plan created — {len(plan.sub_questions)} sub-questions")

        # ── 2. Execute + Reflect loop ──────────────────────────────────
        iterations = 0
        reflection_retries = 0

        for sub_q in plan.sub_questions:
            if iterations >= self.config.max_iterations:
                logger.warning("Max iterations reached; stopping early.")
                break

            tool_name = self._route(sub_q, plan.strategy)
            tool = self.registry.get(tool_name)

            self._log(f"[{iterations+1}] '{sub_q}' → tool: {tool_name}")

            result: ToolResult = tool.run(sub_q)
            memory.add_evidence(
                sub_question=sub_q,
                tool_name=tool_name,
                result=result,
            )

            # Reflection: if evidence is weak, retry with fallback tool
            if not result.success and reflection_retries < self.config.max_reflection_retries:
                fallback = self.registry.fallback_for(tool_name)
                if fallback:
                    self._log(f"Retrying '{sub_q}' with fallback: {fallback}")
                    result = self.registry.get(fallback).run(sub_q)
                    memory.add_evidence(sub_q, fallback, result)
                    reflection_retries += 1

            iterations += 1

        # ── 3. Report ──────────────────────────────────────────────────
        report_text = self.reporter.generate(
            query=query,
            plan=plan,
            evidence=memory.evidence,
        )

        return ResearchResult(
            query=query,
            report=report_text,
            sources=memory.all_sources(),
            steps_taken=iterations,
            plan=plan,
            scratchpad_summary=memory.summary(),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _route(self, sub_question: str, strategy: str) -> str:
        """Simple keyword-based tool router; override for ML-based routing."""
        q_lower = sub_question.lower()
        available = self.registry.available_tools()

        if strategy == "web" and "search" in available:
            return "search"
        if any(kw in q_lower for kw in ("latest", "recent", "current", "news")):
            return "search" if "search" in available else available[0]
        if any(kw in q_lower for kw in ("summarize", "explain", "define", "what is")):
            return "summarize" if "summarize" in available else available[0]

        return available[0]

    def _log(self, msg: str) -> None:
        if self.config.verbose:
            print(f"  [agent] {msg}")
        logger.debug(msg)
