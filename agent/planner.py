"""
planner.py — Decomposes a research query into an actionable ResearchPlan.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field

import anthropic

logger = logging.getLogger(__name__)

_PLAN_SYSTEM_PROMPT = """\
You are an expert research strategist. Given a query, decompose it into a
structured research plan. Return ONLY valid JSON — no markdown, no preamble.

Schema:
{
  "goal": "<one sentence restatement of the research goal>",
  "strategy": "<web | synthesis | hybrid>",
  "sub_questions": ["<question 1>", "<question 2>", ...]
}

Rules:
- sub_questions: 2–5 focused, non-overlapping questions that together answer the goal.
- strategy:
    web      → answer requires current/factual data from external sources
    synthesis → answer can be derived from existing knowledge + reasoning
    hybrid   → mix of both
- Be concise. Every sub-question must be independently resolvable.
"""


@dataclass
class ResearchPlan:
    goal: str
    strategy: str
    sub_questions: list[str] = field(default_factory=list)


class Planner:
    """Uses an LLM to decompose a research query into a structured plan."""

    def __init__(self, model: str = "claude-sonnet-4-20250514") -> None:
        self.model = model
        self._client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def create_plan(self, query: str) -> ResearchPlan:
        """Return a ResearchPlan for *query*."""
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=512,
                system=_PLAN_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": query}],
            )
            raw = response.content[0].text.strip()
            data = json.loads(raw)
            return ResearchPlan(
                goal=data["goal"],
                strategy=data.get("strategy", "hybrid"),
                sub_questions=data.get("sub_questions", [query]),
            )
        except (json.JSONDecodeError, KeyError, anthropic.APIError) as exc:
            logger.warning("Planner LLM call failed (%s); using fallback plan.", exc)
            return self._fallback_plan(query)

    # ------------------------------------------------------------------
    def _fallback_plan(self, query: str) -> ResearchPlan:
        """Deterministic fallback when the LLM is unavailable."""
        return ResearchPlan(
            goal=query,
            strategy="hybrid",
            sub_questions=[
                f"What is {query}?",
                f"What are the key aspects of {query}?",
                f"What are recent developments related to {query}?",
            ],
        )
