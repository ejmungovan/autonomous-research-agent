"""
reporter.py — Synthesises all collected evidence into a structured report.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import anthropic

if TYPE_CHECKING:
    from agent.memory.scratchpad import EvidenceItem
    from agent.planner import ResearchPlan

logger = logging.getLogger(__name__)

_REPORTER_SYSTEM = """\
You are a senior research analyst. Given a research goal and collected evidence,
produce a clear, structured report.

Format requirements:
## Executive Summary
2–3 sentence overview of key findings.

## Key Findings
Bullet-pointed findings, one per major point discovered.

## Analysis
2–3 paragraphs of deeper synthesis and context.

## Limitations & Gaps
What was NOT found or remains uncertain.

## Sources
List all cited sources.

Rules:
- Be concise and professional.
- Do not fabricate information beyond the provided evidence.
- If evidence is weak, say so in Limitations.
- Use markdown formatting.
"""


class Reporter:
    """Synthesises research evidence into a formatted markdown report."""

    def __init__(self, model: str = "claude-sonnet-4-20250514") -> None:
        self.model = model
        self._client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def generate(
        self,
        query: str,
        plan: "ResearchPlan",
        evidence: "list[EvidenceItem]",
    ) -> str:
        """Return a formatted markdown research report."""
        context = self._build_context(query, plan, evidence)
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=1500,
                system=_REPORTER_SYSTEM,
                messages=[{"role": "user", "content": context}],
            )
            return response.content[0].text.strip()
        except anthropic.APIError as exc:
            logger.error("Reporter LLM call failed: %s", exc)
            return self._fallback_report(query, evidence)

    # ------------------------------------------------------------------

    def _build_context(
        self,
        query: str,
        plan: "ResearchPlan",
        evidence: "list[EvidenceItem]",
    ) -> str:
        lines = [
            f"**Research Goal:** {plan.goal}",
            f"**Original Query:** {query}",
            "",
            "**Evidence Collected:**",
        ]
        for i, item in enumerate(evidence, 1):
            status = "SUCCESS" if item.success else "FAILED"
            lines.append(f"\n--- Evidence {i} ({status}) ---")
            lines.append(f"Sub-question: {item.sub_question}")
            lines.append(f"Tool used: {item.tool_name}")
            lines.append(f"Content:\n{item.content}")
            if item.sources:
                lines.append(f"Sources: {', '.join(item.sources)}")

        return "\n".join(lines)

    def _fallback_report(
        self,
        query: str,
        evidence: "list[EvidenceItem]",
    ) -> str:
        items = "\n".join(
            f"- {e.sub_question}: {e.content[:200]}..."
            for e in evidence
        )
        return (
            f"## Research Report: {query}\n\n"
            f"**Note:** Report generation encountered an error. Raw evidence:\n\n"
            f"{items}"
        )
