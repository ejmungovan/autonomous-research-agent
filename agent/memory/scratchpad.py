"""
scratchpad.py — Short-term agent memory for accumulating evidence.

Tracks sub-question answers, sources, and tool metadata across
the full research loop. Provides a formatted summary for the reporter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.planner import ResearchPlan
    from agent.tools.base import ToolResult


@dataclass
class EvidenceItem:
    sub_question: str
    tool_name: str
    content: str
    success: bool
    sources: list[str] = field(default_factory=list)


class Scratchpad:
    """
    Accumulates research evidence across agent iterations.

    Responsibilities:
    - Store evidence per sub-question
    - Track unique sources
    - Produce a formatted summary for the Reporter
    """

    def __init__(self, query: str) -> None:
        self.query = query
        self.evidence: list[EvidenceItem] = []
        self._plan: "ResearchPlan | None" = None
        self._source_set: set[str] = set()

    def set_plan(self, plan: "ResearchPlan") -> None:
        self._plan = plan

    def add_evidence(
        self,
        sub_question: str,
        tool_name: str,
        result: "ToolResult",
    ) -> None:
        item = EvidenceItem(
            sub_question=sub_question,
            tool_name=tool_name,
            content=result.content,
            success=result.success,
            sources=result.sources,
        )
        self.evidence.append(item)
        self._source_set.update(result.sources)

    def all_sources(self) -> list[str]:
        return sorted(self._source_set)

    def summary(self) -> str:
        """Human-readable summary of all collected evidence."""
        if not self.evidence:
            return "No evidence collected."

        lines = [f"Research scratchpad for: {self.query!r}\n"]
        for i, item in enumerate(self.evidence, 1):
            status = "✓" if item.success else "✗"
            lines.append(f"[{i}] {status} {item.sub_question}")
            lines.append(f"    Tool: {item.tool_name}")
            lines.append(f"    Content preview: {item.content[:120].strip()}...")
            if item.sources:
                lines.append(f"    Sources: {', '.join(item.sources[:3])}")
            lines.append("")

        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.evidence)
