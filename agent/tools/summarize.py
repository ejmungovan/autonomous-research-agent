"""
summarize.py — LLM-powered synthesis and knowledge retrieval tool.

Used for sub-questions that can be answered from parametric knowledge,
without needing a live web search.
"""

from __future__ import annotations

import logging
import os

import anthropic

from agent.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

_SUMMARIZE_SYSTEM = """\
You are a precise research assistant. Answer the user's question clearly and
concisely using your knowledge. Structure your answer with:
- A direct 1–2 sentence answer
- 2–4 supporting points or key facts
- Any important caveats or nuances

Be factual. Do not fabricate sources. If uncertain, say so explicitly.
"""


class SummarizeTool(BaseTool):
    """
    Answers sub-questions using LLM parametric knowledge.

    Best for: definitions, explanations, conceptual questions, synthesis.
    Complement to SearchTool for factual/current information.
    """

    name = "summarize"
    description = "Answer conceptual or explanatory questions using LLM reasoning."

    def __init__(self, model: str = "claude-sonnet-4-20250514") -> None:
        self.model = model
        self._client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def run(self, query: str) -> ToolResult:
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=600,
                system=_SUMMARIZE_SYSTEM,
                messages=[{"role": "user", "content": query}],
            )
            content = response.content[0].text.strip()
            return ToolResult(
                tool_name=self.name,
                query=query,
                content=content,
                success=True,
                metadata={
                    "model": self.model,
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
            )
        except anthropic.APIError as exc:
            logger.error("SummarizeTool LLM call failed: %s", exc)
            return ToolResult(
                tool_name=self.name,
                query=query,
                content=f"Summarization failed: {exc}",
                success=False,
            )
