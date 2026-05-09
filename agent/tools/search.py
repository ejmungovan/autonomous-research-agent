"""
search.py — Web search tool with Tavily primary + mock fallback.

To use Tavily: set TAVILY_API_KEY in your environment.
Falls back to a deterministic stub when the key is absent (useful for tests).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from agent.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

try:
    from tavily import TavilyClient  # type: ignore[import]

    _TAVILY_AVAILABLE = True
except ImportError:
    _TAVILY_AVAILABLE = False


class SearchTool(BaseTool):
    """
    Searches the web for current information.

    Priority:
        1. Tavily API  (if tavily-python installed + TAVILY_API_KEY set)
        2. Stub results (testing / offline mode)
    """

    name = "search"
    description = "Search the web for factual, current information on a topic."

    def __init__(self) -> None:
        self._api_key = os.environ.get("TAVILY_API_KEY", "")
        self._client: Any = None

        if _TAVILY_AVAILABLE and self._api_key:
            self._client = TavilyClient(api_key=self._api_key)
            logger.info("SearchTool: using Tavily.")
        else:
            logger.info("SearchTool: running in stub mode (no TAVILY_API_KEY).")

    def run(self, query: str) -> ToolResult:
        if self._client:
            return self._tavily_search(query)
        return self._stub_search(query)

    # ------------------------------------------------------------------

    def _tavily_search(self, query: str) -> ToolResult:
        try:
            response = self._client.search(
                query=query,
                search_depth="advanced",
                max_results=5,
            )
            results = response.get("results", [])
            content = "\n\n".join(
                f"**{r.get('title', 'Untitled')}**\n{r.get('content', '')}"
                for r in results
            )
            sources = [r.get("url", "") for r in results if r.get("url")]
            return ToolResult(
                tool_name=self.name,
                query=query,
                content=content or "No results found.",
                success=bool(results),
                sources=sources,
                metadata={"result_count": len(results)},
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Tavily search failed: %s", exc)
            return ToolResult(
                tool_name=self.name,
                query=query,
                content=f"Search failed: {exc}",
                success=False,
            )

    def _stub_search(self, query: str) -> ToolResult:
        """Deterministic stub for offline/test use."""
        content = (
            f"[STUB] Search results for: '{query}'\n\n"
            "1. Overview: This is a placeholder result returned when no search "
            "API key is configured. In production, real web results appear here.\n\n"
            "2. Key Facts: The stub result contains representative structure "
            "matching what the Tavily API would return.\n\n"
            "3. Additional Context: Set TAVILY_API_KEY to enable live results."
        )
        return ToolResult(
            tool_name=self.name,
            query=query,
            content=content,
            success=True,
            sources=["https://stub.example.com"],
            metadata={"mode": "stub"},
        )
