"""
registry.py — Central registry for all agent tools.

Provides tool lookup, listing, and fallback chain resolution.
"""

from __future__ import annotations

from agent.tools.base import BaseTool


# Fallback chain: if primary tool fails, try the mapped tool
_FALLBACK_MAP: dict[str, str] = {
    "search": "summarize",
    "summarize": "search",
}


class ToolRegistry:
    """
    Manages tool registration and lookup.

    Usage
    -----
    registry = ToolRegistry()
    registry.register(SearchTool())
    registry.register(SummarizeTool())
    tool = registry.get("search")
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> "ToolRegistry":
        """Register a tool. Returns self for fluent chaining."""
        if not isinstance(tool, BaseTool):
            raise TypeError(f"Expected BaseTool subclass, got {type(tool)}")
        self._tools[tool.name] = tool
        return self

    def get(self, name: str) -> BaseTool:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not registered. Available: {list(self._tools)}")
        return self._tools[name]

    def available_tools(self) -> list[str]:
        return list(self._tools.keys())

    def fallback_for(self, tool_name: str) -> str | None:
        candidate = _FALLBACK_MAP.get(tool_name)
        if candidate and candidate in self._tools:
            return candidate
        return None

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __repr__(self) -> str:
        return f"<ToolRegistry tools={list(self._tools.keys())}>"


def build_default_registry() -> ToolRegistry:
    """
    Factory that builds a registry with all standard tools.
    Import and call this to get a ready-to-use registry.
    """
    from agent.tools.search import SearchTool
    from agent.tools.summarize import SummarizeTool

    registry = ToolRegistry()
    registry.register(SearchTool())
    registry.register(SummarizeTool())
    return registry
