"""
base.py — Abstract base class for all agent tools.

Every tool must implement `run(query) -> ToolResult`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    """Standardised envelope returned by every tool."""

    tool_name: str
    query: str
    content: str
    success: bool = True
    sources: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:  # pragma: no cover
        return self.success


class BaseTool(ABC):
    """
    Abstract base for all research tools.

    Subclasses must implement:
        name        — unique identifier string
        description — human-readable purpose (used for routing hints)
        run(query)  — execute the tool and return a ToolResult
    """

    name: str
    description: str

    @abstractmethod
    def run(self, query: str) -> ToolResult:
        """Execute the tool for *query* and return a ToolResult."""
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
