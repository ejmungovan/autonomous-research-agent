"""
test_agent.py — Unit tests for the ResearchAgent core loop.

Uses mocks so tests run without API keys or network access.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent.core import AgentConfig, ResearchAgent
from agent.planner import ResearchPlan
from agent.tools.base import BaseTool, ToolResult
from agent.tools.registry import ToolRegistry


# ── Fixtures ───────────────────────────────────────────────────────────────────

class MockTool(BaseTool):
    name = "mock"
    description = "Mock tool for testing"

    def __init__(self, success: bool = True) -> None:
        self._success = success

    def run(self, query: str) -> ToolResult:
        return ToolResult(
            tool_name=self.name,
            query=query,
            content=f"Mock result for: {query}",
            success=self._success,
            sources=["https://mock.test/source"],
        )


def _make_registry(success: bool = True) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(MockTool(success=success))
    return registry


def _make_plan(sub_questions: list[str] | None = None) -> ResearchPlan:
    return ResearchPlan(
        goal="Test research goal",
        strategy="synthesis",
        sub_questions=sub_questions or ["What is topic A?", "What are key aspects of A?"],
    )


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestResearchAgent:

    def test_run_returns_result(self):
        registry = _make_registry()
        with patch("agent.core.Planner") as mock_planner_cls, \
             patch("agent.core.Reporter") as mock_reporter_cls:

            mock_planner = MagicMock()
            mock_planner.create_plan.return_value = _make_plan()
            mock_planner_cls.return_value = mock_planner

            mock_reporter = MagicMock()
            mock_reporter.generate.return_value = "## Test Report\nFindings here."
            mock_reporter_cls.return_value = mock_reporter

            agent = ResearchAgent(tool_registry=registry)
            result = agent.run("What is machine learning?")

        assert result.query == "What is machine learning?"
        assert result.report == "## Test Report\nFindings here."
        assert result.steps_taken == 2
        assert len(result.sources) > 0

    def test_max_iterations_respected(self):
        registry = _make_registry()
        many_questions = [f"Question {i}" for i in range(20)]

        with patch("agent.core.Planner") as mock_planner_cls, \
             patch("agent.core.Reporter") as mock_reporter_cls:

            mock_planner = MagicMock()
            mock_planner.create_plan.return_value = _make_plan(many_questions)
            mock_planner_cls.return_value = mock_planner

            mock_reporter = MagicMock()
            mock_reporter.generate.return_value = "Report."
            mock_reporter_cls.return_value = mock_reporter

            config = AgentConfig(max_iterations=3)
            agent = ResearchAgent(tool_registry=registry, config=config)
            result = agent.run("Big query")

        assert result.steps_taken == 3

    def test_fallback_tool_used_on_failure(self):
        """When primary tool fails, agent should try the fallback."""
        registry = ToolRegistry()
        failing_tool = MockTool(success=False)
        failing_tool.name = "search"
        backup_tool = MockTool(success=True)
        backup_tool.name = "summarize"

        registry.register(failing_tool)
        registry.register(backup_tool)

        with patch("agent.core.Planner") as mock_planner_cls, \
             patch("agent.core.Reporter") as mock_reporter_cls:

            mock_planner = MagicMock()
            mock_planner.create_plan.return_value = ResearchPlan(
                goal="test", strategy="web", sub_questions=["What is X?"]
            )
            mock_planner_cls.return_value = mock_planner
            mock_reporter = MagicMock()
            mock_reporter.generate.return_value = "Report."
            mock_reporter_cls.return_value = mock_reporter

            config = AgentConfig(max_reflection_retries=1)
            agent = ResearchAgent(tool_registry=registry, config=config)
            result = agent.run("Test query")

        assert result.steps_taken == 1

    def test_result_to_dict(self):
        registry = _make_registry()
        with patch("agent.core.Planner") as mp, patch("agent.core.Reporter") as mr:
            mp.return_value.create_plan.return_value = _make_plan()
            mr.return_value.generate.return_value = "Report text"

            agent = ResearchAgent(tool_registry=registry)
            result = agent.run("test")

        d = result.to_dict()
        assert "query" in d
        assert "report" in d
        assert "plan" in d
        assert "sub_questions" in d["plan"]


class TestToolRegistry:

    def test_register_and_get(self):
        registry = ToolRegistry()
        tool = MockTool()
        registry.register(tool)
        assert registry.get("mock") is tool

    def test_get_missing_raises(self):
        registry = ToolRegistry()
        with pytest.raises(KeyError):
            registry.get("nonexistent")

    def test_available_tools(self):
        registry = ToolRegistry()
        registry.register(MockTool())
        assert "mock" in registry.available_tools()

    def test_fallback_returns_registered(self):
        registry = ToolRegistry()
        search = MockTool()
        search.name = "search"
        summarize = MockTool()
        summarize.name = "summarize"
        registry.register(search)
        registry.register(summarize)
        assert registry.fallback_for("search") == "summarize"

    def test_fallback_returns_none_if_not_registered(self):
        registry = ToolRegistry()
        search = MockTool()
        search.name = "search"
        registry.register(search)
        assert registry.fallback_for("search") is None

    def test_fluent_chaining(self):
        registry = ToolRegistry()
        result = registry.register(MockTool())
        assert result is registry
