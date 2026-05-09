"""
test_tools.py — Unit tests for individual tools.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent.tools.base import BaseTool, ToolResult
from agent.tools.search import SearchTool
from agent.tools.summarize import SummarizeTool


class TestToolResult:
    def test_bool_true_on_success(self):
        r = ToolResult(tool_name="t", query="q", content="c", success=True)
        assert bool(r) is True

    def test_bool_false_on_failure(self):
        r = ToolResult(tool_name="t", query="q", content="c", success=False)
        assert bool(r) is False

    def test_default_sources_empty(self):
        r = ToolResult(tool_name="t", query="q", content="c")
        assert r.sources == []


class TestSearchTool:

    def test_stub_mode_returns_success(self):
        """SearchTool should work in stub mode with no API key."""
        with patch.dict("os.environ", {}, clear=True):
            tool = SearchTool()
            result = tool.run("What is reinforcement learning?")

        assert result.success is True
        assert result.tool_name == "search"
        assert len(result.content) > 0
        assert "STUB" in result.content

    def test_tavily_failure_returns_failed_result(self):
        """API failure should return ToolResult with success=False."""
        mock_client = MagicMock()
        mock_client.search.side_effect = Exception("API error")

        with patch("agent.tools.search._TAVILY_AVAILABLE", True), \
             patch.dict("os.environ", {"TAVILY_API_KEY": "fake-key"}), \
             patch("agent.tools.search.TavilyClient", return_value=mock_client):
            tool = SearchTool()
            result = tool.run("test query")

        assert result.success is False
        assert "Search failed" in result.content

    def test_result_has_correct_tool_name(self):
        tool = SearchTool()
        result = tool.run("test")
        assert result.tool_name == "search"
        assert result.query == "test"


class TestSummarizeTool:

    def test_successful_call(self):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="A clear answer to the question.")]
        mock_response.usage.input_tokens = 50
        mock_response.usage.output_tokens = 30

        with patch("agent.tools.summarize.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_cls.return_value = mock_client

            tool = SummarizeTool()
            result = tool.run("What is a transformer architecture?")

        assert result.success is True
        assert result.content == "A clear answer to the question."
        assert result.tool_name == "summarize"
        assert "model" in result.metadata

    def test_api_error_returns_failed_result(self):
        import anthropic

        with patch("agent.tools.summarize.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_client.messages.create.side_effect = anthropic.APIError(
                message="API error", request=MagicMock(), body={}
            )
            mock_cls.return_value = mock_client

            tool = SummarizeTool()
            result = tool.run("test")

        assert result.success is False
        assert "Summarization failed" in result.content


class TestBaseToolABC:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseTool()  # type: ignore[abstract]

    def test_concrete_subclass_must_implement_run(self):
        class BrokenTool(BaseTool):
            name = "broken"
            description = "missing run"

        with pytest.raises(TypeError):
            BrokenTool()  # type: ignore[abstract]
