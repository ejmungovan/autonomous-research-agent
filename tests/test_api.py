"""
test_api.py — Integration tests for the FastAPI layer.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from agent.core import ResearchResult
from agent.planner import ResearchPlan


@pytest.fixture()
def mock_result() -> ResearchResult:
    return ResearchResult(
        query="What is federated learning?",
        report="## Report\nFederated learning is...",
        sources=["https://example.com/fl"],
        steps_taken=3,
        plan=ResearchPlan(
            goal="Understand federated learning",
            strategy="synthesis",
            sub_questions=["What is FL?", "How does FL work?", "FL use cases?"],
        ),
        scratchpad_summary="Evidence collected across 3 sub-questions.",
    )


@pytest.fixture()
def client(mock_result):
    with patch("api.main.build_default_registry") as mock_reg, \
         patch("api.main.ResearchAgent") as mock_agent_cls:

        mock_registry = MagicMock()
        mock_registry.available_tools.return_value = ["search", "summarize"]
        mock_reg.return_value = mock_registry

        mock_agent = MagicMock()
        mock_agent.run.return_value = mock_result
        mock_agent_cls.return_value = mock_agent

        from api.main import app
        with TestClient(app) as c:
            yield c


class TestHealthEndpoint:
    def test_health_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_health_includes_tools(self, client):
        response = client.get("/health")
        assert "tools_available" in response.json()


class TestToolsEndpoint:
    def test_tools_lists_registered(self, client):
        response = client.get("/tools")
        assert response.status_code == 200
        data = response.json()
        assert "tools" in data
        assert isinstance(data["tools"], list)


class TestResearchEndpoint:
    def test_research_returns_report(self, client):
        response = client.post(
            "/research",
            json={"query": "What is federated learning?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "report" in data
        assert "sources" in data
        assert "steps_taken" in data
        assert "elapsed_seconds" in data
        assert "plan" in data

    def test_research_plan_has_sub_questions(self, client):
        response = client.post("/research", json={"query": "Explain transformers"})
        plan = response.json()["plan"]
        assert "sub_questions" in plan
        assert isinstance(plan["sub_questions"], list)

    def test_research_short_query_rejected(self, client):
        response = client.post("/research", json={"query": "Hi"})
        assert response.status_code == 422  # Pydantic validation error

    def test_research_empty_query_rejected(self, client):
        response = client.post("/research", json={"query": ""})
        assert response.status_code == 422

    def test_research_max_iterations_param(self, client):
        response = client.post(
            "/research",
            json={"query": "What is contrastive learning?", "max_iterations": 3},
        )
        assert response.status_code == 200
