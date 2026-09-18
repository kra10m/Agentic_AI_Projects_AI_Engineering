import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from main import app
from agent import ResearchAgent
from schemas import ResearchReport, Citation

client = TestClient(app)

# 1. Health check test
def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

# 2. Tool dispatcher defensive tests
def test_agent_unknown_tool():
    agent = ResearchAgent()
    error_result = agent._execute_tool("non_existent_tool", "{}")
    assert "Unknown tool" in error_result

def test_agent_malformed_arguments():
    agent = ResearchAgent()
    error_result = agent._execute_tool("web_search", "not-a-json-string")
    assert "Invalid JSON arguments" in error_result

# 3. End-to-end API Contract Test (Mocked Agent)
def test_research_endpoint_success():
    mock_report = ResearchReport(
        title="Test Report",
        executive_summary="Executive summary test answer.",
        key_findings=["Finding A", "Finding B"],
        detailed_analysis="Comprehensive analysis goes here.",
        sources=[Citation(title="Source 1", url="https://example.com")]
    )

    with patch.object(ResearchAgent, "run", return_value=mock_report):
        payload = {"query": "Explain test-driven development in AI", "max_iterations": 2}
        response = client.post("/research", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "Test Report"
        assert len(data["key_findings"]) == 2
        assert data["sources"][0]["url"] == "https://example.com"

# 4. Input validation test
def test_research_endpoint_invalid_input():
    # Query too short (< 5 chars)
    response = client.post("/research", json={"query": "hi"})
    assert response.status_code == 422  # Pydantic validation error