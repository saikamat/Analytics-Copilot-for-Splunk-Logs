"""
Integration tests for FastAPI endpoints - Simplified version.

Tests the complete API flow with mocked Bedrock to avoid AWS API calls.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from src.backend.app import app
from src.shared.bedrock_client import ValidationError, BedrockError
from src.shared.db_utils import QueryResult
from src.shared.result_summarizer import SummaryResult


@pytest.fixture(scope="module")
def test_client():
    """Create TestClient with lifespan context."""
    with TestClient(app) as client:
        yield client


class TestEndpoints:
    """Test API endpoints."""

    def test_root_endpoint(self, test_client):
        """Test root endpoint."""
        response = test_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Log Analytics API"

    def test_health_endpoint(self, test_client):
        """Test health check endpoint."""
        response = test_client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "database_connected" in data

    def test_query_endpoint_with_mock(self, test_client, test_db_with_fixtures):
        """Test query endpoint with mocked Bedrock."""
        mock_sql = "SELECT * FROM logs LIMIT 5;"
        
        with patch('src.backend.routes.sql_generator.generate_sql', return_value=mock_sql):
            response = test_client.post(
                "/api/query",
                json={"query": "show logs", "max_rows": 5}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "rows" in data
        assert "sql_query" in data

    def test_empty_query_validation(self, test_client):
        """Test empty query validation."""
        response = test_client.post("/api/query", json={"query": ""})
        assert response.status_code == 422

    def test_dangerous_sql_blocked(self, test_client):
        """Test SQL validation blocks dangerous queries."""
        with patch('src.backend.routes.sql_generator.generate_sql') as mock:
            mock.side_effect = ValidationError("Only SELECT queries allowed")
            response = test_client.post("/api/query", json={"query": "delete all"})
        
        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["success"] is False

    def test_bedrock_error_handling(self, test_client):
        """Test Bedrock API error handling."""
        with patch('src.backend.routes.sql_generator.generate_sql') as mock:
            mock.side_effect = BedrockError("API error")
            response = test_client.post("/api/query", json={"query": "test"})

        assert response.status_code == 502

    def test_query_too_long_validation(self, test_client):
        """Test query >500 chars fails validation."""
        response = test_client.post("/api/query", json={"query": "a" * 501})
        assert response.status_code == 422

    def test_invalid_max_rows(self, test_client):
        """Test invalid max_rows fails validation."""
        response = test_client.post("/api/query", json={"query": "test", "max_rows": 0})
        assert response.status_code == 422

        response = test_client.post("/api/query", json={"query": "test", "max_rows": 50001})
        assert response.status_code == 422

    def test_invalid_timeout(self, test_client):
        """Test invalid timeout fails validation."""
        response = test_client.post("/api/query", json={"query": "test", "timeout": 0})
        assert response.status_code == 422

        response = test_client.post("/api/query", json={"query": "test", "timeout": 301})
        assert response.status_code == 422

    def test_query_response_structure(self, test_client, test_db_with_fixtures):
        """Test query response has all required fields."""
        mock_sql = "SELECT * FROM logs LIMIT 1;"

        with patch('src.backend.routes.sql_generator.generate_sql', return_value=mock_sql):
            response = test_client.post("/api/query", json={"query": "test"})

        assert response.status_code == 200
        data = response.json()

        # Verify all required fields
        assert "success" in data
        assert "rows" in data
        assert "row_count" in data
        assert "column_names" in data
        assert "execution_time_ms" in data
        assert "truncated" in data
        assert "sql_query" in data

        # Verify types
        assert isinstance(data["success"], bool)
        assert isinstance(data["rows"], list)
        assert isinstance(data["row_count"], int)
        assert isinstance(data["execution_time_ms"], float)

    def test_error_response_structure(self, test_client):
        """Test error response has consistent structure."""
        with patch('src.backend.routes.sql_generator.generate_sql') as mock:
            mock.side_effect = ValidationError("Test error")
            response = test_client.post("/api/query", json={"query": "test"})

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "success" in data["detail"]
        assert "error" in data["detail"]
        assert "error_type" in data["detail"]
        assert data["detail"]["success"] is False

    def test_query_with_summary_enabled(self, test_client, test_db_with_fixtures):
        """Test query endpoint with summary generation enabled."""
        mock_sql = "SELECT * FROM logs LIMIT 5;"
        mock_summary = SummaryResult(
            summary="Found 5 logs across multiple services. Most are INFO level messages.",
            success=True,
            model_id="us.anthropic.claude-3-haiku-20240307-v1:0",
            execution_time_ms=1523.4
        )

        with patch('src.backend.routes.sql_generator.generate_sql', return_value=mock_sql), \
             patch('src.backend.app.result_summarizer') as mock_summarizer:
            # Configure mock summarizer to return our mock result
            mock_summarizer.summarize.return_value = mock_summary

            response = test_client.post(
                "/api/query",
                json={"query": "show logs", "max_rows": 5, "include_summary": True}
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        # Verify summary fields are present
        assert "summary" in data
        assert "summary_success" in data
        assert "summary_execution_time_ms" in data

        # Verify summary values
        assert data["summary"] == mock_summary.summary
        assert data["summary_success"] is True
        assert data["summary_execution_time_ms"] == 1523.4

    def test_query_with_summary_disabled(self, test_client, test_db_with_fixtures):
        """Test query endpoint with summary generation disabled."""
        mock_sql = "SELECT * FROM logs LIMIT 5;"

        with patch('src.backend.routes.sql_generator.generate_sql', return_value=mock_sql):
            response = test_client.post(
                "/api/query",
                json={"query": "show logs", "max_rows": 5, "include_summary": False}
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        # Verify summary fields are None when disabled
        assert data["summary"] is None
        assert data["summary_success"] is None
        assert data["summary_execution_time_ms"] is None

    def test_query_with_summary_default(self, test_client, test_db_with_fixtures):
        """Test query endpoint uses summary by default (include_summary defaults to True)."""
        mock_sql = "SELECT * FROM logs LIMIT 5;"
        mock_summary = SummaryResult(
            summary="Default summary test",
            success=True,
            model_id="us.anthropic.claude-3-haiku-20240307-v1:0",
            execution_time_ms=1500.0
        )

        with patch('src.backend.routes.sql_generator.generate_sql', return_value=mock_sql), \
             patch('src.backend.app.result_summarizer') as mock_summarizer:
            # Configure mock summarizer to return our mock result
            mock_summarizer.summarize.return_value = mock_summary

            # Don't specify include_summary - should default to True
            response = test_client.post(
                "/api/query",
                json={"query": "show logs", "max_rows": 5}
            )

        assert response.status_code == 200
        data = response.json()

        # Summary should be present since default is True
        assert data["summary"] is not None
        assert data["summary_success"] is True

    def test_summary_generation_failure_handling(self, test_client, test_db_with_fixtures):
        """Test that summary generation failures don't break the API."""
        mock_sql = "SELECT * FROM logs LIMIT 5;"

        with patch('src.backend.routes.sql_generator.generate_sql', return_value=mock_sql), \
             patch('src.backend.app.result_summarizer') as mock_summarizer:
            # Make summarize raise an exception
            mock_summarizer.summarize.side_effect = Exception("Bedrock API error")

            response = test_client.post(
                "/api/query",
                json={"query": "show logs", "include_summary": True}
            )

        # Query should still succeed even if summary fails
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        # Summary should indicate failure
        assert data["summary"] is not None
        assert "Summary generation failed" in data["summary"]
        assert data["summary_success"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
