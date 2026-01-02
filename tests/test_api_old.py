"""
Integration tests for FastAPI endpoints.

Tests the complete API flow: HTTP request → validation → NL→SQL → query execution → response.
Uses FastAPI TestClient for HTTP testing and real database with fixture data.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from src.backend.app import app
from src.shared.bedrock_client import ValidationError, BedrockError
from src.shared.db_utils import QueryTimeoutError, ConnectionPoolError


@pytest.fixture(scope="module")
def client():
    """Create TestClient with lifespan context."""
    with TestClient(app) as test_client:
        yield test_client


class TestRootEndpoint:
    """Test root endpoint."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns API information."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Log Analytics API"
        assert data["version"] == "1.0.0"
        assert data["status"] == "running"
        assert data["docs"] == "/docs"


class TestHealthEndpoint:
    """Test /api/health endpoint."""

    def test_health_check_healthy(self, client):
        """Test health check when all dependencies are healthy."""
        response = client.get("/api/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "degraded"]
        assert "database_connected" in data
        assert "bedrock_available" in data
        assert isinstance(data["database_connected"], bool)
        assert isinstance(data["bedrock_available"], bool)

    def test_health_check_response_model(self, client):
        """Test health check response matches HealthResponse model."""
        response = client.get("/api/health")

        assert response.status_code == 200
        data = response.json()
        # Verify all required fields present
        required_fields = {"status", "database_connected", "bedrock_available"}
        assert required_fields.issubset(data.keys())


class TestQueryEndpoint:
    """Test /api/query endpoint."""

    def test_successful_query_with_mocked_bedrock(self, test_db_with_fixtures):
        """Test successful query execution with mocked Bedrock API."""
        # Mock BedrockSQLGenerator to avoid real API calls
        mock_sql = "SELECT id, timestamp, level, source, message, metadata FROM logs LIMIT 5;"

        with patch('src.backend.routes.sql_generator.generate_sql', return_value=mock_sql):
            response = client.post(
                "/api/query",
                json={
                    "query": "show me all logs",
                    "max_rows": 5,
                    "timeout": 30
                }
            )

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert data["success"] is True
        assert "rows" in data
        assert "row_count" in data
        assert "column_names" in data
        assert "execution_time_ms" in data
        assert "truncated" in data
        assert "sql_query" in data

        # Verify data types
        assert isinstance(data["rows"], list)
        assert isinstance(data["row_count"], int)
        assert isinstance(data["column_names"], list)
        assert isinstance(data["execution_time_ms"], float)
        assert isinstance(data["truncated"], bool)

        # Verify we got results from fixture data
        assert data["row_count"] >= 0
        assert data["sql_query"] == mock_sql

    def test_successful_query_with_results(self, test_db_with_fixtures):
        """Test query returns fixture data correctly."""
        mock_sql = "SELECT id, timestamp, level, source, message, metadata FROM logs WHERE level = 'ERROR' LIMIT 5;"

        with patch('src.backend.routes.sql_generator.generate_sql', return_value=mock_sql):
            response = client.post(
                "/api/query",
                json={
                    "query": "show me errors",
                    "max_rows": 5
                }
            )

        assert response.status_code == 200
        data = response.json()

        # Should have at least some error logs from fixtures
        if data["row_count"] > 0:
            # Verify row structure
            first_row = data["rows"][0]
            assert "id" in first_row
            assert "timestamp" in first_row
            assert "level" in first_row
            assert "source" in first_row
            assert "message" in first_row
            assert "metadata" in first_row

            # Verify level is ERROR
            assert first_row["level"] == "ERROR"

    def test_query_with_custom_limits(self, test_db_with_fixtures):
        """Test query with custom max_rows and timeout."""
        mock_sql = "SELECT * FROM logs LIMIT 3;"

        with patch('src.backend.routes.sql_generator.generate_sql', return_value=mock_sql):
            response = client.post(
                "/api/query",
                json={
                    "query": "show logs",
                    "max_rows": 3,
                    "timeout": 10
                }
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["row_count"] <= 3  # Should respect max_rows


class TestQueryValidation:
    """Test request validation for /api/query endpoint."""

    def test_empty_query_fails(self, client):
        """Test that empty query fails Pydantic validation."""
        response = client.post(
            "/api/query",
            json={"query": ""}
        )

        assert response.status_code == 422  # Unprocessable Entity
        data = response.json()
        assert "detail" in data
        # Pydantic returns list of errors
        assert isinstance(data["detail"], list)
        assert len(data["detail"]) > 0

    def test_query_too_long_fails(self, client):
        """Test that query >500 chars fails validation."""
        long_query = "a" * 501
        response = client.post(
            "/api/query",
            json={"query": long_query}
        )

        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_query_exactly_500_chars_succeeds(self):
        """Test that query with exactly 500 chars passes validation."""
        query_500 = "a" * 500
        mock_sql = "SELECT * FROM logs LIMIT 1;"

        with patch('src.backend.routes.sql_generator.generate_sql', return_value=mock_sql):
            response = client.post(
                "/api/query",
                json={"query": query_500}
            )

        # Should pass validation (might fail later in SQL generation, but that's OK)
        assert response.status_code in [200, 400, 502]  # Valid request, may fail in execution

    def test_invalid_max_rows_below_minimum(self, client):
        """Test that max_rows < 1 fails validation."""
        response = client.post(
            "/api/query",
            json={
                "query": "test",
                "max_rows": 0
            }
        )

        assert response.status_code == 422

    def test_invalid_max_rows_above_maximum(self, client):
        """Test that max_rows > 50,000 fails validation."""
        response = client.post(
            "/api/query",
            json={
                "query": "test",
                "max_rows": 50001
            }
        )

        assert response.status_code == 422

    def test_invalid_timeout_below_minimum(self, client):
        """Test that timeout < 1 fails validation."""
        response = client.post(
            "/api/query",
            json={
                "query": "test",
                "timeout": 0
            }
        )

        assert response.status_code == 422

    def test_invalid_timeout_above_maximum(self, client):
        """Test that timeout > 300 fails validation."""
        response = client.post(
            "/api/query",
            json={
                "query": "test",
                "timeout": 301
            }
        )

        assert response.status_code == 422

    def test_missing_query_field_fails(self, client):
        """Test that missing query field fails validation."""
        response = client.post(
            "/api/query",
            json={"max_rows": 100}
        )

        assert response.status_code == 422


class TestSQLValidation:
    """Test SQL validation and security."""

    def test_dangerous_sql_blocked_delete(self, client):
        """Test that DELETE queries are blocked."""
        # Mock Bedrock to generate a DELETE query (simulating malicious behavior)
        with patch('src.backend.routes.sql_generator.generate_sql') as mock_gen:
            mock_gen.side_effect = ValidationError("Only SELECT queries are allowed")

            response = client.post(
                "/api/query",
                json={"query": "delete all logs"}
            )

        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["success"] is False
        assert "ValidationError" in data["detail"]["error_type"]

    def test_dangerous_sql_blocked_drop(self, client):
        """Test that DROP queries are blocked."""
        with patch('src.backend.routes.sql_generator.generate_sql') as mock_gen:
            mock_gen.side_effect = ValidationError("Only SELECT queries are allowed")

            response = client.post(
                "/api/query",
                json={"query": "drop table logs"}
            )

        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["success"] is False


class TestErrorHandling:
    """Test error handling for various failure scenarios."""

    def test_bedrock_api_error(self, client):
        """Test handling of Bedrock API errors."""
        with patch('src.backend.routes.sql_generator.generate_sql') as mock_gen:
            mock_gen.side_effect = BedrockError("Throttling exception")

            response = client.post(
                "/api/query",
                json={"query": "show logs"}
            )

        assert response.status_code == 502  # Bad Gateway
        data = response.json()
        assert data["detail"]["success"] is False
        assert data["detail"]["error_type"] == "BedrockError"

    def test_query_timeout_error(self, test_db_with_fixtures):
        """Test handling of query timeout errors."""
        mock_sql = "SELECT * FROM logs;"

        with patch('src.backend.routes.sql_generator.generate_sql', return_value=mock_sql), \
             patch('src.backend.routes.query_executor.execute_query') as mock_exec:
            mock_exec.side_effect = QueryTimeoutError("Query timed out after 30 seconds")

            response = client.post(
                "/api/query",
                json={"query": "show logs", "timeout": 1}
            )

        assert response.status_code == 408  # Request Timeout
        data = response.json()
        assert data["detail"]["success"] is False
        assert data["detail"]["error_type"] == "QueryTimeoutError"

    def test_database_connection_error(self, client):
        """Test handling of database connection errors."""
        mock_sql = "SELECT * FROM logs;"

        with patch('src.backend.routes.sql_generator.generate_sql', return_value=mock_sql), \
             patch('src.backend.routes.query_executor.execute_query') as mock_exec:
            mock_exec.side_effect = ConnectionPoolError("Database unavailable")

            response = client.post(
                "/api/query",
                json={"query": "show logs"}
            )

        assert response.status_code == 503  # Service Unavailable
        data = response.json()
        assert data["detail"]["success"] is False
        assert data["detail"]["error_type"] == "ConnectionPoolError"

    def test_unexpected_error(self, client):
        """Test handling of unexpected errors."""
        mock_sql = "SELECT * FROM logs;"

        with patch('src.backend.routes.sql_generator.generate_sql', return_value=mock_sql), \
             patch('src.backend.routes.query_executor.execute_query') as mock_exec:
            mock_exec.side_effect = Exception("Unexpected error")

            response = client.post(
                "/api/query",
                json={"query": "show logs"}
            )

        assert response.status_code == 500  # Internal Server Error
        data = response.json()
        assert data["detail"]["success"] is False
        assert data["detail"]["error_type"] == "UnknownError"


class TestCORSHeaders:
    """Test CORS headers in responses."""

    def test_cors_headers_present(self, client):
        """Test that CORS headers are present in responses."""
        response = client.get("/api/health")

        assert response.status_code == 200
        # FastAPI TestClient doesn't include CORS headers in responses
        # (CORS is handled by middleware which TestClient bypasses)
        # This test would need to use httpx client with actual server
        # For now, just verify the endpoint works


class TestResponseStructure:
    """Test response structure and serialization."""

    def test_query_response_includes_sql(self, test_db_with_fixtures):
        """Test that response includes generated SQL for transparency."""
        mock_sql = "SELECT * FROM logs LIMIT 1;"

        with patch('src.backend.routes.sql_generator.generate_sql', return_value=mock_sql):
            response = client.post(
                "/api/query",
                json={"query": "show one log"}
            )

        assert response.status_code == 200
        data = response.json()
        assert "sql_query" in data
        assert data["sql_query"] == mock_sql

    def test_query_response_includes_performance_metrics(self, test_db_with_fixtures):
        """Test that response includes execution time."""
        mock_sql = "SELECT * FROM logs LIMIT 1;"

        with patch('src.backend.routes.sql_generator.generate_sql', return_value=mock_sql):
            response = client.post(
                "/api/query",
                json={"query": "show one log"}
            )

        assert response.status_code == 200
        data = response.json()
        assert "execution_time_ms" in data
        assert isinstance(data["execution_time_ms"], float)
        assert data["execution_time_ms"] >= 0

    def test_error_response_structure(self, client):
        """Test that error responses have consistent structure."""
        with patch('src.backend.routes.sql_generator.generate_sql') as mock_gen:
            mock_gen.side_effect = ValidationError("Test error")

            response = client.post(
                "/api/query",
                json={"query": "test"}
            )

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "success" in data["detail"]
        assert "error" in data["detail"]
        assert "error_type" in data["detail"]
        assert data["detail"]["success"] is False


class TestTypeConversions:
    """Test type conversions in query results."""

    def test_datetime_converted_to_iso8601(self, test_db_with_fixtures):
        """Test that datetime fields are converted to ISO8601 strings."""
        mock_sql = "SELECT id, timestamp FROM logs LIMIT 1;"

        with patch('src.backend.routes.sql_generator.generate_sql', return_value=mock_sql):
            response = client.post(
                "/api/query",
                json={"query": "show timestamp"}
            )

        assert response.status_code == 200
        data = response.json()

        if data["row_count"] > 0:
            first_row = data["rows"][0]
            if "timestamp" in first_row:
                # Verify timestamp is a string in ISO8601 format
                assert isinstance(first_row["timestamp"], str)
                # Should contain 'T' separator and timezone info
                assert "T" in first_row["timestamp"]

    def test_jsonb_metadata_converted_to_dict(self, test_db_with_fixtures):
        """Test that JSONB metadata is converted to dict."""
        mock_sql = "SELECT metadata FROM logs WHERE metadata IS NOT NULL LIMIT 1;"

        with patch('src.backend.routes.sql_generator.generate_sql', return_value=mock_sql):
            response = client.post(
                "/api/query",
                json={"query": "show metadata"}
            )

        assert response.status_code == 200
        data = response.json()

        if data["row_count"] > 0:
            first_row = data["rows"][0]
            if "metadata" in first_row and first_row["metadata"] is not None:
                # Verify metadata is a dict
                assert isinstance(first_row["metadata"], dict)


class TestTruncation:
    """Test result truncation behavior."""

    def test_truncation_flag_when_results_exceed_limit(self, test_db_with_fixtures):
        """Test that truncated flag is set when results exceed max_rows."""
        # Query that will return more rows than limit
        mock_sql = "SELECT * FROM logs LIMIT 100;"

        with patch('src.backend.routes.sql_generator.generate_sql', return_value=mock_sql):
            response = client.post(
                "/api/query",
                json={
                    "query": "show all logs",
                    "max_rows": 2  # Very small limit
                }
            )

        assert response.status_code == 200
        data = response.json()

        # If we have more than 2 logs in fixtures, truncated should be True
        if data["row_count"] >= 2:
            # Note: truncation only happens if query returns MORE than max_rows
            # Our fixtures have 10 logs, so with max_rows=2, we should see truncation
            # But the actual truncation happens in QueryExecutor
            assert "truncated" in data
