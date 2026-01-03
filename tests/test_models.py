"""
Unit tests for Pydantic models.

Tests request/response validation, field constraints, and default values.
"""

import pytest
from pydantic import ValidationError

from src.backend.models import QueryRequest, QueryResponse, ErrorResponse, HealthResponse


class TestQueryRequest:
    """Test QueryRequest model validation."""

    def test_valid_request_with_defaults(self):
        """Test valid request with default values."""
        request = QueryRequest(query="show me errors")

        assert request.query == "show me errors"
        assert request.max_rows == 10000  # Default
        assert request.timeout == 30  # Default

    def test_valid_request_with_custom_values(self):
        """Test valid request with custom values."""
        request = QueryRequest(
            query="show nginx errors from yesterday",
            max_rows=100,
            timeout=60
        )

        assert request.query == "show nginx errors from yesterday"
        assert request.max_rows == 100
        assert request.timeout == 60

    def test_empty_query_fails(self):
        """Test that empty query fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(query="")

        errors = exc_info.value.errors()
        assert any(error["loc"] == ("query",) for error in errors)

    def test_query_too_long_fails(self):
        """Test that query >500 chars fails validation."""
        long_query = "a" * 501

        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(query=long_query)

        errors = exc_info.value.errors()
        assert any(error["loc"] == ("query",) for error in errors)

    def test_query_exactly_500_chars_succeeds(self):
        """Test that query with exactly 500 chars succeeds."""
        query_500 = "a" * 500
        request = QueryRequest(query=query_500)

        assert len(request.query) == 500

    def test_max_rows_below_minimum_fails(self):
        """Test that max_rows < 1 fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(query="test", max_rows=0)

        errors = exc_info.value.errors()
        assert any(error["loc"] == ("max_rows",) for error in errors)

    def test_max_rows_above_maximum_fails(self):
        """Test that max_rows > 50,000 fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(query="test", max_rows=50001)

        errors = exc_info.value.errors()
        assert any(error["loc"] == ("max_rows",) for error in errors)

    def test_max_rows_at_boundaries_succeeds(self):
        """Test that max_rows at boundaries (1 and 50,000) succeeds."""
        request_min = QueryRequest(query="test", max_rows=1)
        request_max = QueryRequest(query="test", max_rows=50000)

        assert request_min.max_rows == 1
        assert request_max.max_rows == 50000

    def test_timeout_below_minimum_fails(self):
        """Test that timeout < 1 fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(query="test", timeout=0)

        errors = exc_info.value.errors()
        assert any(error["loc"] == ("timeout",) for error in errors)

    def test_timeout_above_maximum_fails(self):
        """Test that timeout > 300 fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(query="test", timeout=301)

        errors = exc_info.value.errors()
        assert any(error["loc"] == ("timeout",) for error in errors)

    def test_timeout_at_boundaries_succeeds(self):
        """Test that timeout at boundaries (1 and 300) succeeds."""
        request_min = QueryRequest(query="test", timeout=1)
        request_max = QueryRequest(query="test", timeout=300)

        assert request_min.timeout == 1
        assert request_max.timeout == 300

    def test_missing_query_fails(self):
        """Test that missing query field fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(max_rows=100)

        errors = exc_info.value.errors()
        assert any(error["loc"] == ("query",) for error in errors)


class TestQueryResponse:
    """Test QueryResponse model validation."""

    def test_valid_response_minimal(self):
        """Test valid response with minimal required fields."""
        response = QueryResponse(
            rows=[],
            row_count=0,
            column_names=[],
            execution_time_ms=10.5
        )

        assert response.success is True  # Default
        assert response.rows == []
        assert response.row_count == 0
        assert response.column_names == []
        assert response.execution_time_ms == 10.5
        assert response.truncated is False  # Default
        assert response.sql_query is None  # Default

    def test_valid_response_with_data(self):
        """Test valid response with query results."""
        response = QueryResponse(
            success=True,
            rows=[
                {"id": 1, "level": "ERROR", "message": "Test error"},
                {"id": 2, "level": "WARN", "message": "Test warning"}
            ],
            row_count=2,
            column_names=["id", "level", "message"],
            execution_time_ms=45.2,
            truncated=False,
            sql_query="SELECT * FROM logs LIMIT 2"
        )

        assert response.success is True
        assert len(response.rows) == 2
        assert response.row_count == 2
        assert response.column_names == ["id", "level", "message"]
        assert response.execution_time_ms == 45.2
        assert response.truncated is False
        assert response.sql_query == "SELECT * FROM logs LIMIT 2"

    def test_response_with_truncated_results(self):
        """Test response with truncated flag."""
        response = QueryResponse(
            rows=[{"id": i} for i in range(10000)],
            row_count=10000,
            column_names=["id"],
            execution_time_ms=120.5,
            truncated=True
        )

        assert response.truncated is True
        assert response.row_count == 10000

    def test_missing_required_fields_fails(self):
        """Test that missing required fields fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            QueryResponse(rows=[], row_count=0)  # Missing column_names and execution_time_ms

        errors = exc_info.value.errors()
        assert len(errors) >= 2  # At least 2 missing fields


class TestErrorResponse:
    """Test ErrorResponse model validation."""

    def test_valid_error_response_minimal(self):
        """Test valid error response with minimal fields."""
        response = ErrorResponse(error="Something went wrong")

        assert response.success is False  # Default
        assert response.error == "Something went wrong"
        assert response.error_type is None  # Default

    def test_valid_error_response_with_type(self):
        """Test valid error response with error type."""
        response = ErrorResponse(
            success=False,
            error="SQL validation failed",
            error_type="ValidationError"
        )

        assert response.success is False
        assert response.error == "SQL validation failed"
        assert response.error_type == "ValidationError"

    def test_missing_error_field_fails(self):
        """Test that missing error field fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            ErrorResponse(error_type="SomeError")  # Missing required 'error' field

        errors = exc_info.value.errors()
        assert any(error["loc"] == ("error",) for error in errors)


class TestHealthResponse:
    """Test HealthResponse model validation."""

    def test_valid_health_response_healthy(self):
        """Test valid health response when healthy."""
        response = HealthResponse(
            status="healthy",
            database_connected=True,
            bedrock_available=True
        )

        assert response.status == "healthy"
        assert response.database_connected is True
        assert response.bedrock_available is True

    def test_valid_health_response_degraded(self):
        """Test valid health response when degraded."""
        response = HealthResponse(
            status="degraded",
            database_connected=False,
            bedrock_available=True
        )

        assert response.status == "degraded"
        assert response.database_connected is False
        assert response.bedrock_available is True

    def test_missing_required_fields_fails(self):
        """Test that missing required fields fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            HealthResponse(status="healthy")  # Missing database_connected and bedrock_available

        errors = exc_info.value.errors()
        assert len(errors) >= 2  # At least 2 missing fields


class TestModelSerialization:
    """Test model serialization to JSON."""

    def test_query_request_serialization(self):
        """Test QueryRequest serializes to JSON correctly."""
        request = QueryRequest(query="test query", max_rows=100, timeout=30)
        json_data = request.model_dump()

        assert json_data == {
            "query": "test query",
            "max_rows": 100,
            "timeout": 30,
            "include_summary": True  # Default value
        }

    def test_query_response_serialization(self):
        """Test QueryResponse serializes to JSON correctly."""
        response = QueryResponse(
            rows=[{"id": 1}],
            row_count=1,
            column_names=["id"],
            execution_time_ms=10.5
        )
        json_data = response.model_dump()

        assert json_data["success"] is True
        assert json_data["rows"] == [{"id": 1}]
        assert json_data["row_count"] == 1
        assert json_data["column_names"] == ["id"]
        assert json_data["execution_time_ms"] == 10.5
        assert json_data["truncated"] is False
        assert json_data["sql_query"] is None
