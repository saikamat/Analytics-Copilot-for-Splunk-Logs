"""
Pydantic models for API request/response validation.

Provides type-safe request validation and response serialization with
automatic OpenAPI documentation generation.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class QueryRequest(BaseModel):
    """Request model for natural language query."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Natural language query (e.g., 'show nginx errors from yesterday')",
        examples=["show me all error logs from yesterday"]
    )
    max_rows: int = Field(
        default=10000,
        ge=1,
        le=50000,
        description="Maximum rows to return (1-50,000)",
        examples=[100]
    )
    timeout: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Query timeout in seconds (1-300)",
        examples=[30]
    )
    include_summary: bool = Field(
        default=True,
        description="Whether to generate LLM summary of results (adds ~1.5s latency)",
        examples=[True]
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "show me all nginx errors from yesterday",
                    "max_rows": 100,
                    "timeout": 30,
                    "include_summary": True
                },
                {
                    "query": "count errors by service in the last hour",
                    "max_rows": 50,
                    "timeout": 10,
                    "include_summary": False
                }
            ]
        }
    }


class QueryResponse(BaseModel):
    """Response model for successful query."""

    success: bool = Field(
        default=True,
        description="Whether the query executed successfully"
    )
    rows: List[Dict[str, Any]] = Field(
        description="Query results as list of dictionaries"
    )
    row_count: int = Field(
        description="Number of rows returned"
    )
    column_names: List[str] = Field(
        description="Column names from SELECT statement"
    )
    execution_time_ms: float = Field(
        description="Query execution time in milliseconds"
    )
    truncated: bool = Field(
        default=False,
        description="True if results exceeded max_rows limit"
    )
    sql_query: Optional[str] = Field(
        default=None,
        description="Generated SQL query (for transparency)"
    )
    summary: Optional[str] = Field(
        default=None,
        description="Natural language summary of results (if include_summary=True)"
    )
    summary_success: Optional[bool] = Field(
        default=None,
        description="Whether summary generation succeeded (if include_summary=True)"
    )
    summary_execution_time_ms: Optional[float] = Field(
        default=None,
        description="Time taken for summary generation in milliseconds (if include_summary=True)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "success": True,
                    "rows": [
                        {
                            "id": 42,
                            "timestamp": "2025-01-01T14:30:00+00:00",
                            "level": "ERROR",
                            "source": "nginx",
                            "message": "Connection timeout",
                            "metadata": {"status_code": "500"}
                        }
                    ],
                    "row_count": 15,
                    "column_names": ["id", "timestamp", "level", "source", "message", "metadata"],
                    "execution_time_ms": 1567.3,
                    "truncated": False,
                    "sql_query": "SELECT * FROM logs WHERE level = 'ERROR' AND timestamp >= NOW() - INTERVAL '1 day'",
                    "summary": "Found 15 errors in the last 24 hours. Most errors are from nginx (502 Bad Gateway) and PostgreSQL (connection pool exhaustion). The issues started around 2pm and continued until 5pm, suggesting a service disruption during that window.",
                    "summary_success": True,
                    "summary_execution_time_ms": 1523.4
                }
            ]
        }
    }


class ErrorResponse(BaseModel):
    """Response model for errors."""

    success: bool = Field(
        default=False,
        description="Always False for error responses"
    )
    error: str = Field(
        description="Error message describing what went wrong"
    )
    error_type: Optional[str] = Field(
        default=None,
        description="Type of error (e.g., ValidationError, QueryTimeoutError)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "success": False,
                    "error": "SQL validation failed: DROP statements are not allowed",
                    "error_type": "ValidationError"
                },
                {
                    "success": False,
                    "error": "Query timed out after 30 seconds",
                    "error_type": "QueryTimeoutError"
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(
        description="Overall health status: 'healthy' or 'degraded'"
    )
    database_connected: bool = Field(
        description="Whether database connection pool is operational"
    )
    bedrock_available: bool = Field(
        description="Whether AWS Bedrock service is available"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "healthy",
                    "database_connected": True,
                    "bedrock_available": True
                },
                {
                    "status": "degraded",
                    "database_connected": False,
                    "bedrock_available": True
                }
            ]
        }
    }
