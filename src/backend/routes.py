"""
FastAPI routes for Log Analytics API.

Provides endpoints for natural language log queries and health checks.
"""

from fastapi import APIRouter, HTTPException, status
import logging
import time

from src.backend.models import QueryRequest, QueryResponse, ErrorResponse, HealthResponse
from src.shared.bedrock_client import BedrockSQLGenerator, BedrockError, ValidationError
from src.shared.db_utils import (
    QueryExecutionError,
    QueryTimeoutError,
    ConnectionPoolError
)

# Configure logging
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

# Module-level BedrockSQLGenerator (initialized once)
sql_generator = BedrockSQLGenerator()


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid query or SQL validation failed"},
        408: {"model": ErrorResponse, "description": "Query timeout"},
        502: {"model": ErrorResponse, "description": "Bedrock API error"},
        503: {"model": ErrorResponse, "description": "Database unavailable"},
    },
    summary="Execute natural language query",
    description="""
    Execute a natural language query against the log database.

    Flow:
    1. Translate natural language to SQL using AWS Bedrock
    2. Execute SQL with security guardrails (read-only, timeout, row limits)
    3. Generate LLM summary of results (if include_summary=True, adds ~1.5s)
    4. Return formatted results with execution time and optional summary

    Examples:
    - "show me errors from yesterday"
    - "count nginx 500 errors in the last hour"
    - "find login failures from user admin"
    """
)
async def execute_query(request: QueryRequest):
    """
    Execute a natural language query against the log database.

    Args:
        request: QueryRequest with natural language query and optional limits

    Returns:
        QueryResponse with results, row count, and execution metadata

    Raises:
        HTTPException: Various HTTP errors mapped from underlying exceptions
    """
    start_time = time.time()
    logger.info(f"Query request received: '{request.query}' (max_rows={request.max_rows}, timeout={request.timeout}, include_summary={request.include_summary})")

    try:
        # Step 1: Generate SQL from natural language
        logger.debug("Generating SQL from natural language query...")
        sql = sql_generator.generate_sql(request.query)
        logger.info(f"Generated SQL: {sql}")

        # Step 2: Execute query with timeout and row limits
        logger.debug(f"Executing SQL with timeout={request.timeout}s, max_rows={request.max_rows}")

        # Import query_executor from app module (initialized in lifespan)
        from src.backend.app import query_executor

        if query_executor is None:
            logger.error("QueryExecutor not initialized - application startup may have failed")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "success": False,
                    "error": "Database connection pool not initialized",
                    "error_type": "ServiceUnavailable"
                }
            )

        result = query_executor.execute_query(
            sql=sql,
            timeout=request.timeout,
            max_rows=request.max_rows
        )

        # Step 3: Generate summary (if requested)
        summary = None
        summary_success = None
        summary_execution_time_ms = None

        if request.include_summary:
            logger.debug("Generating summary of query results...")

            # Import result_summarizer from app module (initialized in lifespan)
            from src.backend.app import result_summarizer

            if result_summarizer is None:
                logger.warning("ResultSummarizer not initialized - skipping summary generation")
            else:
                try:
                    summary_result = result_summarizer.summarize(
                        original_query=request.query,
                        sql_query=sql,
                        result=result
                    )
                    summary = summary_result.summary
                    summary_success = summary_result.success
                    summary_execution_time_ms = summary_result.execution_time_ms

                    logger.info(
                        f"Summary generated: {summary_success} in {summary_execution_time_ms:.1f}ms"
                    )
                except Exception as e:
                    logger.warning(f"Summary generation failed: {e}")
                    summary = f"Summary generation failed: {str(e)}"
                    summary_success = False
                    summary_execution_time_ms = 0.0

        # Step 4: Return formatted response
        total_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        logger.info(
            f"Query completed successfully: {result.row_count} rows in {result.execution_time_ms:.1f}ms "
            f"(total: {total_time:.1f}ms, truncated: {result.truncated})"
        )

        return QueryResponse(
            success=result.success,
            rows=result.rows,
            row_count=result.row_count,
            column_names=result.column_names,
            execution_time_ms=result.execution_time_ms,
            truncated=result.truncated,
            sql_query=sql,  # Include generated SQL for transparency
            summary=summary,
            summary_success=summary_success,
            summary_execution_time_ms=summary_execution_time_ms
        )

    except ValidationError as e:
        # SQL validation failed (dangerous operation blocked)
        logger.warning(f"SQL validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "success": False,
                "error": str(e),
                "error_type": "ValidationError"
            }
        )

    except QueryExecutionError as e:
        # SQL execution failed (syntax error, invalid column, etc.)
        logger.error(f"Query execution failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "success": False,
                "error": str(e),
                "error_type": "QueryExecutionError"
            }
        )

    except QueryTimeoutError as e:
        # Query exceeded timeout limit
        logger.warning(f"Query timeout: {e}")
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail={
                "success": False,
                "error": str(e),
                "error_type": "QueryTimeoutError"
            }
        )

    except ConnectionPoolError as e:
        # Database connection unavailable
        logger.error(f"Database connection failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "success": False,
                "error": "Database unavailable",
                "error_type": "ConnectionPoolError"
            }
        )

    except BedrockError as e:
        # Bedrock API error (throttling, service error, etc.)
        logger.error(f"Bedrock API error: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "success": False,
                "error": "LLM service error",
                "error_type": "BedrockError"
            }
        )

    except Exception as e:
        # Unexpected error
        logger.exception(f"Unexpected error during query execution: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "error": "Internal server error",
                "error_type": "UnknownError"
            }
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="""
    Check API health and dependency availability.

    Returns:
    - Overall health status ('healthy' or 'degraded')
    - Database connection pool status
    - AWS Bedrock availability (assumed available unless test call fails)
    """
)
async def health_check():
    """
    Check API health and dependency availability.

    Returns:
        HealthResponse with status indicators for all dependencies
    """
    logger.debug("Health check requested")

    # Check database connection pool
    db_connected = False
    try:
        from src.backend.app import db_pool

        if db_pool is None:
            logger.warning("Database connection pool not initialized")
            db_connected = False
        else:
            db_connected = db_pool.health_check()
            logger.debug(f"Database health check: {'passed' if db_connected else 'failed'}")
    except Exception as e:
        logger.warning(f"Database health check failed: {e}")
        db_connected = False

    # Check Bedrock availability
    # Note: We assume Bedrock is available unless we make a test API call
    # For now, we'll assume it's available to avoid unnecessary API calls
    bedrock_available = True

    # Determine overall status
    overall_status = "healthy" if (db_connected and bedrock_available) else "degraded"

    logger.info(f"Health check: {overall_status} (db={db_connected}, bedrock={bedrock_available})")

    return HealthResponse(
        status=overall_status,
        database_connected=db_connected,
        bedrock_available=bedrock_available
    )
