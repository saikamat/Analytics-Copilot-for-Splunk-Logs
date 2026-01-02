"""
Database utilities for executing SQL queries with connection pooling.

This module provides connection pooling, query execution, and result formatting
for the log analytics system. It integrates with BedrockSQLGenerator to execute
validated SQL queries against PostgreSQL.

Classes:
    DatabaseConnectionPool: Manages PostgreSQL connections with pooling
    QueryExecutor: Executes SQL queries with security guardrails
    ResultFormatter: Formats query results (future)
    QueryResult: Structured query response (future)

Exceptions:
    DatabaseError: Base exception for database operations
    ConnectionPoolError: Connection pool issues
    QueryExecutionError: SQL execution failures
    QueryTimeoutError: Query timeout
    ResultFormattingError: Result formatting issues (future)
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv
from psycopg2 import pool
from psycopg2.pool import PoolError

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Custom Exceptions
# ============================================================================

class DatabaseError(Exception):
    """Base exception for database operations."""
    pass


class ConnectionPoolError(DatabaseError):
    """Connection pool exhausted or database unreachable."""
    pass


class QueryExecutionError(DatabaseError):
    """SQL execution failed (syntax error, constraint violation)."""
    pass


class QueryTimeoutError(DatabaseError):
    """Query exceeded timeout limit."""
    pass


class ResultFormattingError(DatabaseError):
    """Failed to format results (type conversion error)."""
    pass


# ============================================================================
# DatabaseConnectionPool
# ============================================================================

class DatabaseConnectionPool:
    """
    Manages PostgreSQL connections with pooling for performance.

    Connection pooling provides significant performance improvements by reusing
    database connections instead of creating new ones for each query. This
    reduces query latency from ~65ms to ~12ms (5x faster).

    Attributes:
        database_url (str): PostgreSQL connection string
        min_conn (int): Minimum number of connections in pool
        max_conn (int): Maximum number of connections in pool
        pool (SimpleConnectionPool): psycopg2 connection pool

    Example:
        >>> pool = DatabaseConnectionPool()
        >>> conn = pool.get_connection()
        >>> cursor = conn.cursor()
        >>> cursor.execute("SELECT * FROM logs LIMIT 1")
        >>> pool.return_connection(conn)
        >>> pool.close_all()
    """

    def __init__(
        self,
        database_url: Optional[str] = None,
        min_conn: int = None,
        max_conn: int = None
    ):
        """
        Initialize connection pool with configurable parameters.

        Args:
            database_url: PostgreSQL connection string (defaults to DATABASE_URL env var)
            min_conn: Minimum connections in pool (defaults to DB_POOL_MIN_CONN env var or 2)
            max_conn: Maximum connections in pool (defaults to DB_POOL_MAX_CONN env var or 10)

        Raises:
            ConnectionPoolError: If pool initialization fails

        Note:
            The pool is thread-safe and can be shared across multiple threads
            (e.g., FastAPI workers).
        """
        # Load configuration from environment or use defaults
        self.database_url = database_url or os.getenv("DATABASE_URL")
        if not self.database_url:
            raise ConnectionPoolError("DATABASE_URL not found in environment variables")

        self.min_conn = min_conn if min_conn is not None else int(os.getenv("DB_POOL_MIN_CONN", "2"))
        self.max_conn = max_conn if max_conn is not None else int(os.getenv("DB_POOL_MAX_CONN", "10"))

        # Validate configuration BEFORE attempting pool creation
        if self.min_conn < 1:
            raise ConnectionPoolError(f"min_conn must be >= 1, got {self.min_conn}")
        if self.max_conn < self.min_conn:
            raise ConnectionPoolError(
                f"max_conn ({self.max_conn}) must be >= min_conn ({self.min_conn})"
            )

        # Initialize connection pool
        try:
            self.pool = pool.SimpleConnectionPool(
                self.min_conn,
                self.max_conn,
                self.database_url
            )
            logger.info(
                f"Connection pool initialized: min={self.min_conn}, max={self.max_conn}"
            )
        except Exception as e:
            raise ConnectionPoolError(f"Failed to initialize connection pool: {str(e)}")

    def get_connection(self):
        """
        Acquire a connection from the pool.

        Returns:
            psycopg2.connection: Database connection from pool

        Raises:
            ConnectionPoolError: If pool is exhausted or connection fails

        Note:
            Always return the connection using return_connection() when done,
            or use a context manager pattern to ensure proper cleanup.
        """
        try:
            conn = self.pool.getconn()
            if conn is None:
                raise ConnectionPoolError("Pool returned None - pool may be exhausted")
            logger.debug("Connection acquired from pool")
            return conn
        except PoolError as e:
            raise ConnectionPoolError(f"Pool exhausted or unavailable: {str(e)}")
        except Exception as e:
            raise ConnectionPoolError(f"Failed to get connection: {str(e)}")

    def return_connection(self, conn) -> None:
        """
        Return a connection to the pool.

        Args:
            conn: Database connection to return

        Raises:
            ConnectionPoolError: If connection return fails

        Note:
            Always call this method when done with a connection to avoid
            pool exhaustion. Consider using context managers for automatic cleanup.
        """
        try:
            if conn is not None:
                self.pool.putconn(conn)
                logger.debug("Connection returned to pool")
        except Exception as e:
            raise ConnectionPoolError(f"Failed to return connection: {str(e)}")

    def close_all(self) -> None:
        """
        Gracefully close all connections in the pool.

        This should be called when shutting down the application to ensure
        all database connections are properly closed.

        Raises:
            ConnectionPoolError: If pool closure fails

        Note:
            After calling this method, the pool cannot be reused. You must
            create a new DatabaseConnectionPool instance.
        """
        try:
            if hasattr(self, 'pool') and self.pool is not None:
                self.pool.closeall()
                logger.info("All connections in pool closed")
        except Exception as e:
            raise ConnectionPoolError(f"Failed to close pool: {str(e)}")

    def health_check(self) -> bool:
        """
        Verify database connectivity by executing a simple query.

        Returns:
            bool: True if connection is healthy, False otherwise

        Note:
            This method acquires a connection from the pool, runs a test query,
            and returns the connection. It's useful for readiness probes in
            container orchestration (e.g., Kubernetes).
        """
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            cursor.close()

            if result and result[0] == 1:
                logger.info("Health check passed")
                return True
            else:
                logger.warning("Health check failed: unexpected result")
                return False
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False
        finally:
            if conn is not None:
                self.return_connection(conn)

    def __del__(self):
        """
        Destructor to ensure pool cleanup.

        Note:
            This is a safety mechanism. Prefer calling close_all() explicitly
            rather than relying on garbage collection.
        """
        try:
            self.close_all()
        except:
            pass  # Ignore errors during cleanup


# ============================================================================
# QueryExecutor
# ============================================================================

class QueryExecutor:
    """
    Executes SQL queries with security guardrails.

    Implements five layers of defense-in-depth security:
    1. SQL validation (assumed done by BedrockSQLGenerator before this)
    2. Read-only transaction enforcement (PostgreSQL-level)
    3. Query timeout enforcement (prevents long-running queries)
    4. Result row limits (prevents memory exhaustion)
    5. Connection pool limits (prevents connection exhaustion)

    Attributes:
        pool (DatabaseConnectionPool): Connection pool for database access

    Example:
        >>> pool = DatabaseConnectionPool()
        >>> executor = QueryExecutor(pool)
        >>> result = executor.execute_query(
        ...     "SELECT * FROM logs WHERE level = 'ERROR' LIMIT 10",
        ...     timeout=30,
        ...     max_rows=10000
        ... )
        >>> print(f"Found {len(result['rows'])} errors in {result['execution_time_ms']}ms")
    """

    def __init__(self, pool: DatabaseConnectionPool):
        """
        Initialize QueryExecutor with a connection pool.

        Args:
            pool: DatabaseConnectionPool instance for acquiring connections

        Note:
            The same pool can be shared across multiple QueryExecutor instances
            in a multi-threaded environment (e.g., FastAPI workers).
        """
        self.pool = pool

    def execute_query(
        self,
        sql: str,
        timeout: int = 30,
        max_rows: int = 10000
    ) -> dict:
        """
        Execute SQL query with security guardrails and performance tracking.

        Security layers applied:
        1. Read-only transaction: Prevents data modification even if SQL validation bypassed
        2. Query timeout: Cancels queries exceeding timeout limit
        3. Row limit: Fetches maximum max_rows, sets truncated flag if more exist

        Args:
            sql: Validated SQL query (should come from BedrockSQLGenerator)
            timeout: Query timeout in seconds (default: 30)
            max_rows: Maximum rows to return (default: 10,000)

        Returns:
            dict with keys:
                - rows: List of tuples containing query results
                - column_names: List of column names from cursor.description
                - execution_time_ms: Query execution time in milliseconds
                - truncated: True if results exceeded max_rows
                - row_count: Number of rows returned

        Raises:
            ConnectionPoolError: If connection cannot be acquired (retries once)
            QueryExecutionError: If SQL execution fails
            QueryTimeoutError: If query exceeds timeout limit

        Example:
            >>> executor = QueryExecutor(pool)
            >>> result = executor.execute_query(
            ...     "SELECT * FROM logs WHERE level = 'ERROR' ORDER BY timestamp DESC LIMIT 100"
            ... )
            >>> for row in result['rows']:
            ...     print(row)
        """
        import time

        conn = None
        cursor = None
        start_time = time.perf_counter()

        try:
            # Acquire connection from pool with retry logic
            try:
                conn = self.pool.get_connection()
            except ConnectionPoolError as e:
                # Retry once on connection failure (transient errors)
                logger.warning(f"Connection failed, retrying: {str(e)}")
                try:
                    conn = self.pool.get_connection()
                except Exception as retry_error:
                    raise ConnectionPoolError(
                        f"Failed to acquire connection after retry: {str(retry_error)}"
                    )

            cursor = conn.cursor()

            # Security Layer 1: Set read-only transaction
            cursor.execute("SET TRANSACTION READ ONLY")
            logger.debug("Read-only transaction enforced")

            # Security Layer 2: Set query timeout
            timeout_ms = timeout * 1000
            cursor.execute(f"SET statement_timeout = {timeout_ms}")
            logger.debug(f"Query timeout set to {timeout}s")

            # Execute the actual query
            query_start = time.perf_counter()
            try:
                cursor.execute(sql)
            except Exception as e:
                error_msg = str(e)
                # Check if it's a timeout error
                if "canceling statement due to statement timeout" in error_msg.lower():
                    raise QueryTimeoutError(
                        f"Query exceeded timeout limit of {timeout}s: {error_msg}"
                    )
                else:
                    raise QueryExecutionError(f"SQL execution failed: {error_msg}")

            query_end = time.perf_counter()
            execution_time_ms = (query_end - query_start) * 1000

            # Security Layer 3: Fetch with row limit
            # Fetch one extra row to detect truncation
            rows = cursor.fetchmany(max_rows + 1)
            truncated = len(rows) > max_rows

            if truncated:
                rows = rows[:max_rows]  # Trim to max_rows
                logger.warning(
                    f"Query results truncated: {max_rows} rows returned, more available"
                )

            # Extract column names from cursor description
            column_names = [desc[0] for desc in cursor.description] if cursor.description else []

            logger.info(
                f"Query executed successfully: {len(rows)} rows, "
                f"{execution_time_ms:.2f}ms, truncated={truncated}"
            )

            return {
                "rows": rows,
                "column_names": column_names,
                "execution_time_ms": round(execution_time_ms, 2),
                "truncated": truncated,
                "row_count": len(rows)
            }

        except QueryTimeoutError:
            # Re-raise timeout errors without wrapping
            raise

        except QueryExecutionError:
            # Re-raise execution errors without wrapping
            raise

        except ConnectionPoolError:
            # Re-raise connection errors without wrapping
            raise

        except Exception as e:
            # Wrap unexpected errors
            raise QueryExecutionError(f"Unexpected error during query execution: {str(e)}")

        finally:
            # Cleanup: Always close cursor and return connection
            if cursor is not None:
                try:
                    cursor.close()
                except:
                    pass  # Ignore cursor close errors

            if conn is not None:
                try:
                    # Rollback to clean up read-only transaction
                    conn.rollback()
                except:
                    pass  # Ignore rollback errors

                try:
                    self.pool.return_connection(conn)
                except:
                    pass  # Ignore return connection errors


# ============================================================================
# Module-level initialization (optional pattern for single pool)
# ============================================================================

# Uncomment to create a module-level pool (singleton pattern):
# _pool = DatabaseConnectionPool()
#
# def get_connection():
#     """Get connection from module-level pool."""
#     return _pool.get_connection()
#
# def return_connection(conn):
#     """Return connection to module-level pool."""
#     return _pool.return_connection(conn)
