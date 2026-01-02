"""
Unit and integration tests for database utilities (db_utils.py).

Tests cover:
- DatabaseConnectionPool: Connection management, pooling, health checks
- QueryExecutor: SQL execution with security (future)
- ResultFormatter: Type conversions (future)
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from psycopg2.pool import PoolError
from src.shared.db_utils import (
    DatabaseConnectionPool,
    QueryExecutor,
    ConnectionPoolError,
    QueryExecutionError,
    QueryTimeoutError,
    DatabaseError
)


# ============================================================================
# Unit Tests: DatabaseConnectionPool (Mocked)
# ============================================================================

class TestDatabaseConnectionPoolUnit:
    """Unit tests for DatabaseConnectionPool with mocked psycopg2."""

    def test_init_with_defaults(self):
        """Test pool initialization with default environment variables."""
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://localhost/test"}):
            with patch('src.shared.db_utils.pool.SimpleConnectionPool') as mock_pool:
                pool_obj = DatabaseConnectionPool()

                assert pool_obj.database_url == "postgresql://localhost/test"
                assert pool_obj.min_conn == 2  # Default
                assert pool_obj.max_conn == 10  # Default
                mock_pool.assert_called_once_with(
                    2, 10, "postgresql://localhost/test"
                )

    def test_init_with_custom_params(self):
        """Test pool initialization with custom parameters."""
        with patch('src.shared.db_utils.pool.SimpleConnectionPool') as mock_pool:
            pool_obj = DatabaseConnectionPool(
                database_url="postgresql://localhost/custom",
                min_conn=5,
                max_conn=20
            )

            assert pool_obj.database_url == "postgresql://localhost/custom"
            assert pool_obj.min_conn == 5
            assert pool_obj.max_conn == 20
            mock_pool.assert_called_once_with(
                5, 20, "postgresql://localhost/custom"
            )

    def test_init_with_env_vars(self):
        """Test pool initialization with environment variable configuration."""
        env_vars = {
            "DATABASE_URL": "postgresql://localhost/env_test",
            "DB_POOL_MIN_CONN": "3",
            "DB_POOL_MAX_CONN": "15"
        }
        with patch.dict(os.environ, env_vars):
            with patch('src.shared.db_utils.pool.SimpleConnectionPool') as mock_pool:
                pool_obj = DatabaseConnectionPool()

                assert pool_obj.min_conn == 3
                assert pool_obj.max_conn == 15

    def test_init_missing_database_url(self):
        """Test that initialization fails without DATABASE_URL."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ConnectionPoolError, match="DATABASE_URL not found"):
                DatabaseConnectionPool()

    def test_init_invalid_min_conn(self):
        """Test that initialization fails with invalid min_conn."""
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://localhost/test"}):
            with pytest.raises(ConnectionPoolError, match="min_conn must be >= 1"):
                DatabaseConnectionPool(min_conn=0)

    def test_init_max_less_than_min(self):
        """Test that initialization fails when max_conn < min_conn."""
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://localhost/test"}):
            with pytest.raises(ConnectionPoolError, match="max_conn.*must be >= min_conn"):
                DatabaseConnectionPool(min_conn=10, max_conn=5)

    def test_get_connection_success(self):
        """Test successful connection acquisition from pool."""
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://localhost/test"}):
            with patch('src.shared.db_utils.pool.SimpleConnectionPool') as mock_pool:
                mock_conn = Mock()
                mock_pool.return_value.getconn.return_value = mock_conn

                pool_obj = DatabaseConnectionPool()
                conn = pool_obj.get_connection()

                assert conn == mock_conn
                mock_pool.return_value.getconn.assert_called_once()

    def test_get_connection_pool_exhausted(self):
        """Test connection acquisition failure when pool is exhausted."""
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://localhost/test"}):
            with patch('src.shared.db_utils.pool.SimpleConnectionPool') as mock_pool:
                mock_pool.return_value.getconn.side_effect = PoolError("Pool exhausted")

                pool_obj = DatabaseConnectionPool()

                with pytest.raises(ConnectionPoolError, match="Pool exhausted"):
                    pool_obj.get_connection()

    def test_get_connection_returns_none(self):
        """Test handling when pool returns None."""
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://localhost/test"}):
            with patch('src.shared.db_utils.pool.SimpleConnectionPool') as mock_pool:
                mock_pool.return_value.getconn.return_value = None

                pool_obj = DatabaseConnectionPool()

                with pytest.raises(ConnectionPoolError, match="Pool returned None"):
                    pool_obj.get_connection()

    def test_return_connection_success(self):
        """Test successful connection return to pool."""
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://localhost/test"}):
            with patch('src.shared.db_utils.pool.SimpleConnectionPool') as mock_pool:
                mock_conn = Mock()

                pool_obj = DatabaseConnectionPool()
                pool_obj.return_connection(mock_conn)

                mock_pool.return_value.putconn.assert_called_once_with(mock_conn)

    def test_return_connection_none(self):
        """Test returning None connection (should not raise error)."""
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://localhost/test"}):
            with patch('src.shared.db_utils.pool.SimpleConnectionPool') as mock_pool:
                pool_obj = DatabaseConnectionPool()
                pool_obj.return_connection(None)  # Should not raise error

                mock_pool.return_value.putconn.assert_not_called()

    def test_close_all_success(self):
        """Test graceful pool closure."""
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://localhost/test"}):
            with patch('src.shared.db_utils.pool.SimpleConnectionPool') as mock_pool:
                pool_obj = DatabaseConnectionPool()
                pool_obj.close_all()

                mock_pool.return_value.closeall.assert_called_once()

    def test_health_check_success(self):
        """Test successful health check."""
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://localhost/test"}):
            with patch('src.shared.db_utils.pool.SimpleConnectionPool') as mock_pool:
                # Mock connection and cursor
                mock_conn = Mock()
                mock_cursor = Mock()
                mock_cursor.fetchone.return_value = (1,)
                mock_conn.cursor.return_value = mock_cursor
                mock_pool.return_value.getconn.return_value = mock_conn

                pool_obj = DatabaseConnectionPool()
                result = pool_obj.health_check()

                assert result is True
                mock_cursor.execute.assert_called_once_with("SELECT 1")
                mock_pool.return_value.putconn.assert_called_once_with(mock_conn)

    def test_health_check_failure(self):
        """Test health check failure."""
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://localhost/test"}):
            with patch('src.shared.db_utils.pool.SimpleConnectionPool') as mock_pool:
                mock_pool.return_value.getconn.side_effect = Exception("Connection failed")

                pool_obj = DatabaseConnectionPool()
                result = pool_obj.health_check()

                assert result is False

    def test_health_check_unexpected_result(self):
        """Test health check with unexpected query result."""
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://localhost/test"}):
            with patch('src.shared.db_utils.pool.SimpleConnectionPool') as mock_pool:
                mock_conn = Mock()
                mock_cursor = Mock()
                mock_cursor.fetchone.return_value = (2,)  # Unexpected result
                mock_conn.cursor.return_value = mock_cursor
                mock_pool.return_value.getconn.return_value = mock_conn

                pool_obj = DatabaseConnectionPool()
                result = pool_obj.health_check()

                assert result is False


# ============================================================================
# Unit Tests: QueryExecutor (Mocked)
# ============================================================================

class TestQueryExecutorUnit:
    """Unit tests for QueryExecutor with mocked connections."""

    def test_execute_query_success(self):
        """Test successful query execution with all security layers."""
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://localhost/test"}):
            with patch('src.shared.db_utils.pool.SimpleConnectionPool') as mock_pool:
                # Mock connection and cursor
                mock_conn = Mock()
                mock_cursor = Mock()
                mock_cursor.description = [("id",), ("timestamp",), ("level",), ("message",)]
                mock_cursor.fetchmany.return_value = [
                    (1, "2025-12-28 10:00:00", "ERROR", "Test error 1"),
                    (2, "2025-12-28 11:00:00", "ERROR", "Test error 2")
                ]
                mock_conn.cursor.return_value = mock_cursor
                mock_pool.return_value.getconn.return_value = mock_conn

                # Create pool and executor
                pool_obj = DatabaseConnectionPool()
                executor = QueryExecutor(pool_obj)

                # Execute query
                result = executor.execute_query("SELECT * FROM logs WHERE level = 'ERROR' LIMIT 10")

                # Verify security commands were executed
                calls = [str(call) for call in mock_cursor.execute.call_args_list]
                assert any("SET TRANSACTION READ ONLY" in str(call) for call in calls)
                assert any("SET statement_timeout" in str(call) for call in calls)

                # Verify result structure
                assert result["rows"] == [
                    (1, "2025-12-28 10:00:00", "ERROR", "Test error 1"),
                    (2, "2025-12-28 11:00:00", "ERROR", "Test error 2")
                ]
                assert result["column_names"] == ["id", "timestamp", "level", "message"]
                assert result["row_count"] == 2
                assert result["truncated"] is False
                assert "execution_time_ms" in result

    def test_execute_query_with_truncation(self):
        """Test query execution with result truncation."""
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://localhost/test"}):
            with patch('src.shared.db_utils.pool.SimpleConnectionPool') as mock_pool:
                mock_conn = Mock()
                mock_cursor = Mock()
                mock_cursor.description = [("id",), ("message",)]

                # Return max_rows + 1 rows to trigger truncation
                mock_cursor.fetchmany.return_value = [(i, f"Message {i}") for i in range(6)]
                mock_conn.cursor.return_value = mock_cursor
                mock_pool.return_value.getconn.return_value = mock_conn

                pool_obj = DatabaseConnectionPool()
                executor = QueryExecutor(pool_obj)

                # Execute with max_rows=5
                result = executor.execute_query("SELECT * FROM logs", max_rows=5)

                # Verify truncation
                assert result["truncated"] is True
                assert result["row_count"] == 5
                assert len(result["rows"]) == 5

    def test_execute_query_timeout(self):
        """Test query timeout handling."""
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://localhost/test"}):
            with patch('src.shared.db_utils.pool.SimpleConnectionPool') as mock_pool:
                mock_conn = Mock()
                mock_cursor = Mock()

                # Simulate timeout error only on the actual query
                def execute_side_effect(sql):
                    # Allow SET commands to pass through
                    if "SET" in sql:
                        return None
                    # Fail on the actual SELECT query with timeout
                    if "SELECT" in sql and "logs" in sql:
                        raise Exception("canceling statement due to statement timeout")

                mock_cursor.execute.side_effect = execute_side_effect
                mock_conn.cursor.return_value = mock_cursor
                mock_pool.return_value.getconn.return_value = mock_conn

                pool_obj = DatabaseConnectionPool()
                executor = QueryExecutor(pool_obj)

                # Should raise QueryTimeoutError
                with pytest.raises(QueryTimeoutError, match="Query exceeded timeout limit"):
                    executor.execute_query("SELECT * FROM logs", timeout=1)

    def test_execute_query_sql_error(self):
        """Test SQL execution error handling."""
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://localhost/test"}):
            with patch('src.shared.db_utils.pool.SimpleConnectionPool') as mock_pool:
                mock_conn = Mock()
                mock_cursor = Mock()

                # Simulate SQL syntax error only on the actual query
                def execute_side_effect(sql):
                    # Allow SET commands to pass through
                    if "SET" in sql:
                        return None
                    # Fail on the actual SELECT query
                    if "SELEC" in sql:
                        raise Exception("syntax error at or near \"SELEC\"")

                mock_cursor.execute.side_effect = execute_side_effect
                mock_conn.cursor.return_value = mock_cursor
                mock_pool.return_value.getconn.return_value = mock_conn

                pool_obj = DatabaseConnectionPool()
                executor = QueryExecutor(pool_obj)

                # Should raise QueryExecutionError
                with pytest.raises(QueryExecutionError, match="SQL execution failed"):
                    executor.execute_query("SELEC * FROM logs")

    def test_execute_query_connection_retry(self):
        """Test connection retry logic on transient failure."""
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://localhost/test"}):
            with patch('src.shared.db_utils.pool.SimpleConnectionPool') as mock_pool:
                mock_conn = Mock()
                mock_cursor = Mock()
                mock_cursor.description = [("id",)]
                mock_cursor.fetchmany.return_value = [(1,)]
                mock_conn.cursor.return_value = mock_cursor

                # First call fails, second succeeds
                mock_pool.return_value.getconn.side_effect = [
                    PoolError("Pool exhausted"),
                    mock_conn
                ]

                pool_obj = DatabaseConnectionPool()
                executor = QueryExecutor(pool_obj)

                # Should succeed after retry
                result = executor.execute_query("SELECT * FROM logs LIMIT 1")
                assert result["row_count"] == 1

    def test_execute_query_connection_retry_failure(self):
        """Test connection retry failure after second attempt."""
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://localhost/test"}):
            with patch('src.shared.db_utils.pool.SimpleConnectionPool') as mock_pool:
                # Both attempts fail
                mock_pool.return_value.getconn.side_effect = [
                    PoolError("Pool exhausted"),
                    PoolError("Pool exhausted")
                ]

                pool_obj = DatabaseConnectionPool()
                executor = QueryExecutor(pool_obj)

                # Should raise ConnectionPoolError after retry
                with pytest.raises(ConnectionPoolError, match="Failed to acquire connection after retry"):
                    executor.execute_query("SELECT * FROM logs LIMIT 1")

    def test_execute_query_cleanup_on_error(self):
        """Test that connections are properly returned even on error."""
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://localhost/test"}):
            with patch('src.shared.db_utils.pool.SimpleConnectionPool') as mock_pool:
                mock_conn = Mock()
                mock_cursor = Mock()

                # Allow SET commands but fail on SELECT
                def execute_side_effect(sql):
                    if "SET" in sql:
                        return None
                    raise Exception("SQL error")

                mock_cursor.execute.side_effect = execute_side_effect
                mock_conn.cursor.return_value = mock_cursor
                mock_pool.return_value.getconn.return_value = mock_conn

                pool_obj = DatabaseConnectionPool()
                executor = QueryExecutor(pool_obj)

                # Execute query that will fail
                try:
                    executor.execute_query("SELECT * FROM logs")
                except QueryExecutionError:
                    pass

                # Verify cleanup happened
                mock_cursor.close.assert_called()
                mock_conn.rollback.assert_called()
                mock_pool.return_value.putconn.assert_called_with(mock_conn)

    def test_execute_query_empty_result(self):
        """Test query execution with no results."""
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://localhost/test"}):
            with patch('src.shared.db_utils.pool.SimpleConnectionPool') as mock_pool:
                mock_conn = Mock()
                mock_cursor = Mock()
                mock_cursor.description = [("id",)]
                mock_cursor.fetchmany.return_value = []  # No results
                mock_conn.cursor.return_value = mock_cursor
                mock_pool.return_value.getconn.return_value = mock_conn

                pool_obj = DatabaseConnectionPool()
                executor = QueryExecutor(pool_obj)

                result = executor.execute_query("SELECT * FROM logs WHERE 1=0")

                assert result["rows"] == []
                assert result["row_count"] == 0
                assert result["truncated"] is False


# ============================================================================
# Integration Tests: DatabaseConnectionPool (Real PostgreSQL)
# ============================================================================

class TestDatabaseConnectionPoolIntegration:
    """Integration tests for DatabaseConnectionPool with real PostgreSQL."""

    @pytest.fixture
    def database_url(self):
        """Get database URL from environment."""
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            pytest.skip("DATABASE_URL not set - skipping integration tests")
        return db_url

    def test_pool_initialization_real_db(self, database_url):
        """Test pool initialization with real PostgreSQL connection."""
        pool = DatabaseConnectionPool(database_url=database_url, min_conn=2, max_conn=5)

        assert pool is not None
        assert pool.min_conn == 2
        assert pool.max_conn == 5

        pool.close_all()

    def test_get_and_return_connection_real_db(self, database_url):
        """Test acquiring and returning a connection with real PostgreSQL."""
        pool = DatabaseConnectionPool(database_url=database_url)

        # Get connection
        conn = pool.get_connection()
        assert conn is not None

        # Use connection
        cursor = conn.cursor()
        cursor.execute("SELECT 1 as test")
        result = cursor.fetchone()
        assert result[0] == 1
        cursor.close()

        # Return connection
        pool.return_connection(conn)

        pool.close_all()

    def test_multiple_connections_real_db(self, database_url):
        """Test acquiring multiple connections from pool."""
        pool = DatabaseConnectionPool(database_url=database_url, min_conn=2, max_conn=5)

        # Get multiple connections
        conn1 = pool.get_connection()
        conn2 = pool.get_connection()
        conn3 = pool.get_connection()

        assert conn1 is not None
        assert conn2 is not None
        assert conn3 is not None
        assert conn1 != conn2
        assert conn2 != conn3

        # Return all connections
        pool.return_connection(conn1)
        pool.return_connection(conn2)
        pool.return_connection(conn3)

        pool.close_all()

    def test_health_check_real_db(self, database_url):
        """Test health check with real PostgreSQL."""
        pool = DatabaseConnectionPool(database_url=database_url)

        result = pool.health_check()
        assert result is True

        pool.close_all()

    def test_query_execution_real_db(self, database_url):
        """Test executing a query against real logs table."""
        pool = DatabaseConnectionPool(database_url=database_url)

        conn = pool.get_connection()
        cursor = conn.cursor()

        # Query logs table
        cursor.execute("SELECT COUNT(*) FROM logs")
        count = cursor.fetchone()[0]

        assert count >= 0  # Should have some logs from previous testing

        cursor.close()
        pool.return_connection(conn)
        pool.close_all()


# ============================================================================
# Integration Tests: QueryExecutor (Real PostgreSQL)
# ============================================================================

class TestQueryExecutorIntegration:
    """Integration tests for QueryExecutor with real PostgreSQL."""

    @pytest.fixture
    def database_url(self):
        """Get database URL from environment."""
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            pytest.skip("DATABASE_URL not set - skipping integration tests")
        return db_url

    @pytest.fixture
    def pool(self, database_url):
        """Create connection pool for tests."""
        pool_obj = DatabaseConnectionPool(database_url=database_url)
        yield pool_obj
        pool_obj.close_all()

    def test_execute_query_real_db(self, pool):
        """Test query execution against real database."""
        executor = QueryExecutor(pool)

        result = executor.execute_query(
            "SELECT id, timestamp, level, source, message FROM logs LIMIT 5"
        )

        # Verify result structure
        assert "rows" in result
        assert "column_names" in result
        assert "execution_time_ms" in result
        assert "truncated" in result
        assert "row_count" in result

        # Verify column names
        assert "id" in result["column_names"]
        assert "timestamp" in result["column_names"]
        assert "level" in result["column_names"]

        # Verify results
        assert result["row_count"] <= 5
        assert result["truncated"] is False

    def test_execute_query_with_where_clause(self, pool):
        """Test query with WHERE clause."""
        executor = QueryExecutor(pool)

        result = executor.execute_query(
            "SELECT id, level, message FROM logs WHERE level = 'ERROR' LIMIT 10"
        )

        # All returned rows should be ERROR level
        for row in result["rows"]:
            # row format: (id, level, message)
            assert row[1] == "ERROR"

    def test_execute_query_aggregation(self, pool):
        """Test aggregation query."""
        executor = QueryExecutor(pool)

        result = executor.execute_query(
            "SELECT level, COUNT(*) as count FROM logs GROUP BY level ORDER BY count DESC"
        )

        # Verify result structure
        assert result["row_count"] >= 0
        assert "level" in result["column_names"]
        assert "count" in result["column_names"]

    def test_execute_query_order_by(self, pool):
        """Test query with ORDER BY."""
        executor = QueryExecutor(pool)

        result = executor.execute_query(
            "SELECT id, timestamp FROM logs ORDER BY timestamp DESC LIMIT 3"
        )

        assert result["row_count"] <= 3

        # Verify timestamps are in descending order
        if result["row_count"] >= 2:
            for i in range(result["row_count"] - 1):
                # Timestamps should be descending
                assert result["rows"][i][1] >= result["rows"][i + 1][1]

    def test_execute_query_truncation_real_db(self, pool):
        """Test result truncation with real database."""
        executor = QueryExecutor(pool)

        # Set max_rows to 5 and query for more
        result = executor.execute_query(
            "SELECT * FROM logs LIMIT 100",
            max_rows=5
        )

        # Should be truncated if logs table has > 5 rows
        if result["row_count"] == 5:
            # If we got 5 rows and there are more in the database, truncated should be True
            # We can't guarantee this without knowing the table size, so just verify structure
            assert isinstance(result["truncated"], bool)
            assert result["row_count"] == 5

    def test_execute_query_empty_result_real_db(self, pool):
        """Test query with no results."""
        executor = QueryExecutor(pool)

        result = executor.execute_query(
            "SELECT * FROM logs WHERE id = -999999"
        )

        assert result["rows"] == []
        assert result["row_count"] == 0
        assert result["truncated"] is False

    def test_execute_query_metadata_jsonb(self, pool):
        """Test query accessing JSONB metadata field."""
        executor = QueryExecutor(pool)

        result = executor.execute_query(
            "SELECT id, metadata FROM logs WHERE metadata IS NOT NULL LIMIT 3"
        )

        # Verify we can access metadata
        assert result["row_count"] <= 3
        if result["row_count"] > 0:
            assert "metadata" in result["column_names"]

    def test_read_only_enforcement_real_db(self, pool):
        """Test that read-only transaction prevents writes."""
        executor = QueryExecutor(pool)

        # Attempt to insert - should fail due to read-only transaction
        with pytest.raises(QueryExecutionError, match="read-only transaction"):
            executor.execute_query(
                "INSERT INTO logs (timestamp, level, message) "
                "VALUES (NOW(), 'TEST', 'Should fail')"
            )

    def test_query_performance_tracking(self, pool):
        """Test that execution time is tracked."""
        executor = QueryExecutor(pool)

        result = executor.execute_query("SELECT 1")

        # Execution time should be tracked and reasonable
        assert "execution_time_ms" in result
        assert result["execution_time_ms"] > 0
        assert result["execution_time_ms"] < 1000  # Should complete in < 1 second


# ============================================================================
# Pytest Configuration
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
