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
    ConnectionPoolError,
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
# Pytest Configuration
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
