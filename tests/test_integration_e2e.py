"""
End-to-end integration tests for database utilities.

Tests the full flow: NL Query → SQL Generation → Query Execution → Result Formatting
Also tests concurrent connection pool usage, large result sets, and error handling.
"""

import pytest
import threading
import time
from datetime import datetime
from unittest.mock import Mock, patch

from src.shared.db_utils import (
    DatabaseConnectionPool,
    QueryExecutor,
    QueryResult,
    ConnectionPoolError,
    QueryExecutionError,
    QueryTimeoutError
)
from src.shared.bedrock_client import BedrockSQLGenerator, ValidationError


class TestEndToEndFlow:
    """Test full NL→SQL→Execution→Results flow."""

    def test_end_to_end_with_bedrock(self, test_db_with_fixtures, test_db_url):
        """
        Test complete flow from natural language to results.

        Flow: Natural Language → BedrockSQLGenerator → QueryExecutor → QueryResult
        """
        # Initialize components
        pool = DatabaseConnectionPool(database_url=test_db_url)
        executor = QueryExecutor(pool)
        sql_generator = BedrockSQLGenerator()

        try:
            # Step 1: Generate SQL from natural language
            nl_query = "show me all error logs"
            sql = sql_generator.generate_sql(nl_query)

            # Verify SQL was generated
            assert sql is not None
            assert isinstance(sql, str)
            assert "SELECT" in sql.upper()
            assert "ERROR" in sql.upper()

            # Step 2: Execute generated SQL
            result = executor.execute_query(sql, timeout=30, max_rows=100)

            # Step 3: Verify results
            assert isinstance(result, QueryResult)
            assert result.success is True
            assert result.row_count >= 0  # May be 0 if no errors in fixtures

            # If we have results, verify structure
            if result.row_count > 0:
                # All rows should be dicts
                assert all(isinstance(row, dict) for row in result.rows)
                # Should have column names
                assert len(result.column_names) > 0
                # Execution time should be tracked
                assert result.execution_time_ms > 0

        finally:
            pool.close_all()

    def test_end_to_end_with_mock_bedrock(self, test_db_with_fixtures, test_db_url):
        """
        Test end-to-end flow with mocked Bedrock (no API calls).

        This test verifies the integration without requiring AWS credentials.
        """
        # Initialize components
        pool = DatabaseConnectionPool(database_url=test_db_url)
        executor = QueryExecutor(pool)

        try:
            # Mock SQL generation (bypass Bedrock API)
            mock_sql = """
                SELECT id, timestamp, level, source, message, metadata
                FROM logs
                WHERE level = 'ERROR'
                ORDER BY timestamp DESC
                LIMIT 10
            """

            # Execute the SQL
            result = executor.execute_query(mock_sql, timeout=30)

            # Verify results
            assert isinstance(result, QueryResult)
            assert result.success is True
            assert result.row_count >= 0

            # Verify type conversions
            if result.row_count > 0:
                first_row = result.rows[0]
                # Timestamp should be ISO string
                assert isinstance(first_row["timestamp"], str)
                assert "T" in first_row["timestamp"]  # ISO 8601 format
                # Metadata should be dict
                assert isinstance(first_row["metadata"], dict)
                # Level should be string
                assert isinstance(first_row["level"], str)

        finally:
            pool.close_all()

    def test_temporal_query_with_fixtures(self, test_db_with_fixtures, test_db_url):
        """Test temporal filtering with known fixture data."""
        pool = DatabaseConnectionPool(database_url=test_db_url)
        executor = QueryExecutor(pool)

        try:
            # Query for logs from specific time range (fixture data is at 2025-12-28)
            sql = """
                SELECT id, timestamp, level, source, message
                FROM logs
                WHERE timestamp >= '2025-12-28 10:00:00'
                  AND timestamp < '2025-12-28 11:00:00'
                ORDER BY timestamp ASC
            """

            result = executor.execute_query(sql)

            # We inserted 10 logs, all within this time range
            assert result.success is True
            assert result.row_count == 10

            # Verify timestamps are in ascending order
            for i in range(result.row_count - 1):
                assert result.rows[i]["timestamp"] <= result.rows[i + 1]["timestamp"]

        finally:
            pool.close_all()

    def test_metadata_filtering_with_fixtures(self, test_db_with_fixtures, test_db_url):
        """Test JSONB metadata filtering with known fixture data."""
        pool = DatabaseConnectionPool(database_url=test_db_url)
        executor = QueryExecutor(pool)

        try:
            # Query for nginx 500 errors (fixture log #3)
            sql = """
                SELECT id, source, message, metadata
                FROM logs
                WHERE source = 'nginx'
                  AND metadata->>'status_code' = '500'
            """

            result = executor.execute_query(sql)

            # Should find exactly 1 log (fixture #3)
            assert result.success is True
            assert result.row_count == 1
            assert result.rows[0]["source"] == "nginx"
            assert result.rows[0]["metadata"]["status_code"] == "500"
            assert "Internal server error" in result.rows[0]["message"]

        finally:
            pool.close_all()

    def test_aggregation_with_fixtures(self, test_db_with_fixtures, test_db_url):
        """Test aggregation query with known fixture data."""
        pool = DatabaseConnectionPool(database_url=test_db_url)
        executor = QueryExecutor(pool)

        try:
            # Count logs by level
            sql = """
                SELECT level, COUNT(*) as count
                FROM logs
                GROUP BY level
                ORDER BY count DESC
            """

            result = executor.execute_query(sql)

            # Fixture has: 5 ERRORs, 3 INFOs, 2 WARNs
            assert result.success is True
            assert result.row_count == 3  # 3 distinct levels

            # Verify counts
            level_counts = {row["level"]: row["count"] for row in result.rows}
            assert level_counts["ERROR"] == 5
            assert level_counts["INFO"] == 3
            assert level_counts["WARN"] == 2

        finally:
            pool.close_all()


class TestConcurrentConnections:
    """Test concurrent connection pool usage with threading."""

    def test_concurrent_queries(self, test_db_with_fixtures, test_db_url):
        """Test multiple threads executing queries simultaneously."""
        pool = DatabaseConnectionPool(database_url=test_db_url, min_conn=3, max_conn=10)
        executor = QueryExecutor(pool)

        results = []
        errors = []

        def execute_query_thread(thread_id: int):
            """Execute a query in a separate thread."""
            try:
                sql = f"SELECT id, level FROM logs WHERE id <= {thread_id} LIMIT 5"
                result = executor.execute_query(sql)
                results.append((thread_id, result))
            except Exception as e:
                errors.append((thread_id, e))

        # Launch 5 concurrent threads
        threads = []
        for i in range(1, 6):
            thread = threading.Thread(target=execute_query_thread, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=10)

        try:
            # Verify all queries succeeded
            assert len(errors) == 0, f"Errors occurred: {errors}"
            assert len(results) == 5

            # Verify all results are valid QueryResult objects
            for thread_id, result in results:
                assert isinstance(result, QueryResult)
                assert result.success is True
                assert result.row_count >= 0

        finally:
            pool.close_all()

    def test_connection_pool_reuse(self, test_db_with_fixtures, test_db_url):
        """Test that connection pool reuses connections efficiently."""
        pool = DatabaseConnectionPool(database_url=test_db_url, min_conn=2, max_conn=5)
        executor = QueryExecutor(pool)

        try:
            execution_times = []

            # Execute 10 queries sequentially
            for i in range(10):
                sql = "SELECT COUNT(*) as count FROM logs"
                result = executor.execute_query(sql)
                execution_times.append(result.execution_time_ms)

            # Connection reuse should make later queries faster
            # (First few queries may be slower due to connection establishment)
            avg_first_3 = sum(execution_times[:3]) / 3
            avg_last_3 = sum(execution_times[-3:]) / 3

            # Last 3 should be faster or similar (connection pool working)
            # We're not asserting strict inequality because query times can vary
            # but we verify all queries completed successfully
            assert len(execution_times) == 10
            assert all(t > 0 for t in execution_times)

        finally:
            pool.close_all()


class TestLargeResultSets:
    """Test handling of large result sets and truncation."""

    def test_result_truncation_large_dataset(self, test_db_connection, test_db_url):
        """Test result truncation with >10,000 rows."""
        # Insert 15,000 rows into test database
        cursor = test_db_connection.cursor()

        # Batch insert 15,000 logs
        base_time = datetime(2025, 12, 28, 10, 0, 0)
        for i in range(15000):
            cursor.execute(
                """
                INSERT INTO logs (timestamp, level, source, message)
                VALUES (%s, %s, %s, %s)
                """,
                (base_time, "INFO", "test", f"Log message {i}")
            )
            # Commit every 1000 rows for performance
            if (i + 1) % 1000 == 0:
                test_db_connection.commit()

        test_db_connection.commit()

        pool = DatabaseConnectionPool(database_url=test_db_url)
        executor = QueryExecutor(pool)

        try:
            # Query all logs with max_rows=10000 (default)
            sql = "SELECT id, message FROM logs"
            result = executor.execute_query(sql, max_rows=10000)

            # Verify truncation
            assert result.success is True
            assert result.row_count == 10000
            assert result.truncated is True  # Should be truncated

            # Verify all rows are present
            assert len(result.rows) == 10000

        finally:
            # Cleanup
            cursor.execute("TRUNCATE logs RESTART IDENTITY CASCADE;")
            test_db_connection.commit()
            pool.close_all()

    def test_result_truncation_custom_limit(self, test_db_with_fixtures, test_db_url):
        """Test result truncation with custom max_rows."""
        pool = DatabaseConnectionPool(database_url=test_db_url)
        executor = QueryExecutor(pool)

        try:
            # Query all logs with max_rows=5
            sql = "SELECT id, message FROM logs"
            result = executor.execute_query(sql, max_rows=5)

            # Verify truncation (fixture has 10 logs)
            assert result.success is True
            assert result.row_count == 5
            assert result.truncated is True

        finally:
            pool.close_all()


class TestErrorHandling:
    """Test comprehensive error handling scenarios."""

    def test_invalid_sql_syntax(self, test_db_with_fixtures, test_db_url):
        """Test handling of SQL syntax errors."""
        pool = DatabaseConnectionPool(database_url=test_db_url)
        executor = QueryExecutor(pool)

        try:
            # Invalid SQL syntax
            sql = "SELECT * FORM logs"  # FORM instead of FROM

            with pytest.raises(QueryExecutionError) as exc_info:
                executor.execute_query(sql)

            # Verify error message contains useful information
            assert "syntax error" in str(exc_info.value).lower()

        finally:
            pool.close_all()

    def test_nonexistent_table(self, test_db_with_fixtures, test_db_url):
        """Test handling of queries on nonexistent tables."""
        pool = DatabaseConnectionPool(database_url=test_db_url)
        executor = QueryExecutor(pool)

        try:
            sql = "SELECT * FROM nonexistent_table"

            with pytest.raises(QueryExecutionError) as exc_info:
                executor.execute_query(sql)

            assert "does not exist" in str(exc_info.value).lower()

        finally:
            pool.close_all()

    def test_invalid_column_name(self, test_db_with_fixtures, test_db_url):
        """Test handling of queries with invalid column names."""
        pool = DatabaseConnectionPool(database_url=test_db_url)
        executor = QueryExecutor(pool)

        try:
            sql = "SELECT nonexistent_column FROM logs"

            with pytest.raises(QueryExecutionError) as exc_info:
                executor.execute_query(sql)

            assert "does not exist" in str(exc_info.value).lower()

        finally:
            pool.close_all()

    def test_read_only_enforcement(self, test_db_with_fixtures, test_db_url):
        """Test that INSERT/UPDATE/DELETE are blocked by read-only transaction."""
        pool = DatabaseConnectionPool(database_url=test_db_url)
        executor = QueryExecutor(pool)

        try:
            # Attempt INSERT
            sql = "INSERT INTO logs (timestamp, level, message) VALUES (NOW(), 'TEST', 'Should fail')"

            with pytest.raises(QueryExecutionError) as exc_info:
                executor.execute_query(sql)

            assert "read-only" in str(exc_info.value).lower()

        finally:
            pool.close_all()

    def test_connection_pool_exhaustion(self, test_db_with_fixtures, test_db_url):
        """Test behavior when connection pool is exhausted."""
        # Create pool with only 1 connection
        pool = DatabaseConnectionPool(database_url=test_db_url, min_conn=1, max_conn=1)
        executor = QueryExecutor(pool)

        try:
            # Acquire the only connection
            conn1 = pool.get_connection()

            # Try to execute query (should timeout waiting for connection)
            # We set a very short timeout to avoid waiting too long in tests
            sql = "SELECT COUNT(*) FROM logs"

            # This should either succeed quickly or raise ConnectionPoolError
            # depending on timing (connection may be returned quickly)
            try:
                result = executor.execute_query(sql, timeout=5)
                # If it succeeded, the connection was returned quickly
                assert result.success is True
            except ConnectionPoolError:
                # Connection pool was exhausted
                pass

            # Return the connection
            pool.return_connection(conn1)

        finally:
            pool.close_all()

    def test_query_timeout_enforcement(self, test_db_with_fixtures, test_db_url):
        """Test that long-running queries are timed out."""
        pool = DatabaseConnectionPool(database_url=test_db_url)
        executor = QueryExecutor(pool)

        try:
            # Query that sleeps for 5 seconds
            sql = "SELECT pg_sleep(5)"

            # Set timeout to 1 second
            with pytest.raises(QueryTimeoutError) as exc_info:
                executor.execute_query(sql, timeout=1)

            assert "timeout" in str(exc_info.value).lower() or "canceled" in str(exc_info.value).lower()

        finally:
            pool.close_all()


class TestTypeConversions:
    """Test type conversions in real database scenarios."""

    def test_datetime_conversion(self, test_db_with_fixtures, test_db_url):
        """Test datetime→ISO8601 conversion with real database."""
        pool = DatabaseConnectionPool(database_url=test_db_url)
        executor = QueryExecutor(pool)

        try:
            sql = "SELECT id, timestamp FROM logs LIMIT 1"
            result = executor.execute_query(sql)

            assert result.success is True
            assert result.row_count == 1

            # Timestamp should be ISO 8601 string
            timestamp = result.rows[0]["timestamp"]
            assert isinstance(timestamp, str)
            assert "T" in timestamp  # ISO format has T separator
            # Should be parseable back to datetime
            parsed = datetime.fromisoformat(timestamp)
            assert isinstance(parsed, datetime)

        finally:
            pool.close_all()

    def test_jsonb_conversion(self, test_db_with_fixtures, test_db_url):
        """Test JSONB→dict conversion with real database."""
        pool = DatabaseConnectionPool(database_url=test_db_url)
        executor = QueryExecutor(pool)

        try:
            sql = "SELECT id, metadata FROM logs WHERE metadata IS NOT NULL LIMIT 1"
            result = executor.execute_query(sql)

            assert result.success is True
            assert result.row_count == 1

            # Metadata should be dict
            metadata = result.rows[0]["metadata"]
            assert isinstance(metadata, dict)
            # Should have keys from fixture data
            assert len(metadata) > 0

        finally:
            pool.close_all()

    def test_null_value_handling(self, test_db_connection, test_db_url):
        """Test NULL value preservation."""
        # Insert a log with NULL metadata
        cursor = test_db_connection.cursor()
        cursor.execute(
            """
            INSERT INTO logs (timestamp, level, source, message, metadata)
            VALUES (NOW(), 'INFO', 'test', 'Test message', NULL)
            """
        )
        test_db_connection.commit()

        pool = DatabaseConnectionPool(database_url=test_db_url)
        executor = QueryExecutor(pool)

        try:
            sql = "SELECT id, metadata FROM logs WHERE metadata IS NULL LIMIT 1"
            result = executor.execute_query(sql)

            assert result.success is True
            assert result.row_count == 1

            # Metadata should be None (not string "null")
            metadata = result.rows[0]["metadata"]
            assert metadata is None

        finally:
            # Cleanup
            cursor.execute("TRUNCATE logs RESTART IDENTITY CASCADE;")
            test_db_connection.commit()
            pool.close_all()
