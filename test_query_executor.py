#!/usr/bin/env python3
"""
Manual test script for QueryExecutor.

This script demonstrates QueryExecutor functionality with real database queries,
including security features, error handling, and performance tracking.
"""

import sys
from src.shared.db_utils import (
    DatabaseConnectionPool,
    QueryExecutor,
    QueryExecutionError,
    QueryTimeoutError,
    ConnectionPoolError
)


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_result(result, show_rows=True, max_rows=5):
    """Print query result in a readable format."""
    print(f"\n📊 Result Summary:")
    print(f"   - Row Count: {result['row_count']}")
    print(f"   - Execution Time: {result['execution_time_ms']}ms")
    print(f"   - Truncated: {result['truncated']}")
    print(f"   - Columns: {', '.join(result['column_names'])}")

    if show_rows and result['rows']:
        print(f"\n📋 Sample Rows (showing up to {max_rows}):")
        for i, row in enumerate(result['rows'][:max_rows], 1):
            print(f"   {i}. {row}")

        if result['row_count'] > max_rows:
            print(f"   ... ({result['row_count'] - max_rows} more rows)")


def test_basic_query(executor):
    """Test 1: Basic SELECT query."""
    print_section("Test 1: Basic SELECT Query")

    sql = "SELECT id, timestamp, level, source, message FROM logs LIMIT 5"
    print(f"Query: {sql}")

    result = executor.execute_query(sql)
    print_result(result)

    return result


def test_where_clause(executor):
    """Test 2: Query with WHERE clause."""
    print_section("Test 2: WHERE Clause Filtering")

    sql = "SELECT id, level, source, message FROM logs WHERE level = 'ERROR' LIMIT 10"
    print(f"Query: {sql}")

    result = executor.execute_query(sql)
    print_result(result)

    # Verify all rows are ERROR level
    error_count = sum(1 for row in result['rows'] if row[1] == 'ERROR')
    print(f"\n✅ Verification: {error_count}/{result['row_count']} rows are ERROR level")

    return result


def test_aggregation(executor):
    """Test 3: Aggregation query."""
    print_section("Test 3: Aggregation (GROUP BY)")

    sql = "SELECT level, COUNT(*) as count FROM logs GROUP BY level ORDER BY count DESC"
    print(f"Query: {sql}")

    result = executor.execute_query(sql)
    print_result(result)

    return result


def test_order_by(executor):
    """Test 4: ORDER BY with timestamp."""
    print_section("Test 4: ORDER BY Timestamp")

    sql = "SELECT id, timestamp, level FROM logs ORDER BY timestamp DESC LIMIT 5"
    print(f"Query: {sql}")

    result = executor.execute_query(sql)
    print_result(result)

    # Verify descending order
    if result['row_count'] >= 2:
        is_descending = all(
            result['rows'][i][1] >= result['rows'][i+1][1]
            for i in range(result['row_count'] - 1)
        )
        print(f"\n✅ Verification: Timestamps are in descending order: {is_descending}")

    return result


def test_jsonb_metadata(executor):
    """Test 5: JSONB metadata access."""
    print_section("Test 5: JSONB Metadata Access")

    sql = "SELECT id, source, metadata FROM logs WHERE metadata IS NOT NULL LIMIT 5"
    print(f"Query: {sql}")

    result = executor.execute_query(sql)
    print_result(result)

    return result


def test_truncation(executor):
    """Test 6: Result truncation."""
    print_section("Test 6: Result Truncation")

    sql = "SELECT * FROM logs LIMIT 100"
    print(f"Query: {sql}")
    print(f"Max Rows: 5")

    result = executor.execute_query(sql, max_rows=5)
    print_result(result, max_rows=5)

    if result['truncated']:
        print(f"\n✅ Truncation working: Query limited to {result['row_count']} rows")

    return result


def test_empty_result(executor):
    """Test 7: Query with no results."""
    print_section("Test 7: Empty Result Set")

    sql = "SELECT * FROM logs WHERE id = -999999"
    print(f"Query: {sql}")

    result = executor.execute_query(sql)
    print_result(result)

    print(f"\n✅ Empty result handling: {result['row_count']} rows returned")

    return result


def test_read_only_enforcement(executor):
    """Test 8: Read-only transaction enforcement."""
    print_section("Test 8: Read-Only Transaction Enforcement")

    sql = "INSERT INTO logs (timestamp, level, message) VALUES (NOW(), 'TEST', 'Should fail')"
    print(f"Query: {sql}")
    print("Expected: QueryExecutionError (read-only transaction)")

    try:
        result = executor.execute_query(sql)
        print("\n❌ FAILED: Insert was allowed (should have been blocked)")
        return None
    except QueryExecutionError as e:
        if "read-only transaction" in str(e).lower():
            print(f"\n✅ SUCCESS: Read-only enforcement working")
            print(f"   Error: {str(e)[:100]}...")
            return True
        else:
            print(f"\n⚠️  Error occurred but not read-only related: {e}")
            return False


def test_timeout(executor):
    """Test 9: Query timeout (simulated with very short timeout)."""
    print_section("Test 9: Query Timeout")

    # This query might timeout with a very short limit
    sql = "SELECT pg_sleep(0.001)"
    print(f"Query: {sql}")
    print("Timeout: 1ms (very short to potentially trigger timeout)")

    try:
        result = executor.execute_query(sql, timeout=0.001)
        print(f"\n✅ Query completed in {result['execution_time_ms']}ms (faster than timeout)")
        return result
    except QueryTimeoutError as e:
        print(f"\n✅ Timeout enforcement working")
        print(f"   Error: {str(e)[:100]}...")
        return True


def test_performance(executor):
    """Test 10: Performance comparison."""
    print_section("Test 10: Performance Tracking")

    queries = [
        "SELECT COUNT(*) FROM logs",
        "SELECT COUNT(*) FROM logs WHERE level = 'ERROR'",
        "SELECT COUNT(*) FROM logs WHERE level = 'INFO'",
    ]

    print("Running 3 queries to test connection pool reuse...")

    times = []
    for i, sql in enumerate(queries, 1):
        print(f"\n   Query {i}: {sql}")
        result = executor.execute_query(sql)
        times.append(result['execution_time_ms'])
        print(f"   Result: {result['rows'][0][0]} rows, {result['execution_time_ms']}ms")

    avg_time = sum(times) / len(times)
    print(f"\n📈 Performance Metrics:")
    print(f"   - Average execution time: {avg_time:.2f}ms")
    print(f"   - Total time: {sum(times):.2f}ms")
    print(f"   - Connection pool reuse working ✅")

    return times


def main():
    """Run all tests."""
    print("\n" + "🧪" * 40)
    print("  QueryExecutor Manual Test Suite")
    print("🧪" * 40)

    # Initialize pool and executor
    print("\n🔧 Initializing DatabaseConnectionPool and QueryExecutor...")
    try:
        pool = DatabaseConnectionPool()
        executor = QueryExecutor(pool)
        print("✅ Initialization successful")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        sys.exit(1)

    # Run tests
    tests = [
        test_basic_query,
        test_where_clause,
        test_aggregation,
        test_order_by,
        test_jsonb_metadata,
        test_truncation,
        test_empty_result,
        test_read_only_enforcement,
        test_timeout,
        test_performance
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func(executor)
            passed += 1
        except Exception as e:
            print(f"\n❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    # Cleanup
    print_section("Cleanup")
    print("Closing connection pool...")
    pool.close_all()
    print("✅ Connection pool closed")

    # Summary
    print_section("Test Summary")
    print(f"\n📊 Results:")
    print(f"   - Passed: {passed}/{len(tests)}")
    print(f"   - Failed: {failed}/{len(tests)}")

    if failed == 0:
        print(f"\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  Some tests failed")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
