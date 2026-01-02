"""
Pytest configuration and fixtures for integration testing.

This module provides shared fixtures for database integration tests,
including test database setup/teardown and fixture data.
"""

import os
import pytest
import psycopg2
import psycopg2.extras
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# Test database configuration
TEST_DB_NAME = "log_analytics_test"
MAIN_DB_URL = os.getenv("DATABASE_URL", "postgresql://saikamat@localhost:5432/log_analytics")
# Replace database name in URL
TEST_DB_URL = MAIN_DB_URL.rsplit('/', 1)[0] + '/' + TEST_DB_NAME


def get_fixture_logs() -> List[Dict[str, Any]]:
    """
    Generate 10 fixture logs covering all 9 service types.

    Returns:
        List of 10 log dictionaries with known, predictable content for testing.
    """
    base_time = datetime(2025, 12, 28, 10, 0, 0)

    return [
        # 1. sshd - Login success
        {
            "timestamp": base_time,
            "level": "INFO",
            "source": "sshd",
            "message": "User login successful",
            "metadata": {"user": "alice", "ip": "192.168.1.100"}
        },
        # 2. sshd - Login failure
        {
            "timestamp": base_time + timedelta(minutes=5),
            "level": "ERROR",
            "source": "sshd",
            "message": "User login failed",
            "metadata": {"user": "bob", "ip": "192.168.1.101"}
        },
        # 3. nginx - 500 error
        {
            "timestamp": base_time + timedelta(minutes=10),
            "level": "ERROR",
            "source": "nginx",
            "message": "Internal server error",
            "metadata": {"status_code": "500", "path": "/api/users"}
        },
        # 4. nginx - Success
        {
            "timestamp": base_time + timedelta(minutes=15),
            "level": "INFO",
            "source": "nginx",
            "message": "Request processed successfully",
            "metadata": {"status_code": "200", "path": "/api/health"}
        },
        # 5. kernel - OOM
        {
            "timestamp": base_time + timedelta(minutes=20),
            "level": "ERROR",
            "source": "kernel",
            "message": "Out of memory: Kill process",
            "metadata": {"process": "python", "pid": 1234}
        },
        # 6. postgresql - Connection error
        {
            "timestamp": base_time + timedelta(minutes=25),
            "level": "ERROR",
            "source": "postgresql",
            "message": "Connection refused",
            "metadata": {"database": "app_db", "host": "localhost"}
        },
        # 7. docker - Container started
        {
            "timestamp": base_time + timedelta(minutes=30),
            "level": "INFO",
            "source": "docker",
            "message": "Container started",
            "metadata": {"container": "web-server", "id": "abc123"}
        },
        # 8. systemd - Service failed
        {
            "timestamp": base_time + timedelta(minutes=35),
            "level": "ERROR",
            "source": "systemd",
            "message": "Service failed to start",
            "metadata": {"service": "redis.service", "code": "1"}
        },
        # 9. auth - Permission denied
        {
            "timestamp": base_time + timedelta(minutes=40),
            "level": "WARN",
            "source": "auth",
            "message": "Permission denied",
            "metadata": {"user": "charlie", "resource": "/etc/shadow"}
        },
        # 10. networking - DNS timeout
        {
            "timestamp": base_time + timedelta(minutes=45),
            "level": "WARN",
            "source": "networking",
            "message": "DNS lookup timed out",
            "metadata": {"hostname": "api.example.com", "timeout": "5s"}
        }
    ]


@pytest.fixture(scope="session")
def test_db_connection():
    """
    Create test database and return connection.

    This fixture:
    1. Connects to postgres database
    2. Drops test database if it exists (from previous run)
    3. Creates fresh test database
    4. Creates pgvector extension
    5. Creates logs table schema
    6. Yields connection to test database
    7. Drops test database on teardown
    """
    # Connect to postgres database to create test database
    postgres_url = MAIN_DB_URL.rsplit('/', 1)[0] + '/postgres'
    conn = psycopg2.connect(postgres_url)
    conn.autocommit = True
    cursor = conn.cursor()

    # Drop test database if it exists (cleanup from previous run)
    cursor.execute(f"DROP DATABASE IF EXISTS {TEST_DB_NAME};")

    # Create test database
    cursor.execute(f"CREATE DATABASE {TEST_DB_NAME};")

    cursor.close()
    conn.close()

    # Connect to test database
    test_conn = psycopg2.connect(TEST_DB_URL)
    test_cursor = test_conn.cursor()

    # Create pgvector extension
    test_cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # Create logs table schema
    test_cursor.execute("""
        CREATE TABLE logs (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ NOT NULL,
            level VARCHAR(20),
            source VARCHAR(255),
            message TEXT NOT NULL,
            metadata JSONB,
            embedding VECTOR(384)
        );
    """)

    # Create indexes
    test_cursor.execute("""
        CREATE INDEX idx_logs_timestamp ON logs (timestamp DESC);
    """)
    test_cursor.execute("""
        CREATE INDEX idx_logs_level ON logs (level);
    """)
    test_cursor.execute("""
        CREATE INDEX idx_logs_embedding ON logs USING hnsw (embedding vector_cosine_ops);
    """)

    test_conn.commit()

    yield test_conn

    # Teardown: close connection and drop test database
    test_cursor.close()
    test_conn.close()

    # Reconnect to postgres database to drop test database
    conn = psycopg2.connect(postgres_url)
    conn.autocommit = True
    cursor = conn.cursor()
    cursor.execute(f"DROP DATABASE IF EXISTS {TEST_DB_NAME};")
    cursor.close()
    conn.close()


@pytest.fixture(scope="function")
def test_db_with_fixtures(test_db_connection):
    """
    Provide test database with fixture data inserted.

    This fixture:
    1. Inserts 10 known fixture logs
    2. Yields connection
    3. Clears all data after test (TRUNCATE logs table)
    """
    cursor = test_db_connection.cursor()

    # Insert fixture logs (without embeddings for simplicity)
    fixture_logs = get_fixture_logs()
    for log in fixture_logs:
        cursor.execute(
            """
            INSERT INTO logs (timestamp, level, source, message, metadata)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (log["timestamp"], log["level"], log["source"], log["message"],
             psycopg2.extras.Json(log["metadata"]))
        )

    test_db_connection.commit()

    yield test_db_connection

    # Cleanup: truncate logs table after each test
    cursor.execute("TRUNCATE logs RESTART IDENTITY CASCADE;")
    test_db_connection.commit()


@pytest.fixture(scope="function")
def test_db_url():
    """Return test database URL for DatabaseConnectionPool initialization."""
    return TEST_DB_URL
