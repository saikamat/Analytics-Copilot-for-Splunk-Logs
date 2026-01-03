# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## System Overview

This is an **AI-powered log analytics system** - a Splunk alternative that uses:
- **PostgreSQL with pgvector** for storage and semantic search
- **Sentence-transformers** to convert log messages into 384-dimensional embeddings
- **LLMs (AWS Bedrock Claude)** to translate natural language queries into SQL
- **FastAPI** for REST API backend
- **Python** for ETL pipeline and backend services

The project follows a **4-phase development approach**:
1. **Phase 1** (Complete): Foundation & Data Pipeline
2. **Phase 2** (Complete): LLM Integration for natural language queries
   - ✅ 2a: AWS Bedrock SQL Generator
   - ✅ 2b: Database Utilities (connection pooling, query execution)
   - ✅ 2c: FastAPI Backend (REST API endpoints)
   - ✅ 2d: Result Summarization (LLM-based summaries)
3. **Phase 3**: Intelligence Layer (anomaly detection, runbooks)
4. **Phase 4**: Web App & AWS Deployment

## Architecture

### Data Flow

**Data Ingestion**:
```
Log Generation → ETL Pipeline → PostgreSQL (with embeddings)
```

1. **Log Generation** (`data/log_generator.py`): Creates synthetic system logs from 9 services (sshd, nginx, kernel, etc.) with realistic metadata
2. **ETL Pipeline** (`src/data_pipeline/etl.py`):
   - Generates embeddings using sentence-transformers (all-MiniLM-L6-v2)
   - Converts embeddings to lists for PostgreSQL compatibility
   - Serializes metadata dicts to JSON
   - Batch inserts using `psycopg2.extras.execute_batch`
3. **Storage**: PostgreSQL `logs` table with VECTOR(384) column

**Query Pipeline**:
```
HTTP Request → FastAPI → BedrockSQLGenerator → QueryExecutor → PostgreSQL → BedrockResultSummarizer → Results
```

1. **HTTP API** (`src/backend/app.py`): FastAPI REST API with Pydantic validation
2. **NL→SQL Translation** (`src/shared/bedrock_client.py`): Claude translates natural language to SQL
3. **SQL Execution** (`src/shared/db_utils.py`): QueryExecutor with security and timeouts
4. **Database**: PostgreSQL with pgvector for semantic search using `<=>` operator
5. **Result Summarization** (`src/shared/result_summarizer.py`): Optional Claude-powered summaries of query results

### Database Schema

**Table: `logs`** (defined in `config/schemas.md`)
```sql
CREATE TABLE logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    level VARCHAR(20),              -- INFO, WARN, ERROR
    source VARCHAR(255),            -- Service name (sshd, nginx, etc.)
    message TEXT NOT NULL,
    metadata JSONB,                 -- Service-specific fields
    embedding VECTOR(384)           -- Semantic search vector
);
```

**Indexes** (defined in `config/indices.sql`):
- `idx_logs_timestamp`: B-tree index with DESC ordering for time-range queries
- `idx_logs_level`: B-tree index for filtering by severity
- `idx_logs_embedding`: HNSW index with `vector_cosine_ops` for fast similarity search

### Why These Technologies?

**pgvector**:
- Stores 384-dimensional vectors natively as VECTOR type
- HNSW (Hierarchical Navigable Small World) indexing enables logarithmic search time
- Integrates with PostgreSQL's ACID guarantees and familiar SQL

**sentence-transformers (all-MiniLM-L6-v2)**:
- Pre-trained on semantic similarity tasks
- Fast CPU inference (~50ms per embedding)
- 384-dimensional output matches schema
- No fine-tuning needed for log analysis

**PostgreSQL over specialized vector databases**:
- Single system for both logs and embeddings
- Enables complex queries (temporal + semantic filtering)
- Cost-effective for learning project
- Familiar SQL for future LLM-to-SQL translation

## Development Commands

### Virtual Environment

**IMPORTANT**: This project uses a Python virtual environment (`venv_llm_logs`) to manage dependencies. Always activate it before running commands:

```bash
# Activate virtual environment
source venv_llm_logs/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify activation (should show venv path)
which python

# Deactivate when done
deactivate
```

**Why Virtual Environment**: Python 3.14.2 installed via Homebrew uses PEP 668 externally-managed-environment protection, preventing direct `pip install` commands. The virtual environment bypasses this restriction.

**Package Installation**: Always install packages within the activated virtual environment:
```bash
source venv_llm_logs/bin/activate
pip install package-name
```

### Database Setup

```bash
# Create database and extension
psql postgres -c "CREATE DATABASE log_analytics;"
psql -d log_analytics -c "CREATE EXTENSION vector;"

# Create schema and indexes
psql -d log_analytics -f config/schemas.md
psql -d log_analytics -f config/indices.sql

# Verify setup
psql -d log_analytics -c "\d logs"
psql -d log_analytics -c "SELECT * FROM pg_extension WHERE extname='vector';"
```

### Running the ETL Pipeline

**IMPORTANT**: Always run Python modules using the `-m` flag from the project root:

```bash
# Generate and insert 10 logs
python -m src.data_pipeline.etl

# Generate and insert N logs programmatically
python -c "from data.log_generator import generate_logs; from src.data_pipeline.etl import insert_logs; insert_logs(generate_logs(1000))"
```

**Why `-m` flag?**
When you run `python -m src.data_pipeline.etl`, Python sets `sys.path[0]` to the current working directory (project root), which allows imports like `from data.log_generator import generate_logs` to resolve correctly. Running as `python src/data_pipeline/etl.py` sets `sys.path[0]` to `src/data_pipeline`, breaking cross-module imports.

### Running Semantic Search

```bash
# Test semantic search with sample query
python -m src.data_pipeline.search

# Query example: "what were login related issues yesterday?"
# Returns top 5 semantically similar logs with distance scores
```

### Testing & Verification

```bash
# Count total logs
psql -d log_analytics -c "SELECT COUNT(*) FROM logs;"

# View sample logs
psql -d log_analytics -c "SELECT id, timestamp, level, source, message, metadata FROM logs LIMIT 5;"

# Check log level distribution
psql -d log_analytics -c "SELECT level, COUNT(*) FROM logs GROUP BY level ORDER BY COUNT(*) DESC;"

# Verify embeddings exist
psql -d log_analytics -c "SELECT id, message, embedding IS NOT NULL as has_embedding FROM logs LIMIT 3;"

# Interactive database exploration
psql -d log_analytics
# Inside psql:
# \dt               -- List tables
# \d logs           -- Describe logs table
# \x                -- Toggle expanded display (for wide rows)
# \q                -- Exit
```

## Key Patterns

### Configuration Management
- **Environment Variables**: Database connection via `DATABASE_URL` in `.env` file
- **Loading Config**: Uses `python-dotenv` to load `.env` at module initialization
- **Example**: `DATABASE_URL="postgresql://username@localhost:5432/log_analytics"`

### Embedding Model Loading
The embedding model is loaded **once at module level** in `src/data_pipeline/etl.py`:
```python
embedding_model = load_embedding_model()  # Loads 90MB model once
```
This avoids reloading the model on every function call (critical for performance).

### Batch Operations
Use `psycopg2.extras.execute_batch()` for efficient bulk inserts:
```python
execute_batch(cursor, sql, data_to_insert)  # Single transaction vs N roundtrips
```

### Metadata Handling
- Stored as JSONB for flexible schema
- Service-specific fields (e.g., sshd has user/IP, kernel has process/PID)
- Serialized with `json.dumps()` before insertion

## Log Generator Details

**9 Service Types** (`data/log_generator.py`):
- `sshd`, `kernel`, `systemd`, `nginx`, `postgresql`, `docker`, `cron`, `auth`, `networking`

**Log Level Distribution**:
- INFO: 70%
- WARN: 20%
- ERROR: 10%

**Timestamp Generation**:
- Spread over 10,000 minute window (~7 days)
- Uses `datetime.now() - timedelta(minutes=random.randint(0, 10000))`

**Service-Specific Messages**:
Each service has 3-5 realistic message templates (e.g., sshd: "User login successful", "User login failed", "Connection closed")

## AWS Bedrock SQL Generator

**Location**: `src/shared/bedrock_client.py`

Translates natural language queries to SQL using Claude on AWS Bedrock. This is a critical Phase 2 component that enables users to query logs using natural language instead of writing SQL.

### How it Works

```python
from src.shared.bedrock_client import BedrockSQLGenerator

generator = BedrockSQLGenerator()
sql = generator.generate_sql("show errors from yesterday")
# Returns: SELECT id, timestamp, level, source, message, metadata
#          FROM logs WHERE level = 'ERROR'
#          AND timestamp >= NOW() - INTERVAL '1 day'
#          ORDER BY timestamp DESC LIMIT 100;
```

### Architecture

**Core Components**:
1. **Schema Context Builder** (`_build_schema_context()`): Provides the LLM with database schema, indexes, and PostgreSQL-specific operators
2. **Few-Shot Examples** (`_build_few_shot_examples()`): 5 diverse examples teaching SQL patterns for temporal queries, metadata access, aggregations, text search, and combined filters
3. **SQL Generator** (`generate_sql()`): Main API that invokes Bedrock and returns validated SQL
4. **Three-Layer Validator** (`validate_sql()`): Security validation to prevent SQL injection

### Bedrock API Configuration

**Model**: Claude 3 Haiku via inference profile `us.anthropic.claude-3-haiku-20240307-v1:0`
- **Why Haiku**: Cost-effective (~$0.0005/query), fast (3-5s), accurate for single-table SELECT queries
- **Why Inference Profile**: Direct model IDs don't support on-demand throughput; cross-region inference profiles (`us.*`) are required

**Request Format** (Bedrock Messages API):
```python
{
    "anthropic_version": "bedrock-2023-05-31",  # Required for Messages API
    "max_tokens": 1024,
    "temperature": 0.0,  # Deterministic output for SQL reliability
    "messages": [{"role": "user", "content": schema + examples + query}],
    "system": SYSTEM_PROMPT
}
```

**Temperature 0.0**: Same natural language query always produces same SQL (critical for reliability)

### Three-Layer SQL Validation

**Layer 1 - Syntax**: Uses `sqlparse` to verify query is parseable
**Layer 2 - Semantics**:
- Only SELECT statements allowed (blocks INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE, EXEC)
- Only `logs` table allowed
- Must start with SELECT keyword

**Layer 3 - Security**: Checks for injection patterns (`--`, `/*`, `UNION`, `xp_`, `sp_`)

Raises `ValidationError` with descriptive messages on failure.

### Error Handling

**Custom Exceptions**:
- `BedrockError`: Base exception for AWS Bedrock API errors
- `ValidationError`: SQL validation failures

**Retry Logic**: Exponential backoff for `ThrottlingException`
- Max 3 retries with base delay 1.0 seconds
- Delays: 1s, 2s, 4s
- Re-raises non-throttling errors immediately

### Environment Configuration

Required in `.env`:
```bash
AWS_REGION="us-east-1"
BEDROCK_MODEL_ID="us.anthropic.claude-3-haiku-20240307-v1:0"  # Inference profile
AWS_ACCESS_KEY_ID="your_key"
AWS_SECRET_ACCESS_KEY="your_secret"
```

**CRITICAL**: Must use inference profile IDs (prefixed with `us.`, `eu.`, or `ap.`) for on-demand throughput. Direct model IDs like `anthropic.claude-haiku-4-5-20251001-v1:0` will fail with `ValidationException`.

### Testing

```bash
# Activate virtual environment first
source venv_llm_logs/bin/activate

# Run test suite
python -m src.shared.bedrock_client
```

**Test Coverage**: 5 diverse queries testing temporal filtering, text search, aggregations, date functions, and JSONB metadata extraction.

### Few-Shot Learning Strategy

**5 Examples Cover**:
1. **Temporal filtering**: `"show me errors from yesterday"` → Uses indexed `timestamp` and `level` columns
2. **Metadata access**: `"find nginx 500 errors"` → JSONB operator `metadata->>'status_code'`
3. **Aggregation**: `"count errors by service"` → `GROUP BY` with `ORDER BY COUNT(*) DESC`
4. **Text search**: `"find login failures"` → `ILIKE` pattern matching
5. **Combined filters**: `"show nginx errors in the last hour"` → Multiple indexed columns

**Why 5 Examples**: Provides sufficient context for diverse query types while keeping prompt size manageable (~1,420 tokens).

### Cost Estimates

- **Per query**: ~$0.0005 (input: ~1,420 tokens, output: ~100 tokens)
- **1,000 queries/day**: $15/month
- **10,000 queries/day**: $150/month

### Integration Pattern

The Bedrock client is designed to integrate with a future query pipeline:
```
Natural Language → BedrockSQLGenerator → Validated SQL → QueryExecutor → Results → LLM Summarization
```

## Database Utilities

**Location**: `src/shared/db_utils.py`

Provides connection pooling, query execution, and result formatting for executing validated SQL against PostgreSQL.

### Architecture

```python
BedrockSQLGenerator.generate_sql(nl_query)
    ↓ Validated SQL
QueryExecutor.execute_query(sql, timeout=30, max_rows=10000)
    ↓ Security: Read-only transaction, timeout enforcement
PostgreSQL Connection Pool (2-10 connections)
    ↓ Raw results: List[tuple]
ResultFormatter.format_results()
    ↓ Type conversion: datetime→ISO8601, JSONB→dict, NULL→None
QueryResult (immutable dataclass)
```

### Core Components

**DatabaseConnectionPool**:
- Manages PostgreSQL connections with psycopg2.pool.SimpleConnectionPool
- Configuration: `min_conn=2`, `max_conn=10` (configurable via env vars)
- Thread-safe for concurrent FastAPI requests
- Performance: 5x faster queries (12ms vs 65ms) via connection reuse

**QueryExecutor**:
- Executes SQL with three security layers:
  1. Read-only transaction (`SET TRANSACTION READ ONLY`)
  2. Query timeout (`SET statement_timeout = 30000`)
  3. Result limit (max 10,000 rows, truncate if exceeded)
- Retry logic: Retry connection failures once, then raise ConnectionPoolError
- Performance tracking: Returns execution time in milliseconds

**ResultFormatter**:
- Converts raw PostgreSQL tuples to structured dicts
- Type conversions:
  - `datetime` → ISO 8601 string: `"2025-12-28T20:25:48+01:00"`
  - JSONB (str/dict) → dict via `json.loads()`
  - `numpy.ndarray` → list via `.tolist()`
  - `NULL` → `None` (preserved)
- Graceful error handling: Unknown types → `str(value)` with warning

**QueryResult** (dataclass):
```python
@dataclass(frozen=True)
class QueryResult:
    success: bool
    rows: List[Dict[str, Any]]
    row_count: int
    column_names: List[str]
    execution_time_ms: float
    truncated: bool = False
    error_message: Optional[str] = None
```

### Usage

```python
from src.shared.db_utils import DatabaseConnectionPool, QueryExecutor

# Initialize once (module level or application startup)
pool = DatabaseConnectionPool()
executor = QueryExecutor(pool)

# Execute query
sql = "SELECT * FROM logs WHERE level = 'ERROR' LIMIT 10"
result = executor.execute_query(sql, timeout=30, max_rows=100)

# Access structured results
print(f"Found {result.row_count} errors in {result.execution_time_ms}ms")
for row in result.rows:
    print(f"{row['timestamp']}: {row['message']}")

# Cleanup on shutdown
pool.close_all()
```

### Configuration

**Environment Variables** (`.env`):
```bash
DATABASE_URL="postgresql://saikamat@localhost:5432/log_analytics"
DB_POOL_MIN_CONN=2   # Optional, default 2
DB_POOL_MAX_CONN=10  # Optional, default 10
```

**Connection Pool Sizing**:
- Web API: `max_conn = 2 × num_workers`
- Background jobs: `min_conn = 1`, `max_conn = 5`
- High concurrency: `max_conn = 20-50`

### Exception Hierarchy

```python
DatabaseError                    # Base exception
├── ConnectionPoolError          # Pool exhausted or unreachable
├── QueryExecutionError          # SQL errors, read-only violations
├── QueryTimeoutError            # Query exceeded timeout
└── ResultFormattingError        # Type conversion errors
```

### Performance

| Metric | Value |
|--------|-------|
| Connection pool benefit | 5.4x faster (12ms vs 65ms) |
| Query timeout default | 30 seconds |
| Max rows default | 10,000 rows |
| Memory per query | ~20MB (10,000 rows × 2KB/row) |

### Security Layers

1. **SQL Validation** (BedrockSQLGenerator): Blocks DROP/DELETE/UNION
2. **Read-Only Transaction** (PostgreSQL): Enforces at database level
3. **Query Timeout** (PostgreSQL): Prevents long-running queries
4. **Result Limit** (Application): Prevents memory exhaustion
5. **Connection Pool Limit**: Prevents accidental DDoS (max 10 connections)

### Testing

**Test Database**: Separate `log_analytics_test` database with 10 fixture logs

**Run Tests**:
```bash
source venv_llm_logs/bin/activate

# All database utilities tests (70 tests)
python -m pytest tests/ -v

# Unit tests only (fast, no database required)
python -m pytest tests/test_db_utils.py::TestDatabaseConnectionPoolUnit -v

# End-to-end integration tests
python -m pytest tests/test_integration_e2e.py -v
```

**Test Coverage**: 70 tests total
- DatabaseConnectionPool: 20 tests (15 unit + 5 integration)
- QueryExecutor: 17 tests (8 unit + 9 integration)
- ResultFormatter: 15 unit tests
- End-to-end: 18 integration tests (full NL→SQL→Results flow, concurrency, error handling)

**For comprehensive testing documentation**, see README.md "Database Utilities" section and CLAUDE.md "Testing" section.

## Semantic Search Implementation

**Location**: `src/data_pipeline/search.py`

**How it works**:
```python
def search_logs(query: str, limit: int = 5) -> list[dict]:
    # 1. Convert natural language query to embedding
    query_embedding = embedding_model.encode(query)
    query_embedding_list = query_embedding.tolist()

    # 2. Query PostgreSQL with vector similarity
    sql = """
        SELECT id, timestamp, level, source, message, metadata,
               embedding <=> %s::vector AS distance
        FROM logs
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """
    cursor.execute(sql, (query_embedding_list, query_embedding_list, limit))

    # 3. Return results sorted by semantic similarity
    return cursor.fetchall()
```

**Key Implementation Details**:
- **Vector type casting**: The `::vector` cast tells PostgreSQL to treat the list as a vector type (required for `<=>` operator)
- **Duplicate embedding parameter**: The query embedding appears twice - once for calculating distance in SELECT, once for sorting in ORDER BY (SQL doesn't allow referencing aliases in the same query level)
- **Distance scores**: Lower distance = more similar (0 = identical, 2 = opposite)
- **Model loading**: Embedding model is loaded once at module level for performance

**Usage Example**:
```bash
$ python -m src.data_pipeline.search

Search results for => what were login related issues yesterday?

(4, ..., 'INFO', 'auth', 'User login successful', {...}, 0.43)
(7, ..., 'INFO', 'nginx', 'Connection timed out', {...}, 0.64)
(3, ..., 'WARN', 'auth', 'Permission denied', {...}, 0.72)
```

**pgvector Distance Operators**:
- `<=>`: Cosine distance (used for text embeddings)
- `<->`: L2/Euclidean distance
- `<#>`: Negative inner product

**For detailed explanation**, see `notes/vector_search_explained.md`

## Testing

The project includes comprehensive test coverage for all database utilities and end-to-end integration testing.

### Test Organization

**Location**: `/tests` directory with four main files:
- `conftest.py`: Pytest fixtures for test database setup/teardown and fixture data
- `test_db_utils.py`: Unit and integration tests for database utilities (52 tests)
- `test_integration_e2e.py`: End-to-end integration tests (18 tests)
- `test_api.py`: FastAPI integration tests (15 tests)

**Total Test Count**: 85 tests covering unit, integration, end-to-end, and API scenarios

### Running Tests

```bash
# Activate virtual environment first
source venv_llm_logs/bin/activate

# Run ALL tests (database utilities + API integration)
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_db_utils.py -v        # 52 database utility tests
python -m pytest tests/test_integration_e2e.py -v  # 18 end-to-end tests
python -m pytest tests/test_api.py -v             # 15 API integration tests

# Run with short traceback (cleaner output)
python -m pytest tests/ -v --tb=short

# Run specific test class or method
python -m pytest tests/test_integration_e2e.py::TestEndToEndFlow::test_end_to_end_with_bedrock -v
```

**Total Test Coverage**: 85 tests (52 database utilities + 18 end-to-end + 15 API)

### Test Database Setup

Integration tests use a **separate test database** (`log_analytics_test`) to avoid modifying production data:

1. **Session-level fixture** (`test_db_connection`): Creates test database, schema, and indexes
2. **Function-level fixture** (`test_db_with_fixtures`): Inserts 10 known fixture logs and cleans up after each test
3. **Automatic teardown**: Drops test database when test session completes

**Fixture Data**: 10 logs covering all 9 service types (sshd, nginx, kernel, postgresql, docker, systemd, auth, networking, cron) with known timestamps, levels, and metadata.

### Test Coverage by Component

#### DatabaseConnectionPool (20 tests)
- **Unit tests (15)**: Initialization, connection acquisition/return, health checks, error handling
- **Integration tests (5)**: Real database connections, pool reuse, query execution

#### QueryExecutor (17 tests)
- **Unit tests (8)**: Query execution, timeouts, truncation, SQL errors, connection retry
- **Integration tests (9)**: Real queries (SELECT, WHERE, aggregation, ORDER BY, JSONB), read-only enforcement, performance tracking

#### ResultFormatter (15 tests)
- Type conversions: datetime→ISO8601, JSONB→dict, numpy→list, NULL→None, primitives
- Result formatting with metadata (execution time, truncation, column names)
- QueryResult immutability verification

#### End-to-End Integration (18 tests)

**TestEndToEndFlow (5 tests)**:
- Full NL→SQL→Execution→Results flow with Bedrock API
- Mock Bedrock integration (no AWS credentials required)
- Temporal queries with fixture data verification
- JSONB metadata filtering and extraction
- Aggregation queries (GROUP BY with fixtures)

**TestConcurrentConnections (2 tests)**:
- Multi-threaded query execution (5 concurrent threads)
- Connection pool reuse efficiency verification

**TestLargeResultSets (2 tests)**:
- Result truncation with >10,000 rows
- Custom max_rows limit enforcement

**TestErrorHandling (6 tests)**:
- SQL syntax errors
- Nonexistent table/column errors
- Read-only transaction enforcement (blocks INSERT/UPDATE/DELETE)
- Connection pool exhaustion handling
- Query timeout enforcement

**TestTypeConversions (3 tests)**:
- Real database datetime→ISO8601 conversion
- Real database JSONB→dict conversion
- NULL value preservation verification

### Test Performance

- **All 70 tests run in ~4 seconds**
- Test database creation adds ~0.5s overhead per session
- Individual test execution time: <100ms (95th percentile)
- Concurrent tests verify connection pool thread safety

### Mock vs Real Database Tests

**Unit Tests**: Use mocked connections (no real database required)
- Fast execution (<10ms per test)
- Test error scenarios without database dependency
- Isolated component testing

**Integration Tests**: Use real PostgreSQL database
- Verify actual database behavior
- Test type conversions with real PostgreSQL types
- Validate query execution with real data

**End-to-End Tests**: Use real test database with fixture data
- Full workflow verification
- Known fixture data enables deterministic assertions
- Separate test database prevents production data contamination

### Adding New Tests

When adding new database functionality:

1. **Write unit tests first** in `test_db_utils.py` with mocked dependencies
2. **Add integration tests** using `test_db_with_fixtures` fixture for real database verification
3. **Update fixture data** in `conftest.py` if new scenarios require specific test data
4. **Run full test suite** to verify no regressions: `pytest tests/ -v`

## FastAPI Backend

**Location**: `src/backend/`

Provides a REST API for natural language log queries. Integrates BedrockSQLGenerator and QueryExecutor to expose HTTP endpoints for web/mobile applications.

### Architecture

```
HTTP POST /api/query
{"query": "show errors from yesterday", "max_rows": 100}
    ↓
FastAPI Request Validation (Pydantic)
    ↓
BedrockSQLGenerator.generate_sql(query)
    ↓ Validated SQL
QueryExecutor.execute_query(sql, timeout, max_rows)
    ↓ QueryResult
FastAPI Response (JSON)
    ↓
HTTP 200 OK
{"success": true, "rows": [...], "row_count": 15, "execution_time_ms": 1567.3}
```

### Core Components

**`src/backend/app.py`** - FastAPI application setup:
- Lifespan context manager for connection pool lifecycle
- CORS middleware configuration
- Automatic API documentation at `/docs` (Swagger) and `/redoc`

**`src/backend/routes.py`** - API endpoints:
- `POST /api/query`: Execute natural language queries
- `GET /api/health`: Database health check

**`src/backend/models.py`** - Pydantic request/response models:
- `QueryRequest`: Validates query (1-500 chars), max_rows (1-50,000), timeout (1-300s)
- `QueryResponse`: Structured results with rows, row_count, execution_time_ms, truncated flag
- `ErrorResponse`: Consistent error format with success=false, error message, error_type

**`src/backend/config.py`** - Configuration from environment variables using Pydantic Settings

### Running the Server

```bash
# Activate virtual environment first
source venv_llm_logs/bin/activate

# Development mode (hot reload)
uvicorn src.backend.app:app --reload --host 0.0.0.0 --port 8000

# Production mode (multi-worker)
uvicorn src.backend.app:app --host 0.0.0.0 --port 8000 --workers 9 --log-level info
```

**Server URLs**:
- API: http://localhost:8000
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### API Endpoints

#### POST /api/query

Execute a natural language query against logs.

**Request**:
```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "show me nginx errors from yesterday",
    "max_rows": 100,
    "timeout": 30
  }'
```

**Response (200 OK)**:
```json
{
  "success": true,
  "rows": [
    {
      "id": 42,
      "timestamp": "2025-12-28T14:30:00+01:00",
      "level": "ERROR",
      "source": "nginx",
      "message": "Connection timeout",
      "metadata": {"status_code": "500"}
    }
  ],
  "row_count": 15,
  "column_names": ["id", "timestamp", "level", "source", "message", "metadata"],
  "execution_time_ms": 1567.3,
  "truncated": false,
  "sql_query": "SELECT * FROM logs WHERE level = 'ERROR' ..."
}
```

**Error Responses**:

| HTTP Status | Error Type | Description |
|-------------|------------|-------------|
| 400 | ValidationError | SQL validation failed (dangerous operation blocked) |
| 400 | QueryExecutionError | SQL execution error (syntax, invalid column) |
| 408 | QueryTimeoutError | Query exceeded timeout limit |
| 422 | Pydantic | Invalid request parameters |
| 502 | BedrockError | AWS Bedrock API error |
| 503 | ConnectionPoolError | Database unavailable |

#### GET /api/health

Check API health and dependencies.

**Request**:
```bash
curl http://localhost:8000/api/health
```

**Response (200 OK)**:
```json
{
  "status": "healthy",
  "database_connected": true,
  "bedrock_available": true
}
```

### Configuration

**Environment Variables** (`.env`):
```bash
# FastAPI
FASTAPI_HOST="0.0.0.0"
FASTAPI_PORT=8000
FASTAPI_LOG_LEVEL="info"

# CORS (comma-separated origins)
CORS_ORIGINS="http://localhost:3000,http://localhost:8080"

# Query Limits
DEFAULT_TIMEOUT=30
MAX_TIMEOUT=300
DEFAULT_MAX_ROWS=10000
MAX_MAX_ROWS=50000
```

**Pydantic Settings**: Add `extra = "ignore"` to Config class to allow unknown environment variables (like AWS credentials).

### Security

The API implements 8 layers of security:
1. Input Validation (Pydantic)
2. SQL Validation (BedrockSQLGenerator)
3. Read-Only Transaction (QueryExecutor)
4. Query Timeout (QueryExecutor)
5. Result Limits (QueryExecutor)
6. Connection Pool Limits (DatabaseConnectionPool)
7. CORS Policy (FastAPI Middleware)
8. Error Sanitization (no SQL in error messages)

### Testing

**API Integration Tests** (`tests/test_api.py` - 15 tests):

```bash
# Run all API tests
python -m pytest tests/test_api.py -v

# Run with short traceback
python -m pytest tests/test_api.py -v --tb=short
```

**Test Coverage**:
- Root endpoint (API info)
- Health check endpoint
- Query endpoint with mocked Bedrock
- Empty query validation (422)
- Dangerous SQL blocking (400)
- Bedrock error handling (502)
- Query length validation (422)
- Invalid max_rows/timeout validation (422)
- Response structure verification
- Error response consistency
- Summary enabled (`include_summary: true`)
- Summary disabled (`include_summary: false`)
- Summary default behavior (defaults to `true`)
- Summary failure handling (graceful degradation)

**Key Testing Pattern**: Use TestClient with context manager to trigger lifespan events:

```python
@pytest.fixture(scope="module")
def test_client():
    """Create TestClient with lifespan context."""
    with TestClient(app) as client:
        yield client
```

This ensures `db_pool` and `query_executor` are initialized properly.

### Performance

| Metric | Target |
|--------|--------|
| Total API Latency (95th percentile) | <2.5s |
| - Bedrock API (NL→SQL) | ~1,500ms |
| - Database Query | <100ms |
| - Result Formatting | <10ms |
| - FastAPI Overhead | <10ms |

**Memory Usage**:
- Per Request: ~20MB (max 10,000 rows × 2KB/row)
- Connection Pool: ~100MB (10 connections × 10MB each)
- Total Application: ~200MB baseline + (20MB × concurrent requests)

**Concurrency**: Max concurrent requests limited by connection pool (default: 10)

### Documentation

Comprehensive FastAPI backend documentation available at `notes/fastapi_backend.md` (350+ lines) covering:
- Complete API reference
- Architecture diagrams
- Configuration guide
- Security documentation
- Performance metrics
- Troubleshooting guide
- Future enhancements

## Result Summarization (Phase 2d)

**Location**: `src/shared/result_summarizer.py`

Provides LLM-powered natural language summaries of SQL query results. Integrated with the FastAPI `/api/query` endpoint to give users concise, insight-focused summaries instead of raw data.

### Architecture

```python
QueryResult (from QueryExecutor)
    ↓
BedrockResultSummarizer.summarize(original_query, sql_query, result)
    ↓ Few-shot prompt with query context + results preview
AWS Bedrock Claude (Haiku)
    ↓ Natural language summary
SummaryResult (summary, success, model_id, execution_time_ms)
```

### Core Components

**BedrockResultSummarizer**:
- Generates 2-4 sentence summaries highlighting key insights
- Uses Claude 3 Haiku for cost-effective summarization (~$0.0003/summary)
- Temperature 0.3 for slight creativity while maintaining consistency
- Handles 6 diverse query patterns via few-shot learning

**SummaryResult** (dataclass):
```python
@dataclass(frozen=True)
class SummaryResult:
    summary: str                    # Natural language summary text
    success: bool                   # Whether summarization succeeded
    model_id: str                   # Bedrock model ID used
    execution_time_ms: float        # Time taken in milliseconds
    error_message: Optional[str]    # Error if failed
```

### Few-Shot Learning Strategy

**6 Examples Cover**:
1. **Empty results**: Constructive explanation when 0 rows returned
2. **Error logs with temporal filtering**: Pattern detection across services
3. **Aggregation queries**: Highlighting top contributors and percentages
4. **Text search (security patterns)**: Identifying brute force attempts
5. **Metadata filtering**: Correlating errors with specific conditions
6. **Large truncated results**: Advising users to narrow queries

**Example Summary**:
```
Query: "show me errors from yesterday"
Result: 47 ERROR logs

Summary:
Found 47 errors in the last 24 hours affecting multiple services. Top issues
include nginx connection timeouts (502 errors on /api/users), PostgreSQL
connection pool exhaustion (100 active connections), and docker container
failures (redis-cache). Most errors occurred between 8-11 PM, suggesting
increased traffic load.
```

### API Integration

The summarizer is integrated into the `/api/query` endpoint with the `include_summary` field:

**Request**:
```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "show nginx errors from yesterday",
    "include_summary": true
  }'
```

**Response with Summary**:
```json
{
  "success": true,
  "rows": [...],
  "row_count": 15,
  "execution_time_ms": 156.2,
  "summary": "Found 15 nginx errors concentrated around 4:30-4:45 PM...",
  "summary_success": true,
  "summary_execution_time_ms": 1523.4
}
```

**Key Features**:
- **Optional**: Controlled via `include_summary` field (defaults to `true`)
- **Non-breaking**: Summary failures don't affect query success
- **Transparent**: Includes execution time and success status
- **Graceful degradation**: Returns fallback summary on Bedrock errors

### Configuration

**Environment Variables** (`.env`):
```bash
AWS_REGION="us-east-1"
BEDROCK_MODEL_ID="anthropic.claude-haiku-4-5-20251001-v1:0"  # Direct model ID for summarization
AWS_ACCESS_KEY_ID="your_key"
AWS_SECRET_ACCESS_KEY="your_secret"
```

**Note**: Summarization uses direct model IDs (not inference profiles) because it targets a specific model version.

### Error Handling

**Fallback Strategies**:
1. **Failed queries**: Returns `"Query failed: {error_message}"`
2. **Bedrock API errors**: Returns `"Query returned N rows in Xms. Summary generation failed due to API error."`
3. **Malformed responses**: Returns `"Query returned N rows in Xms. Summary generation failed due to response parsing error."`

**Retry Logic**: Exponential backoff for `ThrottlingException` (3 retries: 1s, 2s, 4s delays)

### Performance

| Metric | Value |
|--------|-------|
| Average latency | ~1,500ms (Bedrock API call) |
| Token usage | ~3,000 input + ~150 output tokens |
| Cost per summary | ~$0.0003 (Haiku pricing) |
| Temperature | 0.3 (consistency + natural phrasing) |
| Max tokens | 512 (2-4 sentence summaries) |

**Optimization**:
- Previews only first 10 rows (token efficiency)
- Few-shot examples built once at initialization
- Async-compatible for FastAPI integration

### Testing

**API Tests** (`tests/test_api.py` - 15 tests total):
- Summary enabled (`include_summary: true`)
- Summary disabled (`include_summary: false`)
- Default behavior (defaults to `true`)
- Error handling (summary failures don't break query)

**Run Tests**:
```bash
source venv_llm_logs/bin/activate
python -m pytest tests/test_api.py -v --tb=short
```

**Manual Testing**:
```bash
# Test summarizer standalone
python -m src.shared.result_summarizer

# Test via API
uvicorn src.backend.app:app --reload
curl -X POST http://localhost:8000/api/query -H "Content-Type: application/json" \
  -d '{"query": "show errors from yesterday", "include_summary": true}'
```

### Usage Pattern

```python
from src.shared.result_summarizer import BedrockResultSummarizer
from src.shared.db_utils import QueryResult

# Initialize once (module level or app startup)
summarizer = BedrockResultSummarizer()

# Execute query (from QueryExecutor)
result = executor.execute_query("SELECT * FROM logs WHERE level='ERROR'")

# Generate summary
summary_result = summarizer.summarize(
    original_query="show errors from yesterday",
    sql_query="SELECT * FROM logs WHERE level='ERROR'...",
    result=result
)

print(f"Summary: {summary_result.summary}")
print(f"Success: {summary_result.success}")
print(f"Time: {summary_result.execution_time_ms:.1f}ms")
```

## Project Structure Rationale

See README.md for detailed explanation, but key points:
- `/src`: Separates application code from configuration
- `/src/shared`: Future home for reusable utilities (DB connection pooling, embedding cache)
- `/src/data_pipeline`: ETL logic
- `/src/backend`: Future FastAPI app for query engine
- `/data`: Log generation and storage
- `/config`: Database schemas and configuration files
- `/tests`: Unit tests (to be implemented)

## Current State

**Completed (Phase 1)**:
- ✅ PostgreSQL database with pgvector extension
- ✅ Logs table schema with VECTOR(384) column
- ✅ Three indexes (timestamp, level, embedding HNSW)
- ✅ Synthetic log generator with 9 services
- ✅ ETL pipeline with embedding generation
- ✅ Batch insertion with proper data serialization

**Completed (Phase 2 - Core NL→SQL)**:
- ✅ Semantic search implementation (`src/data_pipeline/search.py`)
- ✅ Natural language query to vector embedding
- ✅ PostgreSQL vector similarity search with pgvector
- ✅ **AWS Bedrock SQL Generator** (`src/shared/bedrock_client.py`)
- ✅ **NL→SQL translation with Claude 3 Haiku**
- ✅ **Three-layer SQL validation** (syntax, semantics, security)
- ✅ **Few-shot learning with 5 examples**
- ✅ **Error handling with exponential backoff**

**Completed (Phase 2b - Database Utilities)**:
- ✅ **DatabaseConnectionPool** (`src/shared/db_utils.py`) - Connection pooling for 5x performance improvement
- ✅ **QueryExecutor** - SQL execution with read-only enforcement, timeouts, and result limits
- ✅ **ResultFormatter** - Type conversion (datetime→ISO8601, JSONB→dict, numpy→list)
- ✅ **QueryResult dataclass** - Immutable structured query responses
- ✅ **Comprehensive test suite** - 70 tests (52 unit/integration + 18 end-to-end)

**Completed (Phase 2c - FastAPI Backend)**:
- ✅ **FastAPI REST API** (`src/backend/`) - HTTP API for natural language log queries
- ✅ **POST /api/query endpoint** - Integrates BedrockSQLGenerator + QueryExecutor + ResultSummarizer
- ✅ **GET /api/health endpoint** - Database health monitoring
- ✅ **Pydantic Models** - Request/response validation with field constraints
- ✅ **Error Handling** - HTTP status code mapping for 6 error types
- ✅ **CORS Middleware** - Configurable cross-origin resource sharing
- ✅ **Lifespan Management** - Connection pool and summarizer startup/shutdown
- ✅ **API Tests** - 15 integration tests with mocked Bedrock
- ✅ **Auto-generated API docs** - Swagger UI (/docs) and ReDoc (/redoc)

**Completed (Phase 2d - Result Summarization)**:
- ✅ **BedrockResultSummarizer** (`src/shared/result_summarizer.py`) - LLM-powered result summaries
- ✅ **SummaryResult dataclass** - Immutable summary responses with metadata
- ✅ **Few-shot learning** - 6 diverse examples covering query patterns
- ✅ **API integration** - Optional `include_summary` field in `/api/query`
- ✅ **Graceful degradation** - Summary failures don't break queries
- ✅ **Error handling** - Fallback summaries with retry logic

**Phase 2 Complete!** All natural language query features implemented and tested.

**Next Steps (Phase 3 - Intelligence Layer)**:
- Anomaly detection using statistical methods
- Automated runbook generation
- Pattern recognition for common issues
