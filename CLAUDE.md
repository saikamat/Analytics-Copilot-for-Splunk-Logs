# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## System Overview

This is an **AI-powered log analytics system** - a Splunk alternative that uses:
- **PostgreSQL with pgvector** for storage and semantic search
- **Sentence-transformers** to convert log messages into 384-dimensional embeddings
- **LLMs** (future phases) to translate natural language queries into SQL
- **Python** for ETL pipeline and backend services

The project follows a **4-phase development approach**:
1. **Phase 1** (Complete): Foundation & Data Pipeline
2. **Phase 2** (In Progress): LLM Integration for natural language queries
3. **Phase 3**: Intelligence Layer (anomaly detection, runbooks)
4. **Phase 4**: Web App & AWS Deployment

## Architecture

### Data Flow
```
Log Generation → ETL Pipeline → PostgreSQL → Semantic Search
```

1. **Log Generation** (`data/log_generator.py`): Creates synthetic system logs from 9 services (sshd, nginx, kernel, etc.) with realistic metadata
2. **ETL Pipeline** (`src/data_pipeline/etl.py`):
   - Generates embeddings using sentence-transformers (all-MiniLM-L6-v2)
   - Converts embeddings to lists for PostgreSQL compatibility
   - Serializes metadata dicts to JSON
   - Batch inserts using `psycopg2.extras.execute_batch`
3. **Storage**: PostgreSQL `logs` table with VECTOR(384) column
4. **Search**: Vector similarity using pgvector's cosine distance operator (`<=>`)

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

**Completed (Phase 2 - Partial)**:
- ✅ Semantic search implementation (`src/data_pipeline/search.py`)
- ✅ Natural language query to vector embedding
- ✅ PostgreSQL vector similarity search with pgvector

**Next Steps (Phase 2 - Remaining)**:
- LLM integration with AWS Bedrock for NL→SQL translation
- Query validator for SQL safety
- FastAPI backend with query endpoints
- Result summarization with LLM
