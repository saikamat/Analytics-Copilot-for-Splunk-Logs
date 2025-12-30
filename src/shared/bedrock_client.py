import boto3
import json
import os
import time
import sqlparse
from dotenv import load_dotenv
from botocore.exceptions import ClientError

load_dotenv()


# Custom Exceptions
class BedrockError(Exception):
    """Base exception for Bedrock-related errors."""
    pass


class ValidationError(Exception):
    """Exception for SQL validation failures."""
    pass


# Translate NLP queries to SQL using Claude on AWS Bedrock
class BedrockSQLGenerator:
    # System prompt for SQL generation
    SYSTEM_PROMPT = """You are an expert PostgreSQL query generator for a log analytics system.

Rules:
1. ONLY generate SELECT statements (no INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE)
2. ALWAYS use indexed columns (timestamp, level, embedding) in WHERE clauses when possible
3. Return ONLY the SQL query, no explanations or markdown code blocks
4. Use PostgreSQL-specific functions (INTERVAL, TIMESTAMPTZ, JSONB operators)
5. Handle ambiguous time references (e.g., "yesterday" = NOW() - INTERVAL '1 day')
6. Limit results to 100 rows maximum unless specified otherwise
7. For metadata queries, use JSONB operators: metadata->>'field_name'
8. Always include ORDER BY timestamp DESC for temporal queries"""

    def __init__(self, region: str = None, model_id: str = None):
        self.region = region or os.getenv("AWS_REGION", "us-east-1")
        self.model_id = model_id or os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-haiku-4-5-20251001-v1:0")

        # Initialize Bedrock client
        self.client = boto3.client("bedrock-runtime", region_name=self.region)

        # Build schema context once at initialization
        self.schema_context = self._build_schema_context()
        self.few_shot_examples = self._build_few_shot_examples()

    def _build_schema_context(self) -> str:
        """Build comprehensive PostgreSQL schema description."""
        return """
## Database Schema

**Table: logs**
```sql
CREATE TABLE logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    level VARCHAR(20),              -- Values: 'INFO', 'WARN', 'ERROR'
    source VARCHAR(255),            -- Service name: 'sshd', 'nginx', 'kernel', 'systemd', 'postgresql', 'docker', 'cron', 'auth', 'networking'
    message TEXT NOT NULL,          -- Log message content
    metadata JSONB,                 -- Service-specific fields (flexible schema)
    embedding VECTOR(384)           -- Semantic search vector (use for similarity queries)
);
```

**Indexes (optimized for these queries):**
- `idx_logs_timestamp`: B-tree index with DESC ordering (fast time-range queries)
- `idx_logs_level`: B-tree index (fast filtering by severity)
- `idx_logs_embedding`: HNSW index with vector_cosine_ops (fast semantic search)

**Common metadata fields by service:**
- sshd: `user`, `ip`, `auth_method`
- nginx: `status_code`, `method`, `path`, `response_time`
- kernel: `process`, `pid`
- postgresql: `database`, `query_type`, `duration_ms`
- docker: `container_id`, `container_name`, `action`

**PostgreSQL-specific operators:**
- Temporal: `NOW()`, `INTERVAL '1 day'`, `INTERVAL '1 hour'`
- JSONB: `metadata->>'field_name'` (extract text field)
- Text search: `message ILIKE '%pattern%'` (case-insensitive)
- Vector similarity: `embedding <=> query_vector` (cosine distance, lower = more similar)
"""

    def _build_few_shot_examples(self) -> str:
        """Build few-shot examples for query translation."""
        return """
## Example Queries

**Example 1: Temporal filtering**
Natural language: "show me errors from yesterday"
SQL: SELECT id, timestamp, level, source, message, metadata FROM logs WHERE level = 'ERROR' AND timestamp >= NOW() - INTERVAL '1 day' ORDER BY timestamp DESC LIMIT 100;

**Example 2: Metadata access**
Natural language: "find nginx 500 errors"
SQL: SELECT id, timestamp, level, source, message, metadata FROM logs WHERE source = 'nginx' AND metadata->>'status_code' = '500' ORDER BY timestamp DESC LIMIT 100;

**Example 3: Aggregation**
Natural language: "count errors by service"
SQL: SELECT source, COUNT(*) as error_count FROM logs WHERE level = 'ERROR' GROUP BY source ORDER BY error_count DESC;

**Example 4: Text search**
Natural language: "find login failures"
SQL: SELECT id, timestamp, level, source, message, metadata FROM logs WHERE message ILIKE '%login%' AND message ILIKE '%fail%' ORDER BY timestamp DESC LIMIT 100;

**Example 5: Combined filters with time range**
Natural language: "show nginx errors in the last hour"
SQL: SELECT id, timestamp, level, source, message, metadata FROM logs WHERE source = 'nginx' AND level = 'ERROR' AND timestamp >= NOW() - INTERVAL '1 hour' ORDER BY timestamp DESC LIMIT 100;
"""

    def _retry_with_backoff(self, func, max_retries: int = 3, base_delay: float = 1.0):
        """Retry function with exponential backoff for throttling errors."""
        for attempt in range(max_retries):
            try:
                return func()
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', '')
                if error_code == 'ThrottlingException':
                    if attempt == max_retries - 1:
                        raise BedrockError(f"Max retries ({max_retries}) exceeded due to throttling")
                    delay = base_delay * (2 ** attempt)
                    time.sleep(delay)
                else:
                    raise BedrockError(f"Bedrock API error: {error_code} - {str(e)}")
            except Exception as e:
                raise BedrockError(f"Unexpected error during Bedrock invocation: {str(e)}")

    def validate_sql(self, sql: str) -> bool:
        """
        Three-layer SQL validation for security.
        Raises ValidationError if validation fails.
        """
        if not sql or not sql.strip():
            raise ValidationError("Empty SQL query")

        # Layer 1: Syntax validation
        try:
            parsed = sqlparse.parse(sql)
            if not parsed:
                raise ValidationError("Could not parse SQL query")
        except Exception as e:
            raise ValidationError(f"SQL syntax error: {str(e)}")

        # Layer 2: Semantic validation
        sql_upper = sql.upper()

        # Only SELECT allowed
        sql_stripped = sql_upper.strip()
        if not sql_stripped.startswith('SELECT'):
            raise ValidationError("Only SELECT queries are allowed")

        # Block dangerous operations
        dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE', 'CREATE', 'EXEC']
        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                raise ValidationError(f"Forbidden SQL keyword: {keyword}")

        # Only 'logs' table allowed
        if 'FROM LOGS' not in sql_upper and 'FROM logs' not in sql:
            raise ValidationError("Only queries on the 'logs' table are allowed")

        # Layer 3: Security patterns
        suspicious_patterns = ['--', '/*', 'xp_', 'sp_', 'UNION']
        for pattern in suspicious_patterns:
            if pattern in sql_upper:
                raise ValidationError(f"Suspicious SQL pattern detected: {pattern}")

        return True

    def generate_sql(self, natural_query: str) -> str:
        """
        Main API: Translate natural language query to SQL.

        Args:
            natural_query: Natural language query (e.g., "show errors from yesterday")

        Returns:
            Validated SQL query string

        Raises:
            BedrockError: If Bedrock API fails
            ValidationError: If generated SQL fails validation
        """
        if not natural_query or not natural_query.strip():
            raise ValueError("Natural query cannot be empty")

        # Build prompt
        user_prompt = f"{self.schema_context}\n\n{self.few_shot_examples}\n\nNow generate SQL for this query:\nNatural language: \"{natural_query}\"\nSQL:"

        # Build Bedrock request
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "temperature": 0.0,  # Deterministic for SQL reliability
            "messages": [
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            "system": self.SYSTEM_PROMPT
        }

        # Invoke Bedrock with retry logic
        def invoke():
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body)
            )
            return json.loads(response['body'].read())

        try:
            response_body = self._retry_with_backoff(invoke)
        except Exception as e:
            raise BedrockError(f"Failed to generate SQL: {str(e)}")

        # Extract SQL from response
        try:
            sql = response_body['content'][0]['text'].strip()
        except (KeyError, IndexError) as e:
            raise BedrockError(f"Unexpected response format from Bedrock: {str(e)}")

        # Clean output (remove markdown code blocks if present)
        if sql.startswith('```sql'):
            sql = sql.replace('```sql', '').replace('```', '').strip()
        elif sql.startswith('```'):
            sql = sql.replace('```', '').strip()

        # Validate generated SQL
        self.validate_sql(sql)

        return sql


# Testing entry point
if __name__ == "__main__":
    print("Testing BedrockSQLGenerator...\n")

    try:
        generator = BedrockSQLGenerator()

        test_queries = [
            "show errors from yesterday",
            "find login failures in the last hour",
            "count errors by service",
            "what were nginx issues today",
            "show postgresql slow queries"
        ]

        for query in test_queries:
            try:
                print(f"Natural Query: {query}")
                sql = generator.generate_sql(query)
                print(f"Generated SQL: {sql}")
                print("-" * 80)
            except ValidationError as e:
                print(f"Validation Error: {e}")
                print("-" * 80)
            except BedrockError as e:
                print(f"Bedrock Error: {e}")
                print("-" * 80)
            except Exception as e:
                print(f"Unexpected Error: {e}")
                print("-" * 80)

    except Exception as e:
        print(f"Failed to initialize BedrockSQLGenerator: {e}")
        print("\nMake sure you have:")
        print("1. AWS credentials configured (run 'aws configure' or set env vars)")
        print("2. Bedrock model access enabled in AWS console")
        print("3. .env file with AWS_REGION and BEDROCK_MODEL_ID")
