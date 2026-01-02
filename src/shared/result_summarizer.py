"""
Result Summarization using AWS Bedrock Claude.

This module provides natural language summarization of SQL query results for the
log analytics system. It uses AWS Bedrock Claude to generate concise, insight-focused
summaries for system administrators and DevOps engineers.

Classes:
    BedrockResultSummarizer: Generates natural language summaries from QueryResult objects

Exceptions:
    BedrockError: Inherited from bedrock_client for API errors

Usage Example:
    >>> from src.shared.result_summarizer import BedrockResultSummarizer
    >>> from src.shared.db_utils import QueryResult
    >>>
    >>> summarizer = BedrockResultSummarizer()
    >>> summary = summarizer.generate_summary(
    ...     original_query="show me errors from yesterday",
    ...     sql_query="SELECT * FROM logs WHERE level = 'ERROR'...",
    ...     result=query_result_object
    ... )
    >>> print(summary)
    "Found 47 errors in the last 24 hours affecting multiple services..."
"""

import boto3
import json
import os
import time
from dotenv import load_dotenv
from botocore.exceptions import ClientError

from src.shared.db_utils import QueryResult
from src.shared.bedrock_client import BedrockError  # Reuse exception

# Load environment variables
load_dotenv()


class BedrockResultSummarizer:
    """
    Generates natural language summaries of SQL query results using Claude on AWS Bedrock.

    This class takes QueryResult objects and produces concise, insight-focused summaries
    suitable for system administrators and DevOps engineers. It uses few-shot learning
    to handle diverse query patterns including temporal filters, aggregations, text search,
    metadata filtering, and combined queries.

    Attributes:
        region (str): AWS region for Bedrock API
        model_id (str): Bedrock model ID (Claude Haiku by default)
        client: Boto3 Bedrock runtime client
        few_shot_examples (str): Pre-built few-shot examples for prompt

    Supported Query Types:
        - Temporal analysis: "show errors from yesterday"
        - Statistical grouping: "count errors by service"
        - Text search: "find login failures"
        - Metadata filtering: "nginx 500 errors"
        - Combined filters: "nginx errors in the last hour"

    Edge Cases Handled:
        - Empty results (0 rows)
        - Single row results
        - Large result sets (10,000+ rows, truncated)
        - Aggregation queries
        - Security-related patterns (failed logins, brute force)

    Example:
        >>> summarizer = BedrockResultSummarizer()
        >>> result = QueryResult(
        ...     success=True,
        ...     rows=[{"level": "ERROR", "source": "nginx", "message": "Connection timeout"}],
        ...     row_count=1,
        ...     column_names=["level", "source", "message"],
        ...     execution_time_ms=45.2
        ... )
        >>> summary = summarizer.generate_summary(
        ...     original_query="show nginx errors",
        ...     sql_query="SELECT * FROM logs WHERE source='nginx' AND level='ERROR'",
        ...     result=result
        ... )
    """

    # System prompt defining role and rules for summarization
    SYSTEM_PROMPT = """You are an expert log analytics assistant that summarizes query results for system administrators and DevOps engineers.

Rules:
1. Provide CONCISE summaries (2-4 sentences maximum)
2. Highlight KEY INSIGHTS: patterns, anomalies, trends, or critical issues
3. Use SPECIFIC NUMBERS from the data (counts, timestamps, percentages)
4. Mention TIME RANGES when relevant (e.g., "in the last hour", "between 2-3 PM")
5. For ERROR logs, PRIORITIZE severity and affected systems
6. For aggregations, highlight TOP contributors or outliers
7. If results are empty, explain possible reasons
8. Use clear, professional language - avoid jargon
9. NEVER fabricate data - only use information from the provided results"""

    def __init__(self, region: str = None, model_id: str = None):
        """
        Initialize BedrockResultSummarizer.

        Args:
            region: AWS region for Bedrock (defaults to AWS_REGION env var or us-east-1)
            model_id: Bedrock model ID (defaults to BEDROCK_MODEL_ID env var or Claude Haiku)

        Raises:
            BedrockError: If Bedrock client initialization fails
        """
        self.region = region or os.getenv("AWS_REGION", "us-east-1")
        self.model_id = model_id or os.getenv(
            "BEDROCK_MODEL_ID",
            "anthropic.claude-haiku-4-5-20251001-v1:0"
        )

        # Initialize Bedrock client
        try:
            self.client = boto3.client("bedrock-runtime", region_name=self.region)
        except Exception as e:
            raise BedrockError(f"Failed to initialize Bedrock client: {str(e)}")

        # Build few-shot examples once at initialization
        self.few_shot_examples = self._build_few_shot_examples()

    def _build_few_shot_examples(self) -> str:
        """
        Build few-shot examples for result summarization.

        Provides 6 diverse examples covering:
        - Empty results (0 rows)
        - Error logs with temporal filtering
        - Aggregation queries
        - Text search (security patterns)
        - Metadata filtering
        - Large truncated result sets

        Returns:
            Formatted string with example query contexts and summaries
        """
        return """
## Example Summaries

**Example 1: Empty Results**
Query Context:
- User Query: "show kernel panics from yesterday"
- Results: 0 rows returned
- Execution Time: 12.3ms

Summary:
No kernel panics were found in the last 24 hours. This indicates system stability. If you expected to see panic logs, verify that kernel logging is enabled or check a wider time range.

---

**Example 2: Error Logs (Temporal Filter)**
Query Context:
- User Query: "show me errors from yesterday"
- Results: 47 rows returned
- Execution Time: 156.2ms
- Columns: id, timestamp, level, source, message, metadata

Results Preview (first 3 of 47):
[
  {
    "id": 1523,
    "timestamp": "2025-12-28T23:45:12+01:00",
    "level": "ERROR",
    "source": "nginx",
    "message": "Connection timeout",
    "metadata": {"status_code": "502", "path": "/api/users"}
  },
  {
    "id": 1489,
    "timestamp": "2025-12-28T22:15:33+01:00",
    "level": "ERROR",
    "source": "postgresql",
    "message": "Connection pool exhausted",
    "metadata": {"database": "analytics", "active_connections": 100}
  },
  {
    "id": 1467,
    "timestamp": "2025-12-28T20:30:05+01:00",
    "level": "ERROR",
    "source": "docker",
    "message": "Container failed to start",
    "metadata": {"container_name": "redis-cache", "exit_code": 1}
  }
]

Summary:
Found 47 errors in the last 24 hours affecting multiple services. Top issues include nginx connection timeouts (502 errors on /api/users), PostgreSQL connection pool exhaustion (100 active connections), and docker container failures (redis-cache). Most errors occurred between 8-11 PM, suggesting increased traffic load.

---

**Example 3: Aggregation Query**
Query Context:
- User Query: "count errors by service"
- Results: 5 rows returned
- Execution Time: 78.4ms
- Columns: source, error_count

Results:
[
  {"source": "nginx", "error_count": 142},
  {"source": "postgresql", "error_count": 67},
  {"source": "docker", "error_count": 34},
  {"source": "sshd", "error_count": 18},
  {"source": "kernel", "error_count": 3}
]

Summary:
Analyzed 264 total errors across 5 services. Nginx accounts for 54% of all errors (142 occurrences), followed by PostgreSQL at 25% (67 errors). Docker, SSH, and kernel errors are comparatively minor. Nginx errors warrant immediate investigation as they likely indicate API or web server issues.

---

**Example 4: Text Search (Login Failures)**
Query Context:
- User Query: "find login failures"
- Results: 23 rows returned
- Execution Time: 203.1ms
- Columns: id, timestamp, level, source, message, metadata

Results Preview (first 3 of 23):
[
  {
    "id": 892,
    "timestamp": "2025-12-28T14:22:18+01:00",
    "level": "WARN",
    "source": "sshd",
    "message": "User login failed",
    "metadata": {"user": "admin", "ip": "192.168.1.50", "auth_method": "password"}
  },
  {
    "id": 887,
    "timestamp": "2025-12-28T14:21:45+01:00",
    "level": "WARN",
    "source": "sshd",
    "message": "User login failed",
    "metadata": {"user": "admin", "ip": "192.168.1.50", "auth_method": "password"}
  },
  {
    "id": 834,
    "timestamp": "2025-12-28T09:15:32+01:00",
    "level": "ERROR",
    "source": "auth",
    "message": "Authentication failed - invalid credentials",
    "metadata": {"user": "root", "ip": "203.0.113.42"}
  }
]

Summary:
Detected 23 login failures across SSH and authentication services. Notable pattern: Multiple failed login attempts for user "admin" from IP 192.168.1.50 at 2:21-2:22 PM (potential brute force). Also found failed root authentication from external IP 203.0.113.42 at 9:15 AM. Recommend reviewing access controls and enabling fail2ban.

---

**Example 5: Metadata Filtering (nginx 500 errors)**
Query Context:
- User Query: "find nginx 500 errors"
- Results: 15 rows returned
- Execution Time: 95.7ms
- Columns: id, timestamp, level, source, message, metadata

Results Preview (first 3 of 15):
[
  {
    "id": 2103,
    "timestamp": "2025-12-28T16:45:22+01:00",
    "level": "ERROR",
    "source": "nginx",
    "message": "Internal server error",
    "metadata": {"status_code": "500", "method": "POST", "path": "/api/orders", "response_time": "5234ms"}
  },
  {
    "id": 2098,
    "timestamp": "2025-12-28T16:42:15+01:00",
    "level": "ERROR",
    "source": "nginx",
    "message": "Internal server error",
    "metadata": {"status_code": "500", "method": "GET", "path": "/api/orders", "response_time": "4891ms"}
  },
  {
    "id": 2087,
    "timestamp": "2025-12-28T16:38:04+01:00",
    "level": "ERROR",
    "source": "nginx",
    "message": "Internal server error",
    "metadata": {"status_code": "500", "method": "POST", "path": "/api/payment", "response_time": "6120ms"}
  }
]

Summary:
Found 15 nginx 500 errors concentrated around 4:30-4:45 PM. All errors show abnormally high response times (4.8-6.1 seconds) on /api/orders and /api/payment endpoints. The timing and affected endpoints suggest a backend service outage or database performance issue during that window.

---

**Example 6: Large Result Set (Truncated)**
Query Context:
- User Query: "show all INFO logs from last week"
- Results: 10,000 rows returned (TRUNCATED)
- Execution Time: 1,456.8ms
- Columns: id, timestamp, level, source, message, metadata

Results Preview (first 5 of 10,000):
[
  {"id": 15234, "timestamp": "2025-12-28T23:59:58+01:00", "level": "INFO", "source": "systemd", "message": "Service started"},
  {"id": 15233, "timestamp": "2025-12-28T23:59:45+01:00", "level": "INFO", "source": "nginx", "message": "Request processed"},
  {"id": 15232, "timestamp": "2025-12-28T23:59:32+01:00", "level": "INFO", "source": "cron", "message": "Job completed"},
  {"id": 15231, "timestamp": "2025-12-28T23:59:18+01:00", "level": "INFO", "source": "docker", "message": "Container health check passed"},
  {"id": 15230, "timestamp": "2025-12-28T23:59:05+01:00", "level": "INFO", "source": "postgresql", "message": "Checkpoint completed"}
]

Summary:
Retrieved maximum 10,000 INFO logs from the past week (results truncated). Logs show normal system operations across all services including systemd, nginx, cron, docker, and postgresql. To analyze specific INFO events, narrow your query with time ranges, service filters, or text search keywords.
"""

    def _build_summary_context(
        self,
        original_query: str,
        sql_query: str,
        result: QueryResult
    ) -> str:
        """
        Build context for summarization prompt.

        Creates a structured context including query metadata and result preview.
        Limits results to first 10 rows to stay under token budget (~3K tokens).

        Args:
            original_query: User's natural language query
            sql_query: Generated SQL query
            result: QueryResult object from QueryExecutor

        Returns:
            Formatted context string with metadata and JSON results

        Example Output:
            ## Query Context
            - **User Query**: "show errors from yesterday"
            - **SQL Query**: SELECT * FROM logs...
            - **Results**: 47 rows returned
            ...
        """
        # Preview first 10 rows only (token efficiency)
        rows_preview = result.rows[:10] if len(result.rows) > 10 else result.rows

        # Build metadata section
        context = f"""
## Query Context
- **User Query**: "{original_query}"
- **SQL Query**: {sql_query}
- **Results**: {result.row_count} rows returned
- **Execution Time**: {result.execution_time_ms:.1f}ms
- **Truncated**: {result.truncated}
- **Columns**: {', '.join(result.column_names)}

## Query Results (Preview)
```json
{json.dumps(rows_preview, indent=2, default=str)}
```
"""

        # Add truncation note if applicable
        if result.truncated:
            context += f"\n**Note**: Results truncated. Showing first 10 of {result.row_count} total rows.\n"

        return context

    def _retry_with_backoff(self, func, max_retries: int = 3, base_delay: float = 1.0):
        """
        Retry function with exponential backoff for throttling errors.

        Implements exponential backoff strategy for AWS Bedrock throttling.
        Retries up to max_retries times with delays: 1s, 2s, 4s.

        Args:
            func: Function to execute (should return Bedrock response)
            max_retries: Maximum number of retry attempts (default: 3)
            base_delay: Base delay in seconds for exponential backoff (default: 1.0)

        Returns:
            Result from successful function execution

        Raises:
            BedrockError: If max retries exceeded or non-throttling error occurs
        """
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

    def generate_summary(
        self,
        original_query: str,
        sql_query: str,
        result: QueryResult
    ) -> str:
        """
        Generate natural language summary of query results.

        Main API method that orchestrates the summarization process:
        1. Validates input QueryResult
        2. Builds context from query metadata and results
        3. Invokes AWS Bedrock Claude with few-shot prompt
        4. Extracts and returns summary text

        Uses temperature 0.3 for slight creativity in phrasing while maintaining
        consistency. Targets 2-4 sentence summaries with specific numbers and insights.

        Args:
            original_query: User's natural language query (e.g., "show errors from yesterday")
            sql_query: Generated SQL query
            result: QueryResult object from QueryExecutor containing rows, metadata, etc.

        Returns:
            Natural language summary (2-4 sentences) highlighting key insights

        Raises:
            BedrockError: If Bedrock API fails after retries
            ValueError: If inputs are invalid

        Examples:
            >>> summarizer = BedrockResultSummarizer()
            >>> result = QueryResult(success=True, rows=[...], row_count=47, ...)
            >>> summary = summarizer.generate_summary(
            ...     "show errors from yesterday",
            ...     "SELECT * FROM logs WHERE level='ERROR'...",
            ...     result
            ... )
            >>> print(summary)
            "Found 47 errors in the last 24 hours affecting multiple services..."

        Edge Cases:
            - Empty results (0 rows): Returns constructive explanation
            - Single row: Provides detailed summary of that log entry
            - Large results (truncated): Summarizes preview + advises narrowing query
            - Failed query: Returns error message from QueryResult
        """
        # Validate inputs
        if not original_query or not original_query.strip():
            raise ValueError("Original query cannot be empty")

        if not sql_query or not sql_query.strip():
            raise ValueError("SQL query cannot be empty")

        if result is None:
            raise ValueError("QueryResult cannot be None")

        # Handle failed queries
        if not result.success:
            error_msg = result.error_message or "Unknown error"
            return f"Query failed: {error_msg}"

        # Build prompt
        context = self._build_summary_context(original_query, sql_query, result)
        user_prompt = f"{self.few_shot_examples}\n\n{context}\n\nGenerate a concise summary of these query results:"

        # Build Bedrock request (Messages API format)
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",  # Required for Messages API
            "max_tokens": 512,  # Summaries are short (2-4 sentences)
            "temperature": 0.3,  # Slight creativity for natural phrasing
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
        except BedrockError:
            # Re-raise BedrockError as-is
            raise
        except Exception as e:
            # Fallback summary on unexpected errors
            return (
                f"Query returned {result.row_count} rows in {result.execution_time_ms:.1f}ms. "
                f"Summary generation unavailable due to error: {str(e)}"
            )

        # Extract summary from response
        try:
            summary = response_body['content'][0]['text'].strip()
        except (KeyError, IndexError) as e:
            raise BedrockError(f"Unexpected response format from Bedrock: {str(e)}")

        return summary


# Testing entry point
if __name__ == "__main__":
    """
    Test BedrockResultSummarizer with mock QueryResult objects.

    Run from project root:
        python -m src.shared.result_summarizer

    Requires:
        - AWS credentials configured (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
        - Bedrock model access enabled
        - .env file with AWS_REGION and BEDROCK_MODEL_ID
    """
    from src.shared.db_utils import QueryResult

    print("Testing BedrockResultSummarizer...\n")
    print("=" * 80)

    # Test Case 1: Error logs with temporal filter
    print("\nTest Case 1: Error Logs (Temporal Filter)")
    print("-" * 80)

    test_result_1 = QueryResult(
        success=True,
        rows=[
            {
                "id": 1523,
                "timestamp": "2025-12-28T23:45:12+01:00",
                "level": "ERROR",
                "source": "nginx",
                "message": "Connection timeout",
                "metadata": {"status_code": "502", "path": "/api/users"}
            },
            {
                "id": 1489,
                "timestamp": "2025-12-28T22:15:33+01:00",
                "level": "ERROR",
                "source": "postgresql",
                "message": "Connection pool exhausted",
                "metadata": {"database": "analytics", "active_connections": 100}
            }
        ],
        row_count=47,
        column_names=["id", "timestamp", "level", "source", "message", "metadata"],
        execution_time_ms=156.2,
        truncated=False
    )

    # Test Case 2: Empty results
    print("\nTest Case 2: Empty Results")
    print("-" * 80)

    test_result_2 = QueryResult(
        success=True,
        rows=[],
        row_count=0,
        column_names=["id", "timestamp", "level", "source", "message"],
        execution_time_ms=12.3,
        truncated=False
    )

    # Test Case 3: Aggregation query
    print("\nTest Case 3: Aggregation Query")
    print("-" * 80)

    test_result_3 = QueryResult(
        success=True,
        rows=[
            {"source": "nginx", "error_count": 142},
            {"source": "postgresql", "error_count": 67},
            {"source": "docker", "error_count": 34},
            {"source": "sshd", "error_count": 18},
            {"source": "kernel", "error_count": 3}
        ],
        row_count=5,
        column_names=["source", "error_count"],
        execution_time_ms=78.4,
        truncated=False
    )

    try:
        summarizer = BedrockResultSummarizer()

        # Test Case 1
        print("Query: 'show me errors from yesterday'")
        summary_1 = summarizer.generate_summary(
            original_query="show me errors from yesterday",
            sql_query="SELECT * FROM logs WHERE level = 'ERROR' AND timestamp >= NOW() - INTERVAL '1 day'",
            result=test_result_1
        )
        print(f"Summary: {summary_1}\n")

        # Test Case 2
        print("Query: 'show kernel panics from yesterday'")
        summary_2 = summarizer.generate_summary(
            original_query="show kernel panics from yesterday",
            sql_query="SELECT * FROM logs WHERE source = 'kernel' AND message ILIKE '%panic%'",
            result=test_result_2
        )
        print(f"Summary: {summary_2}\n")

        # Test Case 3
        print("Query: 'count errors by service'")
        summary_3 = summarizer.generate_summary(
            original_query="count errors by service",
            sql_query="SELECT source, COUNT(*) as error_count FROM logs WHERE level = 'ERROR' GROUP BY source",
            result=test_result_3
        )
        print(f"Summary: {summary_3}\n")

        print("=" * 80)
        print("All tests completed successfully!")

    except BedrockError as e:
        print(f"\nBedrock Error: {e}")
        print("\nMake sure you have:")
        print("1. AWS credentials configured (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)")
        print("2. Bedrock model access enabled in AWS console")
        print("3. .env file with AWS_REGION and BEDROCK_MODEL_ID")
    except Exception as e:
        print(f"\nUnexpected Error: {e}")
        import traceback
        traceback.print_exc()
