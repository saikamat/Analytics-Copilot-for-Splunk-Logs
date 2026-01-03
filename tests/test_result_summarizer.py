"""
Unit tests for BedrockResultSummarizer.

Tests summarization logic, error handling, edge cases, and initialization
with mocked Bedrock API to avoid AWS calls.
"""

import pytest
from unittest.mock import patch, MagicMock
from botocore.exceptions import ClientError

from src.shared.result_summarizer import BedrockResultSummarizer, SummaryResult
from src.shared.bedrock_client import BedrockError
from src.shared.db_utils import QueryResult


class TestSummarizationLogic:
    """Test summarization logic with various result types."""

    def test_summarize_empty_results(self):
        """Test summarization with 0 rows returned."""
        summarizer = BedrockResultSummarizer()

        result = QueryResult(
            success=True,
            rows=[],
            row_count=0,
            column_names=["id", "timestamp", "level", "source", "message"],
            execution_time_ms=12.3,
            truncated=False
        )

        mock_response = {
            'content': [{
                'text': 'No results found. This indicates the query matched no log entries.'
            }]
        }

        with patch.object(summarizer.client, 'invoke_model', return_value={'body': MagicMock(read=lambda: str(mock_response).encode())}):
            with patch('json.loads', return_value=mock_response):
                summary = summarizer.summarize(
                    original_query="show kernel panics",
                    sql_query="SELECT * FROM logs WHERE message ILIKE '%panic%'",
                    result=result
                )

        assert summary.success is True
        assert "No results" in summary.summary or "no log entries" in summary.summary.lower()
        assert summary.execution_time_ms > 0

    def test_summarize_single_row(self):
        """Test summarization with single row result."""
        summarizer = BedrockResultSummarizer()

        result = QueryResult(
            success=True,
            rows=[{
                "id": 42,
                "timestamp": "2025-12-28T14:30:00+01:00",
                "level": "ERROR",
                "source": "nginx",
                "message": "Connection timeout"
            }],
            row_count=1,
            column_names=["id", "timestamp", "level", "source", "message"],
            execution_time_ms=23.4,
            truncated=False
        )

        mock_response = {
            'content': [{
                'text': 'Found 1 nginx error at 2:30 PM: Connection timeout. Investigate nginx server health.'
            }]
        }

        with patch.object(summarizer.client, 'invoke_model', return_value={'body': MagicMock(read=lambda: str(mock_response).encode())}):
            with patch('json.loads', return_value=mock_response):
                summary = summarizer.summarize(
                    original_query="show nginx errors",
                    sql_query="SELECT * FROM logs WHERE source='nginx' AND level='ERROR'",
                    result=result
                )

        assert summary.success is True
        assert "1" in summary.summary or "single" in summary.summary.lower()
        assert summary.model_id == summarizer.model_id

    def test_summarize_typical_results(self):
        """Test summarization with typical result set (10-100 rows)."""
        summarizer = BedrockResultSummarizer()

        # Create 47 rows
        rows = []
        for i in range(47):
            rows.append({
                "id": 1000 + i,
                "timestamp": f"2025-12-28T{10 + i % 14}:00:00+01:00",
                "level": "ERROR",
                "source": ["nginx", "postgresql", "docker"][i % 3],
                "message": f"Error message {i}"
            })

        result = QueryResult(
            success=True,
            rows=rows,
            row_count=47,
            column_names=["id", "timestamp", "level", "source", "message"],
            execution_time_ms=156.2,
            truncated=False
        )

        mock_response = {
            'content': [{
                'text': 'Found 47 errors across nginx, postgresql, and docker services. Most errors occurred between 10 AM and midnight.'
            }]
        }

        with patch.object(summarizer.client, 'invoke_model', return_value={'body': MagicMock(read=lambda: str(mock_response).encode())}):
            with patch('json.loads', return_value=mock_response):
                summary = summarizer.summarize(
                    original_query="show errors from yesterday",
                    sql_query="SELECT * FROM logs WHERE level='ERROR'",
                    result=result
                )

        assert summary.success is True
        assert "47" in summary.summary
        assert summary.execution_time_ms > 0

    def test_summarize_large_results(self):
        """Test summarization with large result set (10,000 rows - sampling)."""
        summarizer = BedrockResultSummarizer()

        # Create 10,000 rows
        rows = [{"id": i, "level": "INFO", "source": "systemd"} for i in range(10000)]

        result = QueryResult(
            success=True,
            rows=rows,
            row_count=10000,
            column_names=["id", "level", "source"],
            execution_time_ms=1456.8,
            truncated=True
        )

        mock_response = {
            'content': [{
                'text': 'Retrieved maximum 10,000 INFO logs (results truncated). To analyze specific events, narrow your query with filters.'
            }]
        }

        with patch.object(summarizer.client, 'invoke_model', return_value={'body': MagicMock(read=lambda: str(mock_response).encode())}):
            with patch('json.loads', return_value=mock_response):
                summary = summarizer.summarize(
                    original_query="show all INFO logs",
                    sql_query="SELECT * FROM logs WHERE level='INFO'",
                    result=result
                )

        assert summary.success is True
        assert "10,000" in summary.summary or "10000" in summary.summary
        assert "truncated" in summary.summary.lower()

    def test_summarize_aggregation_results(self):
        """Test summarization with aggregation query results."""
        summarizer = BedrockResultSummarizer()

        result = QueryResult(
            success=True,
            rows=[
                {"source": "nginx", "error_count": 142},
                {"source": "postgresql", "error_count": 67},
                {"source": "docker", "error_count": 34}
            ],
            row_count=3,
            column_names=["source", "error_count"],
            execution_time_ms=78.4,
            truncated=False
        )

        mock_response = {
            'content': [{
                'text': 'Analyzed 243 total errors. Nginx accounts for 58% (142 errors), PostgreSQL 28% (67 errors), Docker 14% (34 errors).'
            }]
        }

        with patch.object(summarizer.client, 'invoke_model', return_value={'body': MagicMock(read=lambda: str(mock_response).encode())}):
            with patch('json.loads', return_value=mock_response):
                summary = summarizer.summarize(
                    original_query="count errors by service",
                    sql_query="SELECT source, COUNT(*) as error_count FROM logs WHERE level='ERROR' GROUP BY source",
                    result=result
                )

        assert summary.success is True
        assert "142" in summary.summary or "nginx" in summary.summary.lower()


class TestErrorHandling:
    """Test error handling with mocked Bedrock failures."""

    def test_bedrock_api_error_fallback(self):
        """Test fallback summary when Bedrock API errors occur."""
        summarizer = BedrockResultSummarizer()

        result = QueryResult(
            success=True,
            rows=[{"id": 1, "level": "ERROR"}],
            row_count=1,
            column_names=["id", "level"],
            execution_time_ms=23.4,
            truncated=False
        )

        # Mock Bedrock API error
        with patch.object(summarizer.client, 'invoke_model', side_effect=ClientError(
            {'Error': {'Code': 'ServiceUnavailable', 'Message': 'Service unavailable'}},
            'invoke_model'
        )):
            summary = summarizer.summarize(
                original_query="test query",
                sql_query="SELECT * FROM logs",
                result=result
            )

        assert summary.success is False
        assert "Summary generation failed" in summary.summary or "API error" in summary.summary
        assert summary.error_message is not None
        assert "ServiceUnavailable" in summary.error_message

    def test_bedrock_throttling_retry(self):
        """Test retry logic for ThrottlingException."""
        summarizer = BedrockResultSummarizer()

        result = QueryResult(
            success=True,
            rows=[{"id": 1}],
            row_count=1,
            column_names=["id"],
            execution_time_ms=10.0,
            truncated=False
        )

        # Mock throttling on first call, success on second
        mock_response = {'content': [{'text': 'Success after retry'}]}

        call_count = 0
        def mock_invoke(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ClientError(
                    {'Error': {'Code': 'ThrottlingException', 'Message': 'Rate exceeded'}},
                    'invoke_model'
                )
            return {'body': MagicMock(read=lambda: str(mock_response).encode())}

        with patch.object(summarizer.client, 'invoke_model', side_effect=mock_invoke):
            with patch('json.loads', return_value=mock_response):
                with patch('time.sleep'):  # Skip actual sleep
                    summary = summarizer.summarize(
                        original_query="test",
                        sql_query="SELECT * FROM logs",
                        result=result
                    )

        assert call_count == 2  # Should have retried once
        assert summary.success is True
        assert "Success after retry" in summary.summary

    def test_max_retries_exceeded(self):
        """Test failure after max retries exceeded."""
        summarizer = BedrockResultSummarizer()

        result = QueryResult(
            success=True,
            rows=[{"id": 1}],
            row_count=1,
            column_names=["id"],
            execution_time_ms=10.0,
            truncated=False
        )

        # Always throttle
        with patch.object(summarizer.client, 'invoke_model', side_effect=ClientError(
            {'Error': {'Code': 'ThrottlingException', 'Message': 'Rate exceeded'}},
            'invoke_model'
        )):
            with patch('time.sleep'):  # Skip actual sleep
                summary = summarizer.summarize(
                    original_query="test",
                    sql_query="SELECT * FROM logs",
                    result=result
                )

        assert summary.success is False
        assert "failed" in summary.summary.lower() or "API error" in summary.summary
        assert "Max retries" in summary.error_message or "throttling" in summary.error_message.lower()

    def test_malformed_response(self):
        """Test handling of malformed Bedrock response."""
        summarizer = BedrockResultSummarizer()

        result = QueryResult(
            success=True,
            rows=[{"id": 1}],
            row_count=1,
            column_names=["id"],
            execution_time_ms=10.0,
            truncated=False
        )

        # Mock malformed response (missing 'content' key)
        malformed_response = {'unexpected_key': 'value'}

        with patch.object(summarizer.client, 'invoke_model', return_value={'body': MagicMock(read=lambda: str(malformed_response).encode())}):
            with patch('json.loads', return_value=malformed_response):
                summary = summarizer.summarize(
                    original_query="test",
                    sql_query="SELECT * FROM logs",
                    result=result
                )

        assert summary.success is False
        assert "failed" in summary.summary.lower() or "parsing error" in summary.summary.lower()
        assert summary.error_message is not None

    def test_failed_query_result(self):
        """Test summarization with failed QueryResult."""
        summarizer = BedrockResultSummarizer()

        result = QueryResult(
            success=False,
            rows=[],
            row_count=0,
            column_names=[],
            execution_time_ms=0.0,
            truncated=False,
            error_message="SQL syntax error"
        )

        summary = summarizer.summarize(
            original_query="invalid query",
            sql_query="INVALID SQL",
            result=result
        )

        assert summary.success is False
        assert "Query failed" in summary.summary
        assert "SQL syntax error" in summary.summary
        assert summary.error_message == "SQL syntax error"


class TestEdgeCases:
    """Test edge cases and input validation."""

    def test_truncated_results(self):
        """Test handling of truncated result flag."""
        summarizer = BedrockResultSummarizer()

        result = QueryResult(
            success=True,
            rows=[{"id": i} for i in range(100)],
            row_count=100,
            column_names=["id"],
            execution_time_ms=50.0,
            truncated=True  # Truncated flag
        )

        mock_response = {'content': [{'text': 'Results truncated at 100 rows'}]}

        with patch.object(summarizer.client, 'invoke_model', return_value={'body': MagicMock(read=lambda: str(mock_response).encode())}):
            with patch('json.loads', return_value=mock_response):
                summary = summarizer.summarize(
                    original_query="show all logs",
                    sql_query="SELECT * FROM logs",
                    result=result
                )

        assert summary.success is True
        # Context building should include truncation note
        # (verified via the summary being generated successfully)

    def test_null_values_in_rows(self):
        """Test handling of NULL values in result rows."""
        summarizer = BedrockResultSummarizer()

        result = QueryResult(
            success=True,
            rows=[
                {"id": 1, "message": "Valid message", "metadata": None},
                {"id": 2, "message": None, "metadata": {"key": "value"}}
            ],
            row_count=2,
            column_names=["id", "message", "metadata"],
            execution_time_ms=15.0,
            truncated=False
        )

        mock_response = {'content': [{'text': 'Found 2 logs with some NULL values'}]}

        with patch.object(summarizer.client, 'invoke_model', return_value={'body': MagicMock(read=lambda: str(mock_response).encode())}):
            with patch('json.loads', return_value=mock_response):
                summary = summarizer.summarize(
                    original_query="test",
                    sql_query="SELECT * FROM logs",
                    result=result
                )

        assert summary.success is True

    def test_empty_query_validation(self):
        """Test validation of empty query strings."""
        summarizer = BedrockResultSummarizer()

        result = QueryResult(
            success=True,
            rows=[],
            row_count=0,
            column_names=[],
            execution_time_ms=0.0,
            truncated=False
        )

        with pytest.raises(ValueError, match="Original query cannot be empty"):
            summarizer.summarize(
                original_query="",
                sql_query="SELECT * FROM logs",
                result=result
            )

    def test_empty_sql_validation(self):
        """Test validation of empty SQL strings."""
        summarizer = BedrockResultSummarizer()

        result = QueryResult(
            success=True,
            rows=[],
            row_count=0,
            column_names=[],
            execution_time_ms=0.0,
            truncated=False
        )

        with pytest.raises(ValueError, match="SQL query cannot be empty"):
            summarizer.summarize(
                original_query="test query",
                sql_query="",
                result=result
            )

    def test_none_result_validation(self):
        """Test validation of None QueryResult."""
        summarizer = BedrockResultSummarizer()

        with pytest.raises(ValueError, match="QueryResult cannot be None"):
            summarizer.summarize(
                original_query="test query",
                sql_query="SELECT * FROM logs",
                result=None
            )


class TestContextBuilding:
    """Test summary context building logic."""

    def test_build_summary_context_structure(self):
        """Test that context includes all required fields."""
        summarizer = BedrockResultSummarizer()

        result = QueryResult(
            success=True,
            rows=[{"id": 1, "level": "ERROR"}],
            row_count=1,
            column_names=["id", "level"],
            execution_time_ms=25.5,
            truncated=False
        )

        context = summarizer._build_summary_context(
            original_query="show errors",
            sql_query="SELECT * FROM logs WHERE level='ERROR'",
            result=result
        )

        # Verify context contains required sections
        assert "Query Context" in context
        assert "User Query" in context
        assert "show errors" in context
        assert "SQL Query" in context
        assert "SELECT * FROM logs WHERE level='ERROR'" in context
        assert "Results" in context
        assert "1 rows returned" in context
        assert "Execution Time" in context
        assert "25.5ms" in context
        assert "Truncated" in context
        assert "False" in context

    def test_build_summary_context_preview_limit(self):
        """Test that context only includes first 10 rows."""
        summarizer = BedrockResultSummarizer()

        # Create 50 rows
        rows = [{"id": i, "level": "INFO"} for i in range(50)]

        result = QueryResult(
            success=True,
            rows=rows,
            row_count=50,
            column_names=["id", "level"],
            execution_time_ms=100.0,
            truncated=False
        )

        context = summarizer._build_summary_context(
            original_query="show logs",
            sql_query="SELECT * FROM logs",
            result=result
        )

        # Context should mention 50 rows returned
        assert "50 rows returned" in context

        # But preview should only include first 10 (id: 0-9)
        # Count occurrences of "id" in the JSON preview section
        import json
        # The context should have the JSON preview
        assert '"id": 0' in context
        assert '"id": 9' in context
        assert '"id": 49' not in context  # Should not include last row

    def test_build_summary_context_truncated_note(self):
        """Test truncation note appears when results truncated."""
        summarizer = BedrockResultSummarizer()

        result = QueryResult(
            success=True,
            rows=[{"id": i} for i in range(10)],
            row_count=10,
            column_names=["id"],
            execution_time_ms=50.0,
            truncated=True
        )

        context = summarizer._build_summary_context(
            original_query="test",
            sql_query="SELECT * FROM logs",
            result=result
        )

        assert "truncated" in context.lower()
        assert "first 10 of 10" in context.lower()


class TestInitialization:
    """Test BedrockResultSummarizer initialization."""

    def test_initialization_default(self):
        """Test initialization with default configuration."""
        with patch.dict('os.environ', {'AWS_REGION': 'us-east-1'}):
            with patch('boto3.client') as mock_boto:
                summarizer = BedrockResultSummarizer()

                assert summarizer.region == "us-east-1"
                assert "claude" in summarizer.model_id.lower()
                mock_boto.assert_called_once_with("bedrock-runtime", region_name="us-east-1")

    def test_initialization_custom(self):
        """Test initialization with custom region and model."""
        with patch('boto3.client') as mock_boto:
            summarizer = BedrockResultSummarizer(
                region="eu-west-1",
                model_id="anthropic.claude-3-opus-20240229-v1:0"
            )

            assert summarizer.region == "eu-west-1"
            assert summarizer.model_id == "anthropic.claude-3-opus-20240229-v1:0"
            mock_boto.assert_called_once_with("bedrock-runtime", region_name="eu-west-1")

    def test_initialization_from_env(self):
        """Test initialization from environment variables."""
        with patch.dict('os.environ', {
            'AWS_REGION': 'ap-southeast-1',
            'BEDROCK_MODEL_ID': 'anthropic.claude-3-sonnet-20240229-v1:0'
        }):
            with patch('boto3.client'):
                summarizer = BedrockResultSummarizer()

                assert summarizer.region == "ap-southeast-1"
                assert summarizer.model_id == "anthropic.claude-3-sonnet-20240229-v1:0"

    def test_few_shot_examples_built_on_init(self):
        """Test that few-shot examples are built during initialization."""
        with patch('boto3.client'):
            summarizer = BedrockResultSummarizer()

            # Verify few_shot_examples is populated
            assert summarizer.few_shot_examples is not None
            assert len(summarizer.few_shot_examples) > 0
            assert "Example" in summarizer.few_shot_examples
            assert "Summary" in summarizer.few_shot_examples

    def test_initialization_bedrock_client_failure(self):
        """Test initialization failure when Bedrock client cannot be created."""
        with patch('boto3.client', side_effect=Exception("AWS credentials not found")):
            with pytest.raises(BedrockError, match="Failed to initialize Bedrock client"):
                BedrockResultSummarizer()


class TestRetryLogic:
    """Test retry mechanism for Bedrock API calls."""

    def test_retry_unexpected_exception(self):
        """Test unexpected exception during retry."""
        summarizer = BedrockResultSummarizer()

        result = QueryResult(
            success=True,
            rows=[{"id": 1}],
            row_count=1,
            column_names=["id"],
            execution_time_ms=10.0,
            truncated=False
        )

        # Mock an unexpected exception (not ClientError)
        with patch.object(summarizer.client, 'invoke_model', side_effect=RuntimeError("Unexpected error")):
            summary = summarizer.summarize(
                original_query="test",
                sql_query="SELECT * FROM logs",
                result=result
            )

        assert summary.success is False
        assert "Unexpected error during Bedrock invocation" in summary.error_message

    def test_summarize_json_parsing_error(self):
        """Test JSON parsing error during response handling."""
        summarizer = BedrockResultSummarizer()

        result = QueryResult(
            success=True,
            rows=[{"id": 1}],
            row_count=1,
            column_names=["id"],
            execution_time_ms=10.0,
            truncated=False
        )

        # Mock response body read to return invalid JSON
        mock_body = MagicMock()
        mock_body.read.return_value = b"not valid json"

        with patch.object(summarizer.client, 'invoke_model', return_value={'body': mock_body}):
            summary = summarizer.summarize(
                original_query="test",
                sql_query="SELECT * FROM logs",
                result=result
            )

        # Should get fallback summary due to JSON parsing error
        assert summary.success is False
        assert "Query returned 1 rows" in summary.summary
        assert "failed" in summary.summary.lower() or "unavailable" in summary.summary.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
