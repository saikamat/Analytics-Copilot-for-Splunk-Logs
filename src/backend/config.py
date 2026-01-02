"""
FastAPI application configuration.

Loads configuration from environment variables using Pydantic Settings.
"""

from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application configuration from environment variables."""

    # FastAPI
    app_name: str = "Log Analytics API"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    # CORS (comma-separated string will be split into list)
    cors_origins: str = "*"

    # Query Limits
    default_timeout: int = 30
    max_timeout: int = 300
    default_max_rows: int = 10000
    max_max_rows: int = 50000

    # Database (already in .env)
    database_url: str
    db_pool_min_conn: int = 2
    db_pool_max_conn: int = 10

    # AWS Bedrock (already in .env)
    aws_region: str
    bedrock_model_id: str

    class Config:
        env_file = ".env"
        case_sensitive = False

    def get_cors_origins(self) -> List[str]:
        """Parse CORS origins string into list."""
        if self.cors_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",")]


# Global settings instance
settings = Settings()
