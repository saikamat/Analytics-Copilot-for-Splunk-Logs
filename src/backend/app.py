"""
FastAPI application for Log Analytics API.

Provides REST API endpoints for natural language log queries using
BedrockSQLGenerator and QueryExecutor.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from src.backend.config import settings
from src.shared.db_utils import DatabaseConnectionPool, QueryExecutor
from src.shared.result_summarizer import BedrockResultSummarizer

# Configure logging
logging.basicConfig(
    level=settings.log_level.upper(),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global connection pool, executor, and summarizer (initialized on startup)
db_pool = None
query_executor = None
result_summarizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle (startup/shutdown).

    Startup:
    - Initialize database connection pool
    - Create QueryExecutor instance
    - Create BedrockResultSummarizer instance

    Shutdown:
    - Close all database connections
    """
    global db_pool, query_executor, result_summarizer

    # Startup
    logger.info("Starting Log Analytics API...")
    try:
        db_pool = DatabaseConnectionPool(
            database_url=settings.database_url,
            min_conn=settings.db_pool_min_conn,
            max_conn=settings.db_pool_max_conn
        )
        query_executor = QueryExecutor(db_pool)
        logger.info(f"✅ Database connection pool initialized ({settings.db_pool_min_conn}-{settings.db_pool_max_conn} connections)")

        result_summarizer = BedrockResultSummarizer()
        logger.info(f"✅ Result summarizer initialized (model: {result_summarizer.model_id})")
    except Exception as e:
        logger.error(f"❌ Failed to initialize database connection pool: {e}")
        raise

    yield  # Application runs

    # Shutdown
    logger.info("Shutting down Log Analytics API...")
    try:
        if db_pool:
            db_pool.close_all()
            logger.info("✅ Database connection pool closed")
    except Exception as e:
        logger.error(f"❌ Error closing database connection pool: {e}")


# Initialize FastAPI application
app = FastAPI(
    title=settings.app_name,
    description="Natural language query interface for system logs using LLMs and vector search",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info(f"CORS enabled for origins: {settings.get_cors_origins()}")


# Include API routes
from src.backend.routes import router
app.include_router(router, prefix="/api", tags=["API"])


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - API information."""
    return {
        "name": settings.app_name,
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc"
    }
