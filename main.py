# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
from config import config
from processor import processor
from database_sqlite import create_tables, test_connection

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Hybrid PDF QA API",
    description="API for querying both structured and unstructured data with natural language",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
os.makedirs(config.upload_folder, exist_ok=True)
os.makedirs(config.index_folder, exist_ok=True)


@app.on_event("startup")
async def startup_event():
    try:
        # Initialize database
        logger.info("Testing database connection...")
        if test_connection():
            logger.info("Database connection successful")
            create_tables()
            logger.info("Database tables created/verified")
        else:
            logger.warning("Database connection failed - hybrid features may not work")

        # Initialize ML processor
        processor.initialize_components()
        logger.info("Application startup completed")
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "initialized": hasattr(processor, '_initialized')
        and processor._initialized,
        "collections_count": len(processor.get_all_collections())
    }

# Include routers
try:
    from router.upload import router as upload_router
    from router.query import router as query_router
    from router.collections import router as collections_router
    from router.hybrid_query import router as hybrid_router

    app.include_router(upload_router, prefix="/api/v1")
    app.include_router(query_router, prefix="/api/v1")
    app.include_router(collections_router, prefix="/api/v1")
    app.include_router(hybrid_router, prefix="/api/v2", tags=["hybrid"])  # New hybrid endpoints
except ImportError as e:
    logger.warning(
        f"Router import failed: {e}. Some endpoints may not be available.")


@app.get("/")
async def root():
    return {
        "message": "Hybrid PDF QA API is running",
        "version": "2.0.0",
        "features": {
            "structured_data": "PostgreSQL",
            "unstructured_data": "PDF + Chat logs",
            "hybrid_search": "Combined querying"
        },
        "endpoints": {
            "v1": "/api/v1 (original endpoints)",
            "v2": "/api/v2 (hybrid endpoints)"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
