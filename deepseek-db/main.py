# main.py - Update the router imports section
# Include routers
try:
    from router.upload import router as upload_router
    from router.query import router as query_router
    from router.collections import router as collections_router
    from router.database import router as database_router  # Add this line

    app.include_router(upload_router, prefix="/api/v1")
    app.include_router(query_router, prefix="/api/v1")
    app.include_router(collections_router, prefix="/api/v1")
    app.include_router(database_router, prefix="/api/v1")  # Add this line
except ImportError as e:
    logger.warning(
        f"Router import failed: {e}. Some endpoints may not be available.")
