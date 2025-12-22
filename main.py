# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
from config import config
from processor import processor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="PDF QA API",
    description="API for querying PDF documents with natural language",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
os.makedirs(config.upload_folder, exist_ok=True)
os.makedirs(config.index_folder, exist_ok=True)
os.makedirs(config.chat_upload_folder, exist_ok=True)
os.makedirs(config.chat_index_folder, exist_ok=True)


@app.on_event("startup")
async def startup_event():
    try:
        processor.initialize_components()
        logger.info("Application startup completed")
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise


@app.get("/health")
async def health_check():
    """Comprehensive health check including database status"""
    from datetime import datetime
    try:
        # Check processor initialization
        pdf_collections = len(processor.get_all_collections()) if processor else 0
        chat_collections = len(processor.get_all_chat_collections()) if processor else 0
        
        # Check database health
        db_status = processor.db_manager.is_healthy() if processor and processor.db_manager else {
            "status": "not_initialized",
            "message": "Database manager not initialized", 
            "can_query": False
        }
        
        return {
            "status": "healthy" if db_status["can_query"] else "degraded",
            "initialized": hasattr(processor, '_initialized') and processor._initialized,
            "pdf_collections_count": pdf_collections,
            "chat_collections_count": chat_collections,
            "database": db_status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "database": {"status": "error", "message": str(e), "can_query": False},
            "timestamp": datetime.now().isoformat()
        }

# Include routers
from router.upload import router as upload_router
from router.collections import router as collections_router
from router.hybrid import router as hybrid_router
from router.chat import router as chat_router

app.include_router(upload_router, prefix="/api/v1")       # PDF upload
app.include_router(collections_router, prefix="/api/v1")  # PDF collections management
app.include_router(hybrid_router, prefix="/api/v1")       # Main hybrid search (PDF + DB + Chat)
app.include_router(chat_router, prefix="/api/v1")         # Chat upload & collections


@app.get("/")
async def root():
    return {"message": "PDF QA API is running", "version": "1.0.0"}


@app.get("/api/v1/version")
async def get_version():
    """Get backend version and last deployment timestamp"""
    import sys
    from datetime import datetime
    
    return {
        "backend": {
            "version": "1.0.0",
            "last_updated": "2025-12-18T12:03:56",
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        },
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/v1/debug/paths")
async def debug_paths():
    """Debug endpoint to check file paths and directory structure"""
    import os
    
    def get_collection_info(upload_folder):
        if not os.path.exists(upload_folder):
            return {"error": "Upload folder does not exist", "path": upload_folder}
        
        collections = []
        try:
            for col in os.listdir(upload_folder):
                col_path = os.path.join(upload_folder, col)
                if os.path.isdir(col_path):
                    try:
                        files = os.listdir(col_path)
                        file_info = []
                        for f in files:
                            fp = os.path.join(col_path, f)
                            if os.path.isfile(fp):
                                file_info.append({
                                    "name": f,
                                    "size": os.path.getsize(fp),
                                    "exists": True
                                })
                        collections.append({
                            "id": col,
                            "files": file_info,
                            "count": len(file_info)
                        })
                    except Exception as e:
                        collections.append({
                            "id": col,
                            "error": str(e)
                        })
        except Exception as e:
            return {"error": f"Cannot list collections: {str(e)}"}
        
        return collections
    
    return {
        "cwd": os.getcwd(),
        "upload_folder": config.upload_folder,
        "upload_folder_abs": os.path.abspath(config.upload_folder),
        "upload_folder_exists": os.path.exists(config.upload_folder),
        "index_folder": config.index_folder,
        "index_folder_exists": os.path.exists(config.index_folder),
        "collections": get_collection_info(config.upload_folder),
        "environment": {
            "HOME": os.environ.get("HOME", "not set"),
            "SPACE_ID": os.environ.get("SPACE_ID", "not set (local)"),
            "USER": os.environ.get("USER", "not set")
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
