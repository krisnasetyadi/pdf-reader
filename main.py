# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
from config import config
from processor import processor
from router.upload import router as upload_router
from router.query import router as query_router
from router.collections import router as collections_router
from qa_dataset_generator import build_qa_dataset

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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(config.upload_folder, exist_ok=True)
os.makedirs(config.index_folder, exist_ok=True)


@app.on_event("startup")
async def startup_event():
    try:
        processor.initialize_components()
        logger.info("Application startup completed")
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise

app.include_router(upload_router)
app.include_router(query_router)
app.include_router(collections_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
