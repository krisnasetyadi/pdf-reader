# router/upload.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import os
import shutil
import uuid
from utils import process_pdfs
from models import UploadResponse
from config import config
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/upload", response_model=UploadResponse)
async def upload_pdfs(files: List[UploadFile] = File(...)):
    """Upload and process PDF files"""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    collection_id = str(uuid.uuid4())
    collection_path = os.path.join(config.upload_folder, collection_id)
    os.makedirs(collection_path, exist_ok=True)

    saved_files = []
    for file in files:
        if not file.filename.lower().endswith('.pdf'):
            continue

        file_path = os.path.join(collection_path, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_files.append(file_path)

    if not saved_files:
        raise HTTPException(
            status_code=400, detail="No valid PDF files uploaded")

    try:
        chunk_count = process_pdfs(saved_files, collection_id)

        return UploadResponse(
            collection_id=collection_id,
            file_count=len(saved_files),
            status="success"
        )
    except Exception as e:
        # Clean up on error
        shutil.rmtree(collection_path, ignore_errors=True)
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process PDFs")
