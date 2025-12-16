# router/collections.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from models import CollectionInfo
from config import config
# from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
from processor import processor
import os
import shutil
from typing import List
from datetime import datetime
import logging
import urllib.parse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/collections", response_model=List[CollectionInfo])
async def list_collections():
    """List available document collections"""
    try:
        collections = []
        for entry in os.listdir(config.index_folder):
            entry_path = os.path.join(config.index_folder, entry)
            if os.path.isdir(entry_path):
                index_file = os.path.join(entry_path, "index.faiss")
                if os.path.exists(index_file):
                    index_mtime = os.path.getmtime(index_file)
                    created_at = datetime.fromtimestamp(index_mtime)
                    
                    # Count documents in collection
                    try:
                        vector_store = processor.get_vector_store(entry)
                        if vector_store:
                            # Get unique source files
                            source_files = set()
                            if hasattr(vector_store, 'docstore'):
                                for doc_id in vector_store.docstore._dict:
                                    doc = vector_store.docstore._dict[doc_id]
                                    if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                                        source_files.add(doc.metadata['source'])
                            
                            collections.append(CollectionInfo(
                                collection_id=entry,
                                document_count=len(source_files),
                                created_at=created_at.isoformat(),
                                file_names=list(source_files)
                            ))
                    except Exception as e:
                        logger.warning(f"Skipping collection {entry}: {str(e)}")
                        continue
        
        return collections
    
    except Exception as e:
        logger.error(f"Failed to list collections: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/collection/{collection_id}")
async def delete_collection(collection_id: str):
    """Delete a document collection"""
    try:
        index_path = os.path.join(config.index_folder, collection_id)
        if not os.path.exists(index_path):
            raise HTTPException(status_code=404, detail="Collection not found")
        
        # Delete index files
        index_files = [
            os.path.join(index_path, "index.faiss"),
            os.path.join(index_path, "index.pkl")
        ]
        
        deleted = False
        for f in index_files:
            if os.path.exists(f):
                os.remove(f)
                deleted = True
        
        # Delete upload folder
        upload_dir = os.path.join(config.upload_folder, collection_id)
        if os.path.exists(upload_dir):
            shutil.rmtree(upload_dir)
            deleted = True
        
        # Delete index folder
        if os.path.exists(index_path):
            shutil.rmtree(index_path)
            deleted = True
        
        if not deleted:
            raise HTTPException(status_code=404, detail="Collection not found")
        
        # Invalidate cache
        processor.invalidate_vector_store_cache(collection_id)
        
        return {"status": "success", "message": "Collection deleted"}
    
    except Exception as e:
        logger.error(f"Failed to delete collection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/files/{collection_id}/{file_name:path}")
async def serve_pdf_file(collection_id: str, file_name: str):
    """
    Serve a PDF file from a collection.
    
    Use with #page=N fragment to jump to specific page in browser's PDF viewer.
    Example: /api/v1/files/abc123/document.pdf#page=5
    """
    try:
        # Decode URL-encoded filename
        decoded_file_name = urllib.parse.unquote(file_name)
        
        # Security: prevent path traversal
        if '..' in decoded_file_name or decoded_file_name.startswith('/'):
            raise HTTPException(status_code=400, detail="Invalid file name")
        
        # Build file path
        file_path = os.path.join(config.upload_folder, collection_id, decoded_file_name)
        
        # Check if file exists
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            raise HTTPException(status_code=404, detail="File not found")
        
        # Check if it's a PDF
        if not file_path.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        logger.info(f"📄 Serving PDF: {file_path}")
        
        return FileResponse(
            path=file_path,
            media_type="application/pdf",
            filename=decoded_file_name,
            # Allow browser to display PDF inline instead of downloading
            headers={
                "Content-Disposition": f'inline; filename="{decoded_file_name}"'
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to serve file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collection/{collection_id}/files")
async def list_collection_files(collection_id: str):
    """List all PDF files in a collection"""
    try:
        upload_dir = os.path.join(config.upload_folder, collection_id)
        
        if not os.path.exists(upload_dir):
            raise HTTPException(status_code=404, detail="Collection not found")
        
        files = []
        for file_name in os.listdir(upload_dir):
            if file_name.lower().endswith('.pdf'):
                file_path = os.path.join(upload_dir, file_name)
                file_stat = os.stat(file_path)
                files.append({
                    "file_name": file_name,
                    "size_bytes": file_stat.st_size,
                    "modified_at": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                    "url": f"/api/v1/files/{collection_id}/{urllib.parse.quote(file_name)}"
                })
        
        return {
            "collection_id": collection_id,
            "file_count": len(files),
            "files": files
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))