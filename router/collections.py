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
    """List available PDF document collections"""
    try:
        collections = []
        
        # Check if index folder exists
        if not os.path.exists(config.index_folder):
            logger.warning(f"Index folder not found: {config.index_folder}")
            return collections
            
        for entry in os.listdir(config.index_folder):
            entry_path = os.path.join(config.index_folder, entry)
            if os.path.isdir(entry_path):
                index_file = os.path.join(entry_path, "index.faiss")
                if os.path.exists(index_file):
                    index_mtime = os.path.getmtime(index_file)
                    created_at = datetime.fromtimestamp(index_mtime)
                    
                    # Get file names from uploads folder
                    upload_path = os.path.join(config.upload_folder, entry)
                    file_names = []
                    if os.path.exists(upload_path):
                        file_names = [f for f in os.listdir(upload_path) if f.endswith('.pdf')]
                    
                    # Try to get more info from vector store, but don't fail if it doesn't work
                    source_files = set()
                    try:
                        vector_store = processor.get_vector_store(entry)
                        if vector_store and hasattr(vector_store, 'docstore'):
                            for doc_id in vector_store.docstore._dict:
                                doc = vector_store.docstore._dict[doc_id]
                                if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                                    source_files.add(os.path.basename(doc.metadata['source']))
                    except Exception as e:
                        logger.warning(f"Could not load vector store for {entry}: {str(e)}")
                    
                    # Use file_names from uploads if source_files is empty
                    final_file_names = list(source_files) if source_files else file_names
                    
                    collections.append(CollectionInfo(
                        collection_id=entry,
                        document_count=len(final_file_names) if final_file_names else 1,
                        created_at=created_at.isoformat(),
                        file_names=final_file_names
                    ))
        
        logger.info(f"Found {len(collections)} collections")
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
            logger.warning(f"🚨 Path traversal attempt: {decoded_file_name}")
            raise HTTPException(status_code=400, detail="Invalid file name")
        
        # Build file path - use absolute path for better compatibility
        upload_folder_abs = os.path.abspath(config.upload_folder)
        file_path = os.path.join(upload_folder_abs, collection_id, decoded_file_name)
        
        # Enhanced logging for debugging
        logger.info(f"🔍 PDF request - Collection: {collection_id}, File: {decoded_file_name}")
        logger.info(f"🔍 CWD: {os.getcwd()}")
        logger.info(f"🔍 Upload folder (config): {config.upload_folder}")
        logger.info(f"🔍 Upload folder (absolute): {upload_folder_abs}")
        logger.info(f"🔍 Looking for file at: {file_path}")
        logger.info(f"🔍 Upload folder exists: {os.path.exists(upload_folder_abs)}")
        logger.info(f"🔍 Collection folder exists: {os.path.exists(os.path.join(upload_folder_abs, collection_id))}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            # Enhanced error details for debugging
            collection_path = os.path.join(config.upload_folder, collection_id)
            if not os.path.exists(collection_path):
                logger.error(f"❌ Collection folder not found: {collection_path}")
                raise HTTPException(status_code=404, detail=f"Collection '{collection_id}' not found")
            
            # List available files in collection for debugging
            try:
                available_files = os.listdir(collection_path)
                logger.error(f"❌ File '{decoded_file_name}' not found in collection '{collection_id}'")
                logger.error(f"📁 Available files: {available_files}")
                raise HTTPException(
                    status_code=404, 
                    detail=f"File '{decoded_file_name}' not found in collection '{collection_id}'. Available files: {', '.join(available_files) if available_files else 'none'}"
                )
            except OSError:
                logger.error(f"❌ Cannot access collection folder: {collection_path}")
                raise HTTPException(status_code=404, detail=f"Cannot access collection '{collection_id}'")
        
        # Check if it's a PDF
        if not file_path.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        logger.info(f"✅ Serving PDF: {file_path}")
        
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