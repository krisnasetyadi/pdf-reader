# router/collections.py
from fastapi import APIRouter, HTTPException
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