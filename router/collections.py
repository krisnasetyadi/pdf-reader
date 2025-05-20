from fastapi import APIRouter, HTTPException
from models import CollectionInfo
from config import config
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
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
        for root, dirs, files in os.walk(config.index_folder):
            for f in files:
                if f.endswith('.faiss'):
                    collection_id = os.path.basename(root)
                    created_at = datetime.fromtimestamp(
                        os.path.getmtime(os.path.join(root, f))
                    )
                    
                    # Count documents in collection (approximate)
                    vector_store = FAISS.load_local(
                        os.path.join(config.index_folder, collection_id),
                        HuggingFaceEmbeddings(model_name=config.embedding_model),
                        allow_dangerous_deserialization=True
                    )
                    
                    # Get unique source files
                    source_files = set()
                    if hasattr(vector_store, 'docstore'):
                        for doc_id in vector_store.docstore._dict:
                            doc = vector_store.docstore._dict[doc_id]
                            if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                                source_files.add(doc.metadata['source'])
                    
                    collections.append(CollectionInfo(
                        collection_id=collection_id,
                        document_count=len(source_files),
                        created_at=created_at.isoformat(),
                        file_names=list(source_files)
                    ))
        
        return collections
    
    except Exception as e:
        logger.error(f"Failed to list collections: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/collection/{collection_id}")
async def delete_collection(collection_id: str):
    """Delete a document collection"""
    try:
        index_path = os.path.join(config.index_folder, collection_id)
        if not os.path.exists(f"{index_path}.faiss"):
            raise HTTPException(status_code=404, detail="Collection not found")
        
        index_files = [
            os.path.join(config.index_folder, f"{collection_id}.faiss"),
            os.path.join(config.index_folder, f"{collection_id}.pkl")
        ]
        
        deleted = False
        for f in index_files:
            if os.path.exists(f):
                os.remove(f)
                deleted = True
        
        upload_dir = os.path.join(config.upload_folder, collection_id)
        if os.path.exists(upload_dir):
            shutil.rmtree(upload_dir)
            deleted = True
        
        if not deleted:
            raise HTTPException(status_code=404, detail="Collection not found")
        
        return {"status": "success", "message": "Collection deleted"}
    
    except Exception as e:
        logger.error(f"Failed to delete collection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
