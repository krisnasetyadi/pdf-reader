# router/chat.py
"""
Chat logs upload and search endpoints
Handles WhatsApp TXT file uploads and indexing to FAISS
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query, Depends
import logging
import os
import uuid
import shutil
from datetime import datetime
from typing import Optional, List

import storage as supabase_storage
from config import config
from models import ChatUploadResponse, ChatPlatform, ChatCollection, SetChatCollectionActiveRequest
from chat_parser import ChatParser
from processor import processor
from router.auth import require_role, UserRecord

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post('/chat/upload', response_model=ChatUploadResponse)
async def upload_chat(
    file: UploadFile = File(...),
    platform: str = Form(default="whatsapp"),
    _: UserRecord = Depends(require_role("admin")),
):
    """
    Upload and process a chat export file
    
    - **file**: Chat export file (TXT for WhatsApp)
    - **platform**: Chat platform (whatsapp, teams, slack). Default: whatsapp
    """
    logger.info(f"📱 Chat upload received: {file.filename}, platform: {platform}")
    
    # Validate platform
    if platform.lower() not in config.supported_chat_platforms:
        raise HTTPException(
            status_code=400,
            detail=f"Platform '{platform}' not supported. Supported: {config.supported_chat_platforms}"
        )
    
    # Validate file extension
    if platform.lower() == "whatsapp" and not file.filename.lower().endswith('.txt'):
        raise HTTPException(
            status_code=400,
            detail="WhatsApp exports should be .txt files"
        )
    
    try:
        # Generate collection ID
        collection_id = str(uuid.uuid4())
        
        # Create directories
        upload_dir = os.path.join(config.chat_upload_folder, collection_id)
        index_dir = os.path.join(config.chat_index_folder, collection_id)
        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(index_dir, exist_ok=True)
        
        # Save uploaded file
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, 'wb') as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"💾 Saved chat file to: {file_path}")
        
        # Parse chat file
        parser = ChatParser()
        messages, metadata = parser.parse_whatsapp(file_path)
        
        if not messages:
            raise HTTPException(
                status_code=400,
                detail="No messages found in chat file. Please check the file format."
            )
        
        # Extract keywords automatically
        keywords = parser.extract_keywords(messages, top_n=30)
        metadata['keywords'] = keywords
        logger.info(f"🔑 Extracted {len(keywords)} keywords for search boosting")
        
        # Chunk messages for vectorization
        chunks = parser.chunk_messages_by_conversation(
            messages,
            chunk_size=config.chat_chunk_size,
            overlap=config.chat_chunk_overlap
        )
        
        # Create vector store from chunks
        await _create_chat_vector_store(collection_id, chunks, metadata, platform)
        
        # Upload to Supabase Storage
        if supabase_storage.is_enabled():
            try:
                index_dir = os.path.join(config.chat_index_folder, collection_id)
                supabase_storage.upload_chat_file(collection_id, file_path, file.filename)
                supabase_storage.upload_chat_index(collection_id, index_dir)
                supabase_storage.register_chat_collection(
                    collection_id=collection_id,
                    file_name=file.filename,
                    platform=platform.lower(),
                    message_count=len(messages),
                    participants=metadata.get("participants", []),
                    date_range=metadata.get("date_range"),
                    keywords=keywords,
                    storage_paths=[f"chat-uploads/{collection_id}/{file.filename}"],
                )
                logger.info(f"☁️ Chat collection synced to Supabase: {collection_id}")
            except Exception as e:
                logger.warning(f"Supabase sync failed (disk fallback): {e}")

        # Save collection metadata
        collection = ChatCollection(
            collection_id=collection_id,
            platform=ChatPlatform(platform.lower()),
            file_name=file.filename,
            message_count=len(messages),
            date_range=metadata.get("date_range"),
            participants=metadata.get("participants", []),
            created_at=datetime.now()
        )
        
        # Add keywords to collection metadata
        collection_dict = collection.model_dump(mode='json')
        collection_dict['keywords'] = keywords
        
        # Save metadata to file
        _save_collection_metadata(collection_id, collection_dict)
        
        logger.info(f"✅ Chat collection created: {collection_id} with {len(messages)} messages")
        
        return ChatUploadResponse(
            collection_id=collection_id,
            file_name=file.filename,
            platform=platform,
            message_count=len(messages),
            participants=metadata.get("participants", []),
            date_range=metadata.get("date_range"),
            status="success"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Chat upload failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process chat file: {str(e)}"
        )


async def _create_chat_vector_store(
    collection_id: str, 
    chunks: List[dict], 
    metadata: dict,
    platform: str
):
    """Create FAISS vector store from chat chunks"""
    from langchain_core.documents import Document
    from langchain_community.vectorstores import FAISS
    
    logger.info(f"🔧 Creating vector store for {len(chunks)} chat chunks")
    
    # Convert chunks to Documents
    documents = []
    for i, chunk in enumerate(chunks):
        doc = Document(
            page_content=chunk["text"],
            metadata={
                "source": metadata.get("file_name", "unknown"),
                "collection_id": collection_id,
                "data_type": "chat",  # Distinguish from PDF
                "platform": platform,
                "chunk_index": i,
                "message_count": chunk["message_count"],
                "participants": ", ".join(chunk["participants"]),
                "time_range_start": chunk["time_range"]["start"] if chunk["time_range"]["start"] else "",
                "time_range_end": chunk["time_range"]["end"] if chunk["time_range"]["end"] else "",
            }
        )
        documents.append(doc)
    
    # Create FAISS index using processor's embeddings
    if not processor.embeddings:
        processor.initialize_components()
    
    vector_store = FAISS.from_documents(documents, processor.embeddings)
    
    # Save to disk
    index_path = os.path.join(config.chat_index_folder, collection_id)
    vector_store.save_local(index_path)
    
    logger.info(f"💾 Chat vector store saved to: {index_path}")


def _save_collection_metadata(collection_id: str, collection: dict):
    """Save collection metadata to JSON file"""
    import json
    
    metadata_path = os.path.join(config.chat_index_folder, collection_id, "metadata.json")
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(collection, f, indent=2, default=str)


@router.get('/chat/collections')
async def list_chat_collections(_: UserRecord = Depends(require_role("admin"))):
    """List all available chat collections"""
    # Query Supabase DB first
    if supabase_storage.has_database():
        try:
            rows = supabase_storage.list_chat_collections()
            if rows:
                return {"collections": rows, "count": len(rows)}
        except Exception as e:
            logger.warning(f"Supabase list_chat_collections failed, using disk: {e}")

    # Disk fallback
    collections = []
    chat_index_folder = config.chat_index_folder
    if not os.path.exists(chat_index_folder):
        return {"collections": []}
    
    for collection_id in os.listdir(chat_index_folder):
        collection_path = os.path.join(chat_index_folder, collection_id)
        if os.path.isdir(collection_path):
            metadata_path = os.path.join(collection_path, "metadata.json")
            if os.path.exists(metadata_path):
                import json
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    collections.append(metadata)
            else:
                collections.append({"collection_id": collection_id, "status": "no_metadata"})
    
    return {"collections": collections, "count": len(collections)}


@router.get('/chat/collection/{collection_id}/preview')
async def preview_chat_collection(
    collection_id: str,
    max_chars: int = Query(default=20000, ge=500, le=200000),
    _: UserRecord = Depends(require_role("admin")),
):
    """Return plain-text preview content from an uploaded chat collection file."""
    collection_info = None
    if supabase_storage.has_database():
        try:
            collection_info = supabase_storage.get_chat_collection(collection_id)
        except Exception as exc:
            logger.warning(f"Preview metadata lookup failed for {collection_id}: {exc}")

    upload_dir = os.path.join(config.chat_upload_folder, collection_id)

    preferred_name = None
    if isinstance(collection_info, dict):
        preferred_name = collection_info.get("file_name")

    # Ephemeral disk (e.g. HF Space restart): restore the raw file from the
    # Supabase bucket before giving up.
    if (not os.path.isdir(upload_dir) or not os.listdir(upload_dir)) and preferred_name:
        try:
            if supabase_storage.is_enabled():
                supabase_storage.download_chat_file(collection_id, preferred_name, upload_dir)
        except Exception as exc:
            logger.warning(f"Chat file restore from bucket failed for {collection_id}: {exc}")

    if not os.path.isdir(upload_dir):
        raise HTTPException(status_code=404, detail="Chat collection file not found")

    candidate_paths: List[str] = []
    if preferred_name:
        candidate_paths.append(os.path.join(upload_dir, preferred_name))

    for entry in sorted(os.listdir(upload_dir)):
        full_path = os.path.join(upload_dir, entry)
        if os.path.isfile(full_path):
            candidate_paths.append(full_path)

    file_path = next((path for path in candidate_paths if os.path.isfile(path)), None)
    if not file_path:
        raise HTTPException(status_code=404, detail="No chat file available for preview")

    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as handle:
            content = handle.read(max_chars + 1)
    except Exception as exc:
        logger.error(f"Failed reading chat file preview for {collection_id}: {exc}")
        raise HTTPException(status_code=500, detail="Failed to read chat file")

    truncated = len(content) > max_chars
    preview_text = content[:max_chars] if truncated else content

    return {
        "collection_id": collection_id,
        "file_name": os.path.basename(file_path),
        "content_preview": preview_text,
        "truncated": truncated,
        "max_chars": max_chars,
    }


@router.post('/chat-collection/activate')
async def set_chat_collection_active(
    body: SetChatCollectionActiveRequest,
    _: UserRecord = Depends(require_role("admin")),
):
    """Toggle a chat collection's active status (used as a knowledge source)."""
    if not supabase_storage.is_enabled() and not supabase_storage.has_database():
        raise HTTPException(status_code=503, detail="Database unavailable")
    updated = supabase_storage.set_chat_collection_status(body.collection_id, body.active)
    if not updated:
        raise HTTPException(status_code=404, detail="Chat collection not found")
    return {"status": "success", "collection_id": body.collection_id, "active": body.active}


@router.delete('/chat/collections/{collection_id}')
@router.delete('/chat/collection/{collection_id}')  # Alternative path for compatibility
async def delete_chat_collection(
    collection_id: str,
    _: UserRecord = Depends(require_role("admin")),
):
    """Delete a chat collection"""
    import shutil
    
    # Remove from cache if exists
    cache_key = f"chat_{collection_id}"
    if cache_key in processor.vector_store_cache:
        del processor.vector_store_cache[cache_key]

    # Delete from Supabase (DB row + S3 objects)
    if supabase_storage.is_enabled() or supabase_storage.has_database():
        try:
            supabase_storage.delete_chat_collection_from_db(collection_id)
        except Exception as e:
            logger.warning(f"Supabase delete failed for {collection_id}: {e}")

    # Delete local upload folder
    upload_path = os.path.join(config.chat_upload_folder, collection_id)
    if os.path.exists(upload_path):
        shutil.rmtree(upload_path)

    # Delete local index folder
    index_path = os.path.join(config.chat_index_folder, collection_id)
    if os.path.exists(index_path):
        shutil.rmtree(index_path)

    return {"status": "deleted", "collection_id": collection_id}
