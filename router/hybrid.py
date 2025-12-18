from fastapi import APIRouter, HTTPException, Request
import logging
import asyncio
from datetime import datetime
from typing import List, Optional

from processor import processor
from config import config, AVAILABLE_MODELS, LLMProvider
from models import HybridQueryRequest, HybridResponse, QueryRequest, SearchType, PdfSourceInfo

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get('/models/available')
async def get_available_models():
    """Get list of available LLM models (all FREE)"""
    return {
        "default_provider": config.llm_provider.value,
        "default_model": config.model_name,
        "available_models": {provider.value: models for provider, models in AVAILABLE_MODELS.items()},
        "usage_hint": "Send 'llm_provider' and 'llm_model' in your query request to switch models"
    }


@router.post('/query/hybrid', response_model=HybridResponse)
async def hybrid_query(request: HybridQueryRequest, req: Request):
    """
    Hybrid query across PDF documents, database, and chat logs.
    
    Optional LLM selection:
    - llm_provider: "huggingface" | "gemini" (default: huggingface)
    - llm_model: specific model name (see /api/v1/models/available)
    """
    start_time = datetime.now()
    
    # Get base URL for file serving
    base_url = str(req.base_url).rstrip('/')

    logger.info(f"🔍 Hybrid query received: {request.question}")
    if request.llm_provider or request.llm_model:
        logger.info(f"🤖 LLM override: provider={request.llm_provider}, model={request.llm_model}")
    
    try:
        # Validate question
        if not request.question.strip() or len(request.question.strip()) < 3:
            raise HTTPException(
                status_code=400,
                detail="Pertanyaan terlalu pendek atau kosong"
            )

        # Determine PDF collections to search
        if request.collection_id:
            collection_ids = [request.collection_id]
            logger.info(f"Searching in specific collection: {request.collection_id}")
        else:
            collection_ids = processor.get_all_collections()
            if collection_ids:
                logger.info(f"Searching across {len(collection_ids)} PDF collections")
            else:
                logger.info("No PDF collections available")

        # Check what to search
        should_search_pdfs = request.include_pdf_results and collection_ids
        should_search_db = request.include_db_results
        should_search_chat = request.include_chat_results

        logger.info(f"PDF: {should_search_pdfs}, DB: {should_search_db}, Chat: {should_search_chat}")

        # Perform hybrid search with all flags
        hybrid_results = await asyncio.to_thread(
            processor.hybrid_search, 
            request.question, 
            collection_ids,
            should_search_chat,  # include_chat
            should_search_pdfs,  # include_pdf
            should_search_db     # include_db
        )

        # Generate answer with optional LLM selection
        answer, model_used = await asyncio.to_thread(
            processor.generate_hybrid_answer, 
            hybrid_results, 
            request.question,
            request.llm_provider,
            request.llm_model
        )

        # Prepare response
        processing_time = (datetime.now() - start_time).total_seconds()

        # Extract PDF sources (simple format for backward compatibility)
        pdf_sources = []
        pdf_sources_detailed = []
        pdf_docs = hybrid_results.get('pdf_documents', [])
        
        for doc in pdf_docs:
            file_name = doc.metadata.get('source', 'Unknown')
            collection_id = doc.metadata.get('collection_id', '')
            page = doc.metadata.get('page')
            score = doc.metadata.get('similarity_score', 0)
            
            # Simple format (backward compatible)
            source_info = f"{file_name}"
            if page:
                source_info += f" (Halaman {page})"
            pdf_sources.append(source_info)
            
            # Detailed format with URLs
            # URL format: /api/v1/files/{collection_id}/{filename}#page={page}
            file_url = f"{base_url}/api/v1/files/{collection_id}/{file_name}" if collection_id else None
            page_url = f"{file_url}#page={page}" if file_url and page else file_url
            
            # Extract first meaningful sentence from content for search highlighting
            content_text = doc.page_content.strip()
            # Get first 50-100 chars as search text (clean version)
            search_text = ' '.join(content_text.split()[:15])  # First ~15 words
            
            pdf_sources_detailed.append(PdfSourceInfo(
                file_name=file_name,
                collection_id=collection_id,
                page=page,
                relevance_score=score,
                content_preview=doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                file_url=file_url,
                page_url=page_url,
                search_text=search_text
            ))

        # Extract chat results
        chat_results = []
        chat_docs = hybrid_results.get('chat_documents', [])
        for doc in chat_docs:
            chat_results.append({
                "source": doc.metadata.get('source', 'Unknown'),
                "platform": doc.metadata.get('platform', 'unknown'),
                "participants": doc.metadata.get('participants', ''),
                "relevance_score": doc.metadata.get('similarity_score', 0),
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            })

        # Log search results
        target_tables = hybrid_results.get('target_tables', [])
        logger.info(f"✅ Search completed - Model: {model_used}, PDFs: {len(pdf_sources)}, DB: {len(hybrid_results.get('database_results', {}))}, Chats: {len(chat_results)}")

        return HybridResponse(
            answer=answer,
            pdf_sources=pdf_sources,
            pdf_sources_detailed=pdf_sources_detailed if pdf_sources_detailed else None,
            db_results=hybrid_results.get('database_results', {}),
            chat_results=chat_results if chat_results else None,
            processing_time=processing_time,
            search_terms=hybrid_results.get('search_terms', []),
            target_tables=target_tables,
            model_used=model_used
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Hybrid query failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Terjadi kesalahan internal: {str(e)}"
        )

@router.post("/query/analyze")
async def analyze_question(request: QueryRequest):
    """Analyze question to recommend search type"""
    try:
        analysis = processor.analyze_question_type(request.question)

        return {
            "question": request.question,
            "recommended_search_type": analysis['recommended_type'],
            "is_db_question": analysis['is_db_question'],
            "is_pdf_question": analysis['is_pdf_question'],
            "extracted_terms": analysis['search_terms'],
        }
    except Exception as e:
        logger.error(f"Question analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Terjadi kesalahan saat menganalisis pertanyaan"
        )

@router.get("/database/tables")
async def get_database_tables():
    """Get available database tables"""
    try:
        if not hasattr(processor, '_db_initialized') or not processor._db_initialized:
            raise HTTPException(status_code=500, detail="Database not initialized")

        tables = processor.db_manager.get_all_tables()
        return {"tables": tables}
    except Exception as e:
        logger.error(f"Failed to get database tables: {e}")
        raise HTTPException(
            status_code=500,
            detail="Terjadi kesalahan saat mengambil tabel database"
        )

@router.get("/database/tables/{table_name}/sample")
async def get_table_sample(table_name: str, limit: int = 5):
    """Get sample data from a table"""
    try:
        if not hasattr(processor, '_db_initialized') or not processor._db_initialized:
            raise HTTPException(status_code=500, detail="Database not initialized")
        
        sample_data = processor.db_manager.get_table_sample(table_name, limit)

        return {
            "table_name": table_name,
            "sample_data": sample_data,
            "record_count": len(sample_data)
        }
    except Exception as e:
        logger.error(f"Failed to get sample data from table {table_name}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Terjadi kesalahan saat mengambil data sampel dari tabel"
        )