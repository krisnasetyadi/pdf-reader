from fastapi import APIRouter, HTTPException
import logging
import asyncio
from datetime import datetime
from typing import List, Optional

from processor import processor
from config import config
from models import HybridQueryRequest, HybridResponse, QueryRequest, SearchType

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post('/query/hybrid', response_model=HybridResponse)
async def hybrid_query(request: HybridQueryRequest):
    """Hybrid query across PDF documents, database, and chat logs"""
    start_time = datetime.now()

    logger.info(f"🔍 Hybrid query received: {request.question}")
    
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

        # Perform hybrid search (now includes chat)
        hybrid_results = await asyncio.to_thread(
            processor.hybrid_search, 
            request.question, 
            collection_ids,
            should_search_chat  # Pass include_chat flag
        )

        # Generate answer
        answer = await asyncio.to_thread(
            processor.generate_hybrid_answer, hybrid_results, request.question
        )

        # Prepare response
        processing_time = (datetime.now() - start_time).total_seconds()

        # Extract PDF sources
        pdf_sources = []
        pdf_docs = hybrid_results.get('pdf_documents', [])
        for doc in pdf_docs:
            source_info = f"{doc.metadata.get('source', 'Unknown')}"
            if 'page' in doc.metadata:
                source_info += f" (Halaman {doc.metadata['page']})"
            pdf_sources.append(source_info)

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
        logger.info(f"✅ Search completed - PDFs: {len(pdf_sources)}, DB: {len(hybrid_results.get('database_results', {}))}, Chats: {len(chat_results)}")

        return HybridResponse(
            answer=answer,
            pdf_sources=pdf_sources,
            db_results=hybrid_results.get('database_results', {}),
            chat_results=chat_results if chat_results else None,
            processing_time=processing_time,
            search_terms=hybrid_results.get('search_terms', []),
            target_tables=target_tables
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