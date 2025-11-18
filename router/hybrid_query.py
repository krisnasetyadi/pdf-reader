"""
New hybrid query router that replaces the old query.py
"""
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
import logging
import asyncio
from datetime import datetime
from typing import Optional, List

from hybrid_processor import hybrid_processor
from database_sqlite import get_db, test_connection
from db_models import ChatSession, ChatMessage
from models_updated import QueryRequest, HybridQAResponse
from processor import processor
import json

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/hybrid-query", response_model=HybridQAResponse)
async def hybrid_query_documents(request: QueryRequest, db: Session = Depends(get_db)):
    """Enhanced hybrid query that searches both structured and unstructured data"""
    start_time = datetime.now()

    try:
        # Validate question
        if not request.question.strip() or len(request.question.strip()) < 3:
            raise HTTPException(
                status_code=400,
                detail="Pertanyaan terlalu pendek atau kosong"
            )

        # Initialize processor if not already done
        processor.initialize_components()

        # Perform hybrid search
        result = await hybrid_processor.hybrid_query(
            query=request.question,
            collection_ids=[request.collection_id] if request.collection_id else None
        )

        # Generate answer using the best results
        top_results = result['fused_results'][:5]  # Use top 5 results

        if not top_results:
            raise HTTPException(
                status_code=404,
                detail="Tidak ditemukan informasi relevan dalam dokumen"
            )

        # Prepare context for LLM
        context_parts = []
        sources = []

        for res in top_results:
            context_parts.append(f"[{res['type']}] {res['content'][:500]}...")

            if request.include_sources:
                if res['type'] == 'document_chunk':
                    source = res['metadata'].get('source', 'Unknown')
                    page = res['metadata'].get('page', 'N/A')
                    sources.append(f"{source} (page {page})")
                elif res['type'] == 'entity':
                    sources.append(f"Entity: {res['metadata'].get('entity_type', 'Unknown')}")
                elif res['type'] == 'qa_pair':
                    sources.append("Previous Q&A")
                elif res['type'] == 'chat_message':
                    sources.append("Chat History")

        combined_context = "\n\n".join(context_parts)

        # Generate answer
        answer = await asyncio.to_thread(
            processor.generate_answer,
            combined_context,
            request.question
        )

        # Save to chat history if session provided
        if hasattr(request, 'session_id') and request.session_id:
            try:
                # Save user question
                user_message = ChatMessage(
                    session_id=request.session_id,
                    role="user",
                    content=request.question,
                    message_metadata={"query_type": "hybrid"}
                )
                db.add(user_message)

                # Save assistant response
                assistant_message = ChatMessage(
                    session_id=request.session_id,
                    role="assistant",
                    content=answer,
                    message_metadata={
                        "sources": sources,
                        "intent_classification": result['intent_classification'],
                        "result_counts": {
                            "structured": result['structured_count'],
                            "unstructured": result['unstructured_count']
                        }
                    }
                )
                db.add(assistant_message)
                db.commit()

            except Exception as e:
                logger.error(f"Error saving chat history: {e}")
                db.rollback()

        processing_time = (datetime.now() - start_time).total_seconds()

        return HybridQAResponse(
            answer=answer,
            sources=list(set(sources)),
            intent_classification=result['intent_classification'],
            structured_results_count=result['structured_count'],
            unstructured_results_count=result['unstructured_count'],
            total_results_count=result['total_results'],
            processing_time=processing_time,
            collection_id=request.collection_id
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Hybrid query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chat-sessions")
async def get_chat_sessions(db: Session = Depends(get_db)):
    """Get all chat sessions"""
    try:
        sessions = db.query(ChatSession).order_by(ChatSession.last_activity.desc()).all()
        return [
            {
                "id": session.id,
                "name": session.session_name or f"Session {session.id[:8]}",
                "created_at": session.created_at.isoformat() if session.created_at else None,
                "last_activity": session.last_activity.isoformat() if session.last_activity else None
            }
            for session in sessions
        ]
    except Exception as e:
        logger.error(f"Error fetching chat sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat-sessions")
async def create_chat_session(session_name: Optional[str] = None, db: Session = Depends(get_db)):
    """Create new chat session"""
    try:
        session = ChatSession(session_name=session_name)
        db.add(session)
        db.commit()
        db.refresh(session)

        return {
            "id": session.id,
            "name": session.session_name or f"Session {session.id[:8]}",
            "created_at": session.created_at.isoformat() if session.created_at else None
        }
    except Exception as e:
        logger.error(f"Error creating chat session: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chat-sessions/{session_id}/messages")
async def get_chat_messages(session_id: str, db: Session = Depends(get_db)):
    """Get messages for a chat session"""
    try:
        messages = db.query(ChatMessage).filter(
            ChatMessage.session_id == session_id
        ).order_by(ChatMessage.timestamp.asc()).all()

        return [
            {
                "id": msg.id,
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
                "metadata": msg.message_metadata
            }
            for msg in messages
        ]
    except Exception as e:
        logger.error(f"Error fetching chat messages: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check(db: Session = Depends(get_db)):
    """Health check for hybrid system"""
    try:
        # Test database connection
        db_status = test_connection()

        # Test processor
        processor_status = processor._initialized

        return {
            "status": "healthy" if db_status and processor_status else "degraded",
            "database": "connected" if db_status else "disconnected",
            "processor": "initialized" if processor_status else "not_initialized",
            "hybrid_processor": "active"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }
