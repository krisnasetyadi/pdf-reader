# router/join.py
from fastapi import APIRouter, HTTPException
import logging
import asyncio
from datetime import datetime

from join_processor import join_processor
from models import JoinQueryRequest, JoinQueryResponse

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/query/join", response_model=JoinQueryResponse)
async def query_join(request: JoinQueryRequest):
    """Execute join/cross table queries"""
    start_time = datetime.now()
    
    try:
        # Validate question
        if not request.question.strip():
            raise HTTPException(
                status_code=400,
                detail="Pertanyaan tidak boleh kosong"
            )
        
        if not join_processor:
            raise HTTPException(
                status_code=500,
                detail="Join processor not initialized"
            )
        
        # Execute join query
        result = await asyncio.to_thread(
            join_processor.execute_join_query,
            request.question,
            request.join_type
        )
        
        return JoinQueryResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Join query failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Terjadi kesalahan dalam memproses join query: {str(e)}"
        )

@router.get("/join/relationships")
async def get_relationships():
    """Get discovered table relationships"""
    try:
        if not join_processor:
            raise HTTPException(status_code=500, detail="Join processor not initialized")
        
        return {
            "relationships": join_processor.table_relationships,
            "count": len(join_processor.table_relationships)
        }
        
    except Exception as e:
        logger.error(f"Failed to get relationships: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get relationships")