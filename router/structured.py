# router/structured.py
from fastapi import APIRouter, HTTPException
import logging
import asyncio
from datetime import datetime

from structured_processor import structured_processor
from models import StructuredQueryRequest, StructuredQueryResponse

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/query/structured", response_model=StructuredQueryResponse)
async def query_structured_data(request: StructuredQueryRequest):
    """Query structured data from database with natural language"""
    start_time = datetime.now()
    print(f"🔍 Structured query received: {request.question}")
    try:
        # Validate question
        if not request.question.strip() or len(request.question.strip()) < 2:
            raise HTTPException(
                status_code=400,
                detail="Pertanyaan terlalu pendek atau kosong"
            )
        
        # Execute structured query
        result = await asyncio.to_thread(
            structured_processor.execute_structured_query,
            request.question,
            request.table_name
        )
        
        return StructuredQueryResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Structured query failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Terjadi kesalahan dalam memproses query: {str(e)}"
        )

@router.get("/structured/tables")
async def get_structured_tables():
    """Get available database tables with schema information"""
    try:
        tables_info = []
        for table_name, schema in structured_processor.table_schemas.items():
            # Get sample data
            sample_data = structured_processor.db_manager.get_table_sample(table_name, 2)
            
            tables_info.append({
                "table_name": table_name,
                "description": schema['description'],
                "columns": schema['columns'],
                "sample_data": sample_data
            })
        
        return {"tables": tables_info}
        
    except Exception as e:
        logger.error(f"Failed to get tables info: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Terjadi kesalahan saat mengambil informasi tabel"
        )