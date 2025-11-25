# router/database.py
from fastapi import APIRouter, HTTPException
import logging
import asyncio
from datetime import datetime

from processor import processor
from config import config
from models import DatabaseQueryRequest, DatabaseQueryResponse, DatabaseSchemaResponse, TableInfo

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/query/database", response_model=DatabaseQueryResponse)
async def query_database(request: DatabaseQueryRequest):
    """Query structured data from database"""
    start_time = datetime.now()

    try:
        if not processor._db_initialized:
            raise HTTPException(status_code=500, detail="Database not initialized")

        # Query database
        db_results = await asyncio.to_thread(
            processor.query_database, request.question, request.table_name
        )

        # Generate answer
        answer = await asyncio.to_thread(
            processor.generate_sql_based_answer, db_results, request.question
        )

        processing_time = (datetime.now() - start_time).total_seconds()

        return DatabaseQueryResponse(
            answer=answer,
            results=db_results.get("results", []),
            table_name=db_results.get("table_used", "unknown"),
            processing_time=processing_time
        )

    except Exception as e:
        logger.error(f"Database query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/database/schema", response_model=DatabaseSchemaResponse)
async def get_database_schema():
    """Get database schema information"""
    try:
        if not processor._db_initialized:
            raise HTTPException(status_code=500, detail="Database not initialized")

        tables_info = []
        all_tables = processor.db_manager.get_all_tables()

        for table_name in all_tables[:10]:  # Limit to first 10 tables
            try:
                schema = processor.db_manager.get_table_schema(table_name)
                sample_data = processor.db_manager.get_table_sample(table_name, 3)

                table_info = TableInfo(
                    table_name=table_name,
                    columns=schema,
                    sample_data=sample_data,
                    row_count=len(sample_data)  # This is just sample count, you might want actual count
                )
                tables_info.append(table_info)
            except Exception as e:
                logger.error(f"Failed to get info for table {table_name}: {str(e)}")
                continue

        return DatabaseSchemaResponse(tables=tables_info)

    except Exception as e:
        logger.error(f"Failed to get database schema: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/database/tables")
async def get_database_tables():
    """Get list of available tables"""
    try:
        if not processor._db_initialized:
            raise HTTPException(status_code=500, detail="Database not initialized")

        tables = processor.db_manager.get_all_tables()
        return {"tables": tables}

    except Exception as e:
        logger.error(f"Failed to get tables: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
