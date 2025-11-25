# models.py
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# Existing models...


class DatabaseQueryRequest(BaseModel):
    question: str
    table_name: Optional[str] = None
    use_semantic_search: bool = True


class DatabaseQueryResponse(BaseModel):
    answer: str
    sql_query: Optional[str] = None
    results: List[Dict[str, Any]] = []
    table_name: str
    processing_time: float


class TableInfo(BaseModel):
    table_name: str
    columns: List[Dict[str, str]]
    sample_data: List[Dict[str, Any]]
    row_count: int


class DatabaseSchemaResponse(BaseModel):
    tables: List[TableInfo]
