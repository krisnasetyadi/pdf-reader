# models.py
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from enum import Enum

class SearchType(str, Enum):
    UNSTRUCTURED = "unstructured"
    STRUCTURED = "structured"
    HYBRID = "hybrid"

class QueryIntent(str, Enum):
    COUNT = "count"
    SEARCH = "search"
    LIST = "list"
    AGGREGATE = "aggregate"
    SHOW_TABLES = "show_tables" 
    SHOW_SCHEMA = "show_schema"
    HELP = "help"
    UNKNOWN = "unknown"

class QueryRequest(BaseModel):
    question: str
    collection_id: Optional[str] = None  # Now optional
    include_sources: bool = True
    # search_type: SearchType = SearchType.UNSTRUCTURED
class StructuredQueryRequest(BaseModel):
    question: str
    table_name: Optional[str] = None


class UploadResponse(BaseModel):
    collection_id: str
    file_count: int
    status: str


class QAResponse(BaseModel):
    answer: str
    sources: List[str]  # Now includes collection IDs
    collection_id: str   # "all_collections" when searching globally
    processing_time: float
    search_type: SearchType


class CollectionInfo(BaseModel):
    collection_id: str
    document_count: int
    created_at: str
    file_names: List[str]

class DatabaseResult(BaseModel):
    table: str
    data: List[Dict[str, Any]]
    record_count: int
    avg_relevance_score: Optional[float] = None  # Average score dari search results

# models.py - Add these fields to HybridQueryRequest
class HybridQueryRequest(BaseModel):
    question: str
    collection_id: Optional[str] = None
    include_pdf_results: bool = True
    include_db_results: bool = True
    # collection_id: Optional[str] = None  # Optional: search specific PDF collection
    # include_pdf_results: bool = True     # Search in PDFs? 
    # include_db_results: bool = True      # Search in database?
    # db_result_limit: int = 5             # Limit DB results
    
    # # New: Control search behavior
    # search_all_collections: bool = True  # If no collection_id, search all PDFs?

class HybridResponse(BaseModel):
    answer: str
    pdf_sources: List[str]
    db_results: Dict[str, Any]
    processing_time: float
    search_terms: List[str]
    target_tables: Optional[List[str]] = None  # Tables that were searched (smart routing)

    # answer: str
    # pdf_sources: List[str]
    # db_results: Dict[str, DatabaseResult]  # FIXED: Dict bukan List
    # processing_time: float
    # search_terms: List[str]

class SourceInfo(BaseModel):
    type: str
    source: str
    confidence: Optional[float] = None
    preview: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class StructuredQueryResponse(BaseModel):
    answer: str
    data: List[Dict[str, Any]]
    intent: QueryIntent
    table_used: str
    sql_query: Optional[str] = None
    processing_time: float

class JoinQueryRequest(BaseModel):
    question: str
    join_type: Optional[str] = "auto"  # auto, inner, left, cross
    limit: int = 20

class JoinQueryResponse(BaseModel):
    answer: str
    data: List[Dict[str, Any]]
    tables_used: List[str]
    join_conditions: List[Dict[str, str]]
    sql_query: Optional[str] = None
    processing_time: float

class TableRelationship(BaseModel):
    table1: str
    table2: str
    join_condition: str
    relationship_type: str 