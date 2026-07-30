# models.py
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from datetime import datetime

class SearchType(str, Enum):
    UNSTRUCTURED = "unstructured"
    STRUCTURED = "structured"
    HYBRID = "hybrid"

class ChatPlatform(str, Enum):
    WHATSAPP = "whatsapp"
    TEAMS = "teams"
    SLACK = "slack"
    TELEGRAM = "telegram"
    GENERIC = "generic"

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
    file_names: Optional[List[str]] = None
    title: Optional[str] = None


class UploadFromUrlRequest(BaseModel):
    url: str
    title: Optional[str] = None


class UploadFromUrlsRequest(BaseModel):
    urls: List[str]
    title: Optional[str] = None


class DriveFolderRequest(BaseModel):
    url: str
    recursive: Optional[bool] = True
    max_depth: Optional[int] = 5


class DriveFolderItem(BaseModel):
    id: str
    name: str
    url: str
    item_type: str


class DriveFolderItemsResponse(BaseModel):
    folder_id: str
    files: List[DriveFolderItem]
    folders: List[DriveFolderItem]
    count: int


class CreatePublicLinkRequest(BaseModel):
    title: Optional[str] = None
    url: str
    item_urls: Optional[List[str]] = None


class SetPublicLinkActiveRequest(BaseModel):
    link_id: str
    active: bool


class PublicLinkItem(BaseModel):
    id: str
    name: str
    url: str
    item_type: str


class PublicLinkSource(BaseModel):
    link_id: str
    workspace_id: Optional[str] = None
    title: str
    url: str
    status: str
    item_count: int
    created_at: Union[str, datetime]
    items: List[PublicLinkItem] = []


class PublicLinksResponse(BaseModel):
    links: List[PublicLinkSource]
    count: int


class CreateDatabaseConnectionRequest(BaseModel):
    label: Optional[str] = None
    url: str


class SetDatabaseConnectionActiveRequest(BaseModel):
    connection_id: str
    active: bool


class DbColumnInfo(BaseModel):
    name: str
    type: str
    nullable: bool = True
    primary_key: bool = False


class DbTableInfo(BaseModel):
    name: str
    row_count: Optional[int] = None
    columns: List[DbColumnInfo] = []


class DatabaseConnectionSource(BaseModel):
    connection_id: str
    workspace_id: Optional[str] = None
    label: str
    url: str
    status: str
    table_count: int
    created_at: Union[str, datetime]
    tables: List[DbTableInfo] = []


class DatabaseConnectionsResponse(BaseModel):
    connections: List[DatabaseConnectionSource]
    count: int


class QAResponse(BaseModel):
    answer: str
    sources: List[str]  # Now includes collection IDs
    collection_id: str   # "all_collections" when searching globally
    processing_time: float
    search_type: SearchType


class CollectionInfo(BaseModel):
    collection_id: str
    document_count: int
    created_at: Union[str, datetime]
    file_names: List[str]
    title: Optional[str] = None
    status: str = "active"
    owner_id: Optional[str] = None


class SetPdfCollectionActiveRequest(BaseModel):
    collection_id: str
    active: bool

class DatabaseResult(BaseModel):
    table: str
    data: List[Dict[str, Any]]
    record_count: int
    avg_relevance_score: Optional[float] = None  # Average score dari search results


class PdfSourceInfo(BaseModel):
    """PDF source with URL for direct access"""
    file_name: str
    collection_id: str
    page: Optional[int] = None
    relevance_score: Optional[float] = None
    content_preview: Optional[str] = None
    file_url: Optional[str] = None  # URL to access the PDF
    page_url: Optional[str] = None  # URL with page parameter for direct jump
    search_text: Optional[str] = None  # Text snippet for highlighting/searching in PDF viewer


# models.py - Add these fields to HybridQueryRequest
class HybridQueryRequest(BaseModel):
    question: str
    collection_id: Optional[str] = None  # DEPRECATED: use pdf_collection_ids instead
    
    # Collection Selection (Optional - if not provided, searches all)
    pdf_collection_ids: Optional[List[str]] = None  # Specific PDF collections to search
    chat_collection_ids: Optional[List[str]] = None  # Specific chat collections to search
    public_link_ids: Optional[List[str]] = None  # Specific Public Link sources to search
    
    include_pdf_results: bool = True
    include_db_results: bool = True
    include_chat_results: bool = True  # NEW: Search in chat logs?
    include_public_links: bool = False
    source_mode: Optional[str] = None  # "pdf" | "chat" | "database" | "public_link" | "mixed" | "none"
    
    # LLM Selection (optional - defaults to config if not provided)
    llm_provider: Optional[str] = None  # "huggingface", "ollama", "gemini"
    llm_model: Optional[str] = None     # specific model name

class HybridResponse(BaseModel):
    answer: str
    pdf_sources: List[str]  # Keep for backward compatibility
    pdf_sources_detailed: Optional[List[PdfSourceInfo]] = None  # NEW: Detailed PDF sources with URLs
    db_results: Dict[str, Any]
    chat_results: Optional[List[Dict[str, Any]]] = None  # NEW: Chat search results
    processing_time: float
    search_terms: List[str]
    target_tables: Optional[List[str]] = None  # Tables that were searched (smart routing)
    
    # Model info - tells user which model generated this response
    model_used: str  # e.g., "huggingface/google/flan-t5-base"
    available_models: Optional[Dict[str, List[str]]] = None  # Only returned on first request or error

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


# ===================== CHAT MODELS =====================

class ChatMessage(BaseModel):
    """Single chat message parsed from export file"""
    message_id: Optional[str] = None
    sender: str
    timestamp: datetime
    content: str
    platform: ChatPlatform = ChatPlatform.WHATSAPP
    thread_id: Optional[str] = None
    conversation_id: Optional[str] = None
    raw_line: Optional[str] = None  # Original line for debugging


class ChatCollection(BaseModel):
    """Metadata for an uploaded chat collection"""
    collection_id: str
    platform: ChatPlatform
    file_name: str
    message_count: int
    date_range: Optional[Dict[str, str]] = None  # {"start": ..., "end": ...}
    participants: List[str]
    created_at: datetime
    status: str = "active"


class SetChatCollectionActiveRequest(BaseModel):
    collection_id: str
    active: bool


class ChatUploadResponse(BaseModel):
    """Response after uploading chat file"""
    collection_id: str
    file_name: str
    platform: str
    message_count: int
    participants: List[str]
    date_range: Optional[Dict[str, str]] = None
    status: str


class ChatSearchResult(BaseModel):
    """Single result from chat search"""
    message: ChatMessage
    relevance_score: float
    context_messages: Optional[List[ChatMessage]] = None  # Surrounding messages for context

# models.py - tambahkan enhanced response model
class EnhancedHybridResponse(BaseModel):
    answer: str
    answer_metadata: Dict[str, Any]
    pdf_sources: List[str]
    pdf_sources_detailed: Optional[List[PdfSourceInfo]] = None
    db_results: Dict[str, Any]
    chat_results: Optional[List[Dict[str, Any]]] = None
    processing_time: float
    search_analysis: Dict[str, Any]
    merged_results_preview: Optional[List[Dict[str, Any]]] = None
    conflicts: Optional[List[Dict[str, Any]]] = None
    model_used: str
    confidence_score: float  # Overall confidence 0-1