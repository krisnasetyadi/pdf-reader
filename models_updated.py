"""
Updated models.py to include hybrid response model
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to ask")
    collection_id: Optional[str] = Field(None, description="Specific collection to search")
    include_sources: bool = Field(True, description="Include source information")
    session_id: Optional[str] = Field(None, description="Chat session ID for history tracking")


class QAResponse(BaseModel):
    answer: str
    sources: List[str] = []
    collection_id: Optional[str] = None
    processing_time: float


class HybridQAResponse(BaseModel):
    """Enhanced response model for hybrid queries"""
    answer: str
    sources: List[str] = []
    intent_classification: Dict[str, float]
    structured_results_count: int
    unstructured_results_count: int
    total_results_count: int
    processing_time: float
    collection_id: Optional[str] = None


class UploadResponse(BaseModel):
    collection_id: str
    file_count: int
    status: str


class CollectionInfo(BaseModel):
    collection_id: str
    file_count: int
    last_modified: Optional[datetime] = None


class EntityCreateRequest(BaseModel):
    """Request model for creating business entities"""
    name: str = Field(..., min_length=1)
    entity_type: str = Field(..., min_length=1)
    description: Optional[str] = None
    attributes: Optional[Dict[str, Any]] = None


class EntityResponse(BaseModel):
    """Response model for business entities"""
    id: str
    name: str
    entity_type: str
    description: Optional[str] = None
    attributes: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: Optional[datetime] = None


class ChatSessionRequest(BaseModel):
    """Request model for chat sessions"""
    session_name: Optional[str] = None


class ChatSessionResponse(BaseModel):
    """Response model for chat sessions"""
    id: str
    name: str
    created_at: datetime
    last_activity: Optional[datetime] = None


class ChatMessageRequest(BaseModel):
    """Request model for chat messages"""
    content: str = Field(..., min_length=1)
    role: str = Field(..., pattern="^(user|assistant|system)$")
    metadata: Optional[Dict[str, Any]] = None


class ChatMessageResponse(BaseModel):
    """Response model for chat messages"""
    id: str
    role: str
    content: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None
