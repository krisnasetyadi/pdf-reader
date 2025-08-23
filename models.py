# models.py
from pydantic import BaseModel
from typing import List, Optional


class QueryRequest(BaseModel):
    question: str
    collection_id: Optional[str] = None  # Now optional
    include_sources: bool = True


class UploadResponse(BaseModel):
    collection_id: str
    file_count: int
    status: str


class QAResponse(BaseModel):
    answer: str
    sources: List[str]  # Now includes collection IDs
    collection_id: str   # "all_collections" when searching globally
    processing_time: float


class CollectionInfo(BaseModel):
    collection_id: str
    document_count: int
    created_at: str
    file_names: List[str]