"""
Database models for structured data
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, Float, ForeignKey, Boolean, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database_sqlite import Base
import uuid


class Document(Base):
    """Documents table for metadata and relationships"""
    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String(255), nullable=False)
    file_name = Column(String(255), nullable=False)
    file_type = Column(String(50), nullable=False)  # 'pdf', 'chat', 'structured'
    collection_id = Column(String, nullable=False, index=True)
    upload_date = Column(DateTime(timezone=True), server_default=func.now())
    file_size = Column(Integer)
    page_count = Column(Integer)
    summary = Column(Text)
    tags = Column(JSON)  # For flexible tagging

    # Relationships
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    qa_pairs = relationship("QAPair", back_populates="document", cascade="all, delete-orphan")


class DocumentChunk(Base):
    """Text chunks with embeddings for similarity search"""
    __tablename__ = "document_chunks"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    page_number = Column(Integer)
    chunk_type = Column(String(50), default='content')  # 'content', 'title', 'metadata'
    embedding_vector = Column(JSON)  # Store as JSON array for now

    # Relationships
    document = relationship("Document", back_populates="chunks")


class QAPair(Base):
    """Question-Answer pairs for training and evaluation"""
    __tablename__ = "qa_pairs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, ForeignKey("documents.id"), nullable=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    context = Column(Text, nullable=True)
    confidence_score = Column(Float)
    human_validated = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    document = relationship("Document", back_populates="qa_pairs")


class ChatSession(Base):
    """Chat sessions for conversation history"""
    __tablename__ = "chat_sessions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_name = Column(String(255))
    user_id = Column(String(100), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_activity = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base):
    """Individual chat messages"""
    __tablename__ = "chat_messages"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(String(20), nullable=False)  # 'user', 'assistant', 'system'
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    message_metadata = Column(JSON)  # For additional context

    # Relationships
    session = relationship("ChatSession", back_populates="messages")


class BusinessEntity(Base):
    """Structured business entities (example)"""
    __tablename__ = "business_entities"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False, index=True)
    entity_type = Column(String(100), nullable=False)  # 'company', 'product', 'person', etc.
    description = Column(Text)
    attributes = Column(JSON)  # Flexible attributes
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class Relationship(Base):
    """Entity relationships"""
    __tablename__ = "relationships"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    source_entity_id = Column(String, nullable=False)
    target_entity_id = Column(String, nullable=False)
    relationship_type = Column(String(100), nullable=False)
    confidence = Column(Float, default=1.0)
    relationship_metadata = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
