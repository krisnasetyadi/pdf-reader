from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List


class Config(BaseSettings):
    # Model settings
    model_name: str = Field(default="google/flan-t5-base")
    embedding_model: str = Field(default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    max_new_tokens: int = Field(default=256)
    temperature: float = Field(default=0.3)

    # File Paths - PDF
    upload_folder: str = Field(default="data/uploads")
    index_folder: str = Field(default="data/indices")
    
    # File Paths - Chat Logs
    chat_upload_folder: str = Field(default="data/chat_uploads")
    chat_index_folder: str = Field(default="data/chat_indices")

    # Retrieval Parameters
    chunk_size: int = Field(default=600)
    chunk_overlap: int = Field(default=100)
    k_results: int = Field(default=5)
    k_per_collection: int = Field(default=3)
    total_k_results: int = Field(default=8)
    
    # Chat-specific retrieval
    chat_chunk_size: int = Field(default=10)  # Messages per chunk
    chat_chunk_overlap: int = Field(default=2)  # Overlapping messages

    # Database Configuration
    db_host: str = Field(default="localhost")
    db_port: int = Field(default=5432)
    db_name: str = Field(default="qa_system")
    db_user: str = Field(default="postgres")
    db_password: str = Field(default="qwerty123")
    db_tables: List[str] = Field(default=["user_profiles", "products", "orders"])
    
    # Hybrid Search
    enable_hybrid_search: bool = Field(default=True)
    db_result_limit: int = Field(default=5)
    min_similarity_score: float = Field(default=0.75)
    
    # Supported chat platforms
    supported_chat_platforms: List[str] = Field(default=["whatsapp"])

    class Config:
        env_file = ".env"
        extra = 'ignore'


config = Config()