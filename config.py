from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from typing import List


# Whitelist of allowed table names (security: prevents SQL injection)
ALLOWED_TABLES = frozenset(["user_profiles", "products", "orders"])


class Config(BaseSettings):
    # Model settings
    model_name: str = Field(...)
    embedding_model: str = Field(...)
    max_new_tokens: int = Field(...)
    temperature: float = Field(...)

    # File Paths - PDF
    upload_folder: str = Field(...)
    index_folder: str = Field(...)
    
    # File Paths - Chat Logs
    chat_upload_folder: str = Field(...)
    chat_index_folder: str = Field(...)

    # Retrieval Parameters
    chunk_size: int = Field(...)
    chunk_overlap: int = Field(...)
    k_results: int = Field(...)
    k_per_collection: int = Field(...)
    total_k_results: int = Field(...)
    
    # Chat-specific retrieval
    chat_chunk_size: int = Field(...)
    chat_chunk_overlap: int = Field(...)

    # Database Configuration
    db_host: str = Field(...)
    db_port: int = Field(...)
    db_name: str = Field(...)
    db_user: str = Field(...)
    db_password: str = Field(...)
    db_tables: List[str] = Field(default=["user_profiles", "products", "orders"])
    
    @field_validator('db_tables')
    @classmethod
    def validate_tables(cls, v):
        """Ensure only whitelisted tables are configured"""
        invalid_tables = set(v) - ALLOWED_TABLES
        if invalid_tables:
            raise ValueError(f"Invalid table names: {invalid_tables}. Allowed: {ALLOWED_TABLES}")
        return v
    
    # CORS Configuration
    cors_origins: str = Field(...)
    
    @property
    def cors_origin_list(self) -> List[str]:
        """Parse CORS origins from comma-separated string"""
        if self.cors_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]
    
    # Hybrid Search
    enable_hybrid_search: bool = Field(...)
    db_result_limit: int = Field(...)
    min_similarity_score: float = Field(...)
    
    # Supported chat platforms
    supported_chat_platforms: List[str] = Field(default=["whatsapp"])

    class Config:
        env_file = ".env"
        extra = 'ignore'


config = Config()