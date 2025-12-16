from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from typing import List, Optional
from enum import Enum


# Whitelist of allowed table names (security: prevents SQL injection)
ALLOWED_TABLES = frozenset(["user_profiles", "products", "orders"])


class LLMProvider(str, Enum):
    """Supported LLM providers (FREE only)"""
    HUGGINGFACE = "huggingface"  # Local, free, default
    OLLAMA = "ollama"            # Local, free, requires Ollama installed
    GEMINI = "gemini"            # Cloud, free tier (60 req/min)


# Available models per provider
AVAILABLE_MODELS = {
    LLMProvider.HUGGINGFACE: [
        "google/flan-t5-base",      # Default, smaller
        "google/flan-t5-large",     # Better quality, more RAM
        "google/flan-t5-xl",        # Best quality, needs GPU
    ],
    LLMProvider.OLLAMA: [
        "qwen2.5:7b",               # Good for Asian languages
        "llama3.2",                 # General purpose
        "mistral",                  # Fast, good quality
        "phi3",                     # Small but capable
    ],
    LLMProvider.GEMINI: [
        "gemini-1.5-flash",         # Fast, free tier
        "gemini-1.5-pro",           # Better quality
    ],
}


class Config(BaseSettings):
    # Default LLM Provider (used if no parameter sent)
    llm_provider: LLMProvider = Field(default=LLMProvider.HUGGINGFACE)
    
    # HuggingFace settings (local - default)
    model_name: str = Field(default="google/flan-t5-base")
    
    # Ollama settings (local)
    ollama_model: str = Field(default="qwen2.5:7b")
    ollama_base_url: str = Field(default="http://localhost:11434")
    
    # Gemini settings (cloud - free tier)
    gemini_api_key: Optional[str] = Field(default=None)
    gemini_model: str = Field(default="gemini-1.5-flash")
    
    # Common LLM settings
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