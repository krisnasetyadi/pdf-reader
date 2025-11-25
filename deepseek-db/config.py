# config.py
import os
from dataclasses import dataclass


@dataclass
class Config:
    # Existing config...
    upload_folder: str = "uploads"
    index_folder: str = "indexes"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    model_name: str = "google/flan-t5-base"
    max_new_tokens: int = 512
    temperature: float = 0.3
    k_per_collection: int = 3
    total_k_results: int = 5

    # Database configuration
    db_host: str = os.getenv("DB_HOST", "localhost")
    db_port: str = os.getenv("DB_PORT", "5432")
    db_name: str = os.getenv("DB_NAME", "qa_system")
    db_user: str = os.getenv("DB_USER", "postgres")
    db_password: str = os.getenv("DB_PASSWORD", "password")

    # Database tables to include (you can modify this based on your schema)
    db_tables: list = None

    def __post_init__(self):
        if self.db_tables is None:
            self.db_tables = ["user_profiles", "products", "orders"]  # Example tables


config = Config()
