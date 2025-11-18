"""
Updated config.py to include database configuration
"""
import os


class Config:
    # Existing configuration
    model_name = "google/flan-t5-large"
    embedding_model = "all-MiniLM-L6-v2"

    # Document processing
    chunk_size = 1000
    chunk_overlap = 200

    # Paths
    upload_folder = "data/uploads"
    index_folder = "data/indices"

    # Query settings
    k_results = 5
    k_per_collection = 3
    relevance_threshold = 0.3

    # Database configuration
    database_url = os.getenv(
        "DATABASE_URL",
        "postgresql://username:password@localhost:5432/pdf_qa"
    )

    # Hybrid system settings
    structured_weight = 0.6
    unstructured_weight = 0.4
    max_hybrid_results = 10

    # PostgreSQL connection settings
    db_pool_size = 5
    db_max_overflow = 10
    db_pool_recycle = 3600


config = Config()
