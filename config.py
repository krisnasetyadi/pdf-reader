from pydantic_settings import BaseSettings
from pydantic import Field
import os
from dataclasses import dataclass


@dataclass
class Config(BaseSettings):
    # Gunakan model yang tersedia dan bekerja dengan baik
    model_name: str = Field(default="google/flan-t5-base")
    embedding_model: str = Field(default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    max_new_tokens: int = Field(default=256)
    temperature: float = Field(default=0.3)

    # File Paths
    upload_folder: str = Field(default="data/uploads")
    index_folder: str = Field(default="data/indices")

    # Retrieval Parameters
    chunk_size: int = Field(default=600)
    chunk_overlap: int = Field(default=100)
    k_results: int = Field(default=5)
    k_per_collection: int = Field(default=3)
    total_k_results: int = Field(default=8)

    class Config:
        env_file = ".env"
        extra = 'ignore'


config = Config()
