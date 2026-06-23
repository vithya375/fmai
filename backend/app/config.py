"""
Configuration module for the RAG backend.
Loads settings from environment variables via a .env file.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Mistral AI Configuration
    mistral_api_key: str = ""

    # Model Configuration
    chat_model: str = "mistral-large-latest"
    embedding_model: str = "mistral-embed"
    audio_model: str = "voxtral-mini-latest"

    # Embedding Dimensions (mistral-embed produces 1024-dim vectors)
    embedding_dimension: int = 1024

    # Chunking Configuration
    chunk_size: int = 2000
    chunk_overlap: int = 200

    # Retrieval Configuration
    top_k: int = 5

    # ChromaDB Configuration
    chroma_collection_name: str = "rag_documents"
    chroma_persist_directory: str = "./chroma_db/"

    # Upload Configuration
    upload_directory: str = "./uploads/"

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000

    # CORS Configuration
    cors_origins: list[str] = ["*"]

    # Frontend directory (relative to backend)
    frontend_directory: str = "../frontend/"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


@lru_cache()
def get_settings() -> Settings:
    """
    Returns a cached instance of Settings.
    Uses lru_cache to avoid re-reading .env on every call.
    """
    return Settings()
