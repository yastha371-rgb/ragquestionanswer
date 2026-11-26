import os
from typing import List, Union, Optional
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration for Academic Q&A RAG System."""
    
    # API Configuration
    API_TITLE: str = "Academic Q&A RAG System"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "Enterprise RAG system for academic Q&A powered by Google Gemini"
    DEBUG: bool = False
    
    # Free AI Provider: Google Gemini
    GEMINI_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""
    MODEL_NAME: str = "gemini-3.8-flash"
    EMBEDDING_MODEL: str = "models/gemini-embedding-001"
    
    # Database Configuration
    CHROMA_PATH: str = "./data/chroma_db"
    VECTOR_STORE_NAME: str = "academic_qa"
    
    # Redis Configuration
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    CACHE_TTL: int = 3600  # 1 hour
    
    # Document Configuration
    DOCUMENT_STORAGE_PATH: str = "./data/documents"
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_FILE_TYPES: Union[List[str], str] = ["pdf", "txt", "docx", "csv", "xlsx", "xls"]
    
    # RAG Configuration
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    RETRIEVAL_TOP_K: int = 5
    MIN_CONFIDENCE_SCORE: float = 0.5

    @property
    def api_key(self) -> str:
        """Returns the configured Gemini API key."""
        return self.GEMINI_API_KEY or self.GOOGLE_API_KEY or os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
    
    @field_validator("ALLOWED_FILE_TYPES", mode="before")
    @classmethod
    def parse_allowed_file_types(cls, v):
        if isinstance(v, str):
            return [ext.strip().lower() for ext in v.split(",") if ext.strip()]
        return [ext.strip().lower() for ext in v]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


settings = Settings()
