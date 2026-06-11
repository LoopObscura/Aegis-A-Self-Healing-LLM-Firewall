import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Production-grade configuration using Pydantic v2 BaseSettings.
    All sensitive keys are loaded from environment variables with secure defaults.
    """

    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    GUARDRAIL_THRESHOLD: float = float(os.getenv("GUARDRAIL_THRESHOLD", "0.85"))
    CACHE_SIMILARITY_THRESHOLD: float = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.92"))
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "production")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    VECTOR_COLLECTION_NAME: str = os.getenv("VECTOR_COLLECTION_NAME", "sentinel_cache")
    VECTOR_DIMENSION: int = int(os.getenv("VECTOR_DIMENSION", "1536"))
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "30"))

    class Config:
        env_file: str = ".env"
        case_sensitive: bool = True
        extra: str = "forbid"

    def validate_production_settings(self) -> bool:
        """
        Validate that all critical production settings are configured.
        Returns True if all checks pass.
        """
        if self.ENVIRONMENT == "production":
            if not self.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY must be set in production environment")
            if not self.QDRANT_API_KEY:
                raise ValueError("QDRANT_API_KEY must be set in production environment")
        return True


settings: Settings = Settings()
settings.validate_production_settings()
