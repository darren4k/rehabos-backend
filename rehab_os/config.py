"""Configuration management for RehabOS."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Local LLM (DGX Spark)
    local_llm_base_url: str = Field(
        default="http://localhost:8000/v1",
        description="OpenAI-compatible endpoint for local LLM",
    )
    local_llm_model: str = Field(
        default="meta-llama/Llama-3.1-70B-Instruct",
        description="Model name for local LLM",
    )
    local_llm_timeout: int = Field(
        default=60,
        description="Timeout in seconds for local LLM requests",
    )
    local_llm_enabled: bool = Field(
        default=True,
        description="Whether to use local LLM as primary",
    )

    # Anthropic Claude (fallback)
    anthropic_api_key: str = Field(
        default="",
        description="Anthropic API key for Claude fallback",
    )
    anthropic_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Claude model to use",
    )

    # PubMed API
    pubmed_api_key: str = Field(
        default="",
        description="NCBI API key for PubMed access",
    )
    pubmed_email: str = Field(
        default="",
        description="Email for PubMed API (required by NCBI)",
    )

    # Qwen3-TTS Voice Server (DGX Spark)
    tts_server_url: str = Field(
        default="http://192.168.68.123:8080",
        description="URL for Qwen3-TTS server running on DGX Spark",
    )
    tts_enabled: bool = Field(
        default=True,
        description="Enable voice synthesis features",
    )

    # Vector Store
    chroma_persist_dir: Path = Field(
        default=Path("./data/chroma"),
        description="Directory for ChromaDB persistence",
    )

    # API Settings
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8080)
    api_key: str = Field(
        default="",
        description="API key for authenticating requests",
    )

    # CORS
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        description="Allowed CORS origins",
    )

    # memU Session Memory
    memu_enabled: bool = Field(
        default=True,
        description="Enable memU-backed patient session memory",
    )
    memu_postgres_dsn: str = Field(
        default="postgresql+psycopg://postgres:postgres@192.168.68.127:5432/memu",
        description="PostgreSQL DSN for memU storage",
    )
    memu_llm_base_url: str = Field(
        default="http://192.168.68.127:11434/v1",
        description="OpenAI-compatible LLM endpoint for memU",
    )
    memu_llm_model: str = Field(
        default="qwen3-next:80b",
        description="LLM model for memU memory extraction",
    )

    # Patient-Core Database
    core_database_url: str = Field(
        default="postgresql+psycopg://postgres:postgres@192.168.68.127:5432/rehab_core",
        description="PostgreSQL DSN for Patient-Core relational schema",
    )

    # Debug
    debug_mode: bool = Field(
        default=False,
        description="Enable debug mode (exposes error details in responses)",
    )

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")

    # Agent Settings
    max_retries: int = Field(default=3, description="Max retries for LLM calls")
    evidence_top_k: int = Field(default=5, description="Number of evidence items to retrieve")

    @property
    def has_anthropic_key(self) -> bool:
        """Check if Anthropic API key is configured."""
        return bool(self.anthropic_api_key)

    @property
    def has_pubmed_key(self) -> bool:
        """Check if PubMed API key is configured."""
        return bool(self.pubmed_api_key and self.pubmed_email)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
