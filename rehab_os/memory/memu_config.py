"""
RehabOS — memU Service Configuration

Configures memU MemoryService for patient session memory persistence.
Uses local LLM (Ollama on DGX) and PostgreSQL with pgvector.

Environment variables (REHAB_MEMU_ prefix):
  REHAB_MEMU_POSTGRES_DSN    - PostgreSQL+pgvector connection string
  REHAB_MEMU_LLM_BASE_URL    - OpenAI-compatible LLM endpoint (Ollama)
  REHAB_MEMU_LLM_MODEL       - Chat model name
  REHAB_MEMU_EMBED_MODEL     - Embedding model name
  REHAB_MEMU_EMBED_BASE_URL  - Embedding endpoint
  REHAB_MEMU_LLM_API_KEY     - API key for LLM (default: "ollama")
"""
from __future__ import annotations

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Database (PostgreSQL + pgvector)
# ---------------------------------------------------------------------------
REHAB_MEMU_POSTGRES_DSN: str = os.getenv(
    "REHAB_MEMU_POSTGRES_DSN",
    "postgresql+psycopg://postgres:postgres@192.168.68.127:5432/memu",
)

# ---------------------------------------------------------------------------
# LLM (Ollama on DGX, OpenAI-compatible)
# ---------------------------------------------------------------------------
REHAB_MEMU_LLM_BASE_URL: str = os.getenv(
    "REHAB_MEMU_LLM_BASE_URL",
    "http://192.168.68.127:11434/v1",
)
REHAB_MEMU_LLM_API_KEY: str = os.getenv("REHAB_MEMU_LLM_API_KEY", "ollama")
REHAB_MEMU_LLM_MODEL: str = os.getenv("REHAB_MEMU_LLM_MODEL", "qwen3-next:80b")

# Embeddings
REHAB_MEMU_EMBED_BASE_URL: str = os.getenv(
    "REHAB_MEMU_EMBED_BASE_URL",
    REHAB_MEMU_LLM_BASE_URL,
)
REHAB_MEMU_EMBED_MODEL: str = os.getenv("REHAB_MEMU_EMBED_MODEL", "nomic-embed-text")

# ---------------------------------------------------------------------------
# Clinical memory categories (rehab-specific)
# ---------------------------------------------------------------------------
REHAB_MEMU_MEMORY_CATEGORIES = [
    {
        "name": "consultation_history",
        "description": "Prior consultation results: diagnoses, plans, confidence scores",
    },
    {
        "name": "treatment_outcomes",
        "description": "Outcome measure results and functional progress over time",
    },
    {
        "name": "patient_preferences",
        "description": "Patient preferences for treatment, scheduling, communication",
    },
    {
        "name": "clinical_observations",
        "description": "Clinician observations, objective findings across visits",
    },
    {
        "name": "functional_progress",
        "description": "Functional status changes: mobility, ADL, cognition trends",
    },
    {
        "name": "discharge_planning",
        "description": "Discharge criteria, readiness indicators, transition plans",
    },
]


def build_memu_service_kwargs() -> dict:
    """Return kwargs dict for ``MemoryService(**kwargs)``."""
    return {
        "llm_profiles": {
            "default": {
                "base_url": REHAB_MEMU_LLM_BASE_URL,
                "api_key": REHAB_MEMU_LLM_API_KEY,
                "chat_model": REHAB_MEMU_LLM_MODEL,
                "embed_model": REHAB_MEMU_EMBED_MODEL,
                "client_backend": "sdk",
            },
            "embedding": {
                "base_url": REHAB_MEMU_EMBED_BASE_URL,
                "api_key": REHAB_MEMU_LLM_API_KEY,
                "chat_model": REHAB_MEMU_LLM_MODEL,
                "embed_model": REHAB_MEMU_EMBED_MODEL,
                "client_backend": "sdk",
            },
        },
        "database_config": {
            "metadata_store": {
                "provider": "postgres",
                "dsn": REHAB_MEMU_POSTGRES_DSN,
                "ddl_mode": "create",
            },
        },
        "memorize_config": {
            "memory_categories": REHAB_MEMU_MEMORY_CATEGORIES,
        },
        "retrieve_config": {
            "method": "rag",
        },
    }
