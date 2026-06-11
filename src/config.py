"""
Centralized configuration for the Adaptive Self-Healing RAG system.
All settings are loaded from environment variables (via .env file).
Every module imports Settings from here — never reads os.environ directly.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


def _require(key: str) -> str:
    """Read a required env var; raise with clear message if missing."""
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(
            f"Missing required environment variable: '{key}'\n"
            f"Copy .env.example to .env and fill in all values."
        )
    return value


def _get(key: str, default: str) -> str:
    """Read an optional env var with a fallback default."""
    return os.getenv(key, default)


@dataclass(frozen=True)
class Settings:
    """
    Immutable application configuration.
    frozen=True ensures no module can accidentally mutate global config at runtime.
    """

    # ── LLM ────────────────────────────────────────────────────────────────────
    groq_api_key: str
    groq_api_key_trad: str | None     # For Phase 9: Traditional RAG evaluator key
    groq_api_key_judge: str | None    # For Phase 9: LLM-as-a-judge key
    fast_model: str           # llama-3.1-8b-instant  — 14,400 RPD on Groq free tier
    power_model: str          # llama-3.3-70b-versatile — 1,000 RPD on Groq free tier

    # Daily budget guard: system warns at 80% and falls back to fast model at limit
    power_model_daily_budget: int

    # ── Vector Store ───────────────────────────────────────────────────────────
    chroma_persist_dir: str
    chroma_collection_name: str

    # ── Embeddings ─────────────────────────────────────────────────────────────
    embedding_model: str      # BAAI/bge-small-en-v1.5 — 384-dim, CPU-only

    # ── Self-Healing Retry Budgets ─────────────────────────────────────────────
    max_query_rewrites: int   # Max rewrite attempts before human escalation
    max_generation_retries: int  # Max regen attempts before human escalation

    # ── Retrieval ──────────────────────────────────────────────────────────────
    retrieval_top_k: int
    chunk_size: int
    chunk_overlap: int


def load_settings() -> Settings:
    """
    Load and validate all settings from environment variables.
    Called once at application startup via create_app_dependencies().
    """
    return Settings(
        groq_api_key=_require("GROQ_API_KEY"),
        groq_api_key_trad=_get("GROQ_API_KEY_TRAD", None),
        groq_api_key_judge=_get("GROQ_API_KEY_JUDGE", None),
        fast_model=_get("FAST_MODEL", "llama-3.1-8b-instant"),
        power_model=_get("POWER_MODEL", "llama-3.3-70b-versatile"),
        power_model_daily_budget=int(_get("POWER_MODEL_DAILY_BUDGET", "950")),
        chroma_persist_dir=_get("CHROMA_PERSIST_DIR", "./chroma_db"),
        chroma_collection_name=_get("CHROMA_COLLECTION_NAME", "ecommerce_support"),
        embedding_model=_get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        max_query_rewrites=int(_get("MAX_QUERY_REWRITES", "2")),
        max_generation_retries=int(_get("MAX_GENERATION_RETRIES", "2")),
        retrieval_top_k=int(_get("RETRIEVAL_TOP_K", "6")),
        chunk_size=int(_get("CHUNK_SIZE", "500")),
        chunk_overlap=int(_get("CHUNK_OVERLAP", "50")),
    )
