"""Typed, env-driven configuration with local-first defaults.

All runtime knobs live here as a single validated `Settings` object. Defaults
keep the system fully local and free; GCP/LLM paths are opt-in via env vars (see
`.env.example`). Nothing here reads secrets at import time beyond what pydantic
pulls from the environment / `.env`.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderMode = Literal["local", "gcp"]
ClassifierBackend = Literal["local", "vertex"]


class Settings(BaseSettings):
    """Application settings, populated from environment variables / `.env`.

    Field names map to UPPER_SNAKE env vars (case-insensitive), e.g.
    `OCR_PROVIDER` -> `ocr_provider`.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Provider selection (local-first) ---
    ocr_provider: ProviderMode = "local"
    embedding_provider: ProviderMode = "local"
    vector_store: ProviderMode = "local"
    classifier_backend: ClassifierBackend = "local"

    # --- RAG (Phase 4) ---
    # Pinned so CI and local machines fetch the identical embedding model.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rag_top_k: int = Field(default=4, ge=1, le=50)

    # --- Optional LLM extraction path (Phase 2) ---
    enable_llm_extraction: bool = False
    anthropic_api_key: str | None = None

    # --- GCP / Vertex AI (Phase 7) ---
    google_cloud_project: str | None = None
    google_cloud_region: str = "us-central1"
    google_application_credentials: str | None = None
    vertex_classifier_endpoint_id: str | None = None
    docai_processor_id: str | None = None

    # --- Serving (Phase 6) ---
    # Binds all interfaces — intended for containerized serving (Cloud Run).
    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = "INFO"
    max_upload_mb: int = Field(default=10, ge=1, le=100)

    # --- Storage paths ---
    model_dir: str = "models"
    chroma_dir: str = "chroma"
    data_dir: str = "data"

    @property
    def llm_extraction_ready(self) -> bool:
        """True only when the optional LLM path is enabled *and* a key is set."""
        return self.enable_llm_extraction and bool(self.anthropic_api_key)

    @property
    def gcp_ready(self) -> bool:
        """True when a GCP project is configured (Phase 7 swap-ins)."""
        return bool(self.google_cloud_project)


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide cached Settings instance."""
    return Settings()
