"""Tests for typed application settings (Phase 1)."""

from __future__ import annotations

import pytest
from config.settings import Settings


def _settings(**overrides: object) -> Settings:
    # _env_file=None disables .env loading so tests are hermetic.
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


def test_defaults_are_local_first() -> None:
    s = _settings()
    assert s.ocr_provider == "local"
    assert s.embedding_provider == "local"
    assert s.vector_store == "local"
    assert s.classifier_backend == "local"
    assert s.enable_llm_extraction is False


def test_llm_extraction_ready_requires_flag_and_key() -> None:
    assert _settings().llm_extraction_ready is False
    assert _settings(enable_llm_extraction=True).llm_extraction_ready is False
    ready = _settings(enable_llm_extraction=True, anthropic_api_key="sk-test")
    assert ready.llm_extraction_ready is True


def test_gcp_ready_reflects_project() -> None:
    assert _settings().gcp_ready is False
    assert _settings(google_cloud_project="my-proj").gcp_ready is True


def test_env_vars_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OCR_PROVIDER", "gcp")
    monkeypatch.setenv("API_PORT", "9001")
    s = Settings(_env_file=None)
    assert s.ocr_provider == "gcp"
    assert s.api_port == 9001


def test_invalid_provider_rejected() -> None:
    with pytest.raises(ValueError):
        _settings(ocr_provider="azure")


def test_port_out_of_range_rejected() -> None:
    with pytest.raises(ValueError):
        _settings(api_port=70000)
