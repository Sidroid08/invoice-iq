"""Tests for the Streamlit demo's API client and view helpers (Phase 8).

Hermetic: the client talks to an in-process FastAPI app via a TestClient (which
subclasses httpx.Client). No network, no model download beyond the tiny trained
classifier, no paid API key.
"""

from __future__ import annotations

import random
from pathlib import Path
from uuid import uuid4

import pytest
from config.settings import Settings
from fastapi.testclient import TestClient

from invoice_iq.agent.state import NextAction
from invoice_iq.classifier import LABELS, Predictor
from invoice_iq.classifier.train import TrainConfig, train_classifier
from invoice_iq.extraction import RuleBasedExtractor
from invoice_iq.rag import ChromaVectorStore, HashingEmbedder, RAGPipeline
from invoice_iq.schemas.invoice import Invoice
from invoice_iq.serving.app import create_app
from invoice_iq.serving.deps import AppDeps
from invoice_iq.synthetic import random_contract_lines, render_invoice_pdf, render_text_pdf
from invoice_iq.ui import AgentAPIClient, invoice_line_item_rows, recommendation_badge


@pytest.fixture(scope="module")
def predictor() -> Predictor:
    model, vocab, _ = train_classifier(
        TrainConfig(n_train_per_class=20, n_test_per_class=5, epochs=10, seed=13)
    )
    return Predictor(model=model, vocab=vocab, labels=list(LABELS))


@pytest.fixture
def api_client(tmp_path: Path, predictor: Predictor) -> AgentAPIClient:
    settings = Settings(chroma_dir=str(tmp_path / "chroma"), model_dir=str(tmp_path / "models"))
    store = ChromaVectorStore(collection_name=f"ui_{uuid4().hex[:8]}")
    deps = AppDeps(
        settings=settings,
        predictor=predictor,
        extractor=RuleBasedExtractor(),
        rag=RAGPipeline(HashingEmbedder(), store),
    )
    return AgentAPIClient(session=TestClient(create_app(deps)))


def _pdf_bytes(path: Path) -> bytes:
    return path.read_bytes()


# --------------------------------------------------------------------------- #
# Client (against the real app)
# --------------------------------------------------------------------------- #
def test_health(api_client: AgentAPIClient) -> None:
    health = api_client.health()
    assert health["status"] == "ok"
    assert health["providers"]["ocr"] == "local"


def test_run_agent_invoice(
    api_client: AgentAPIClient, tmp_path: Path, valid_invoice: Invoice
) -> None:
    pdf = tmp_path / "inv.pdf"
    render_invoice_pdf(pdf, valid_invoice)
    result = api_client.run_agent(_pdf_bytes(pdf), "inv.pdf", question="What is the total?")

    assert result.document_type.value == "invoice"
    assert result.invoice is not None
    assert result.recommendation is NextAction.APPROVE
    assert result.answer is not None
    assert "1210.00" in result.answer.answer


def test_run_agent_non_invoice(api_client: AgentAPIClient, tmp_path: Path) -> None:
    pdf = tmp_path / "contract.pdf"
    render_text_pdf(pdf, random_contract_lines(random.Random(5)))
    result = api_client.run_agent(_pdf_bytes(pdf), "contract.pdf")
    assert result.document_type.value == "contract"
    assert result.invoice is None
    assert result.recommendation is NextAction.ROUTE_NON_INVOICE


# --------------------------------------------------------------------------- #
# View helpers (pure)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("action", list(NextAction))
def test_recommendation_badge_valid_style(action: NextAction) -> None:
    label, style = recommendation_badge(action)
    assert label
    assert style in {"success", "warning", "info", "error"}


def test_invoice_line_item_rows(valid_invoice: Invoice) -> None:
    rows = invoice_line_item_rows(valid_invoice)
    assert len(rows) == len(valid_invoice.line_items)
    assert set(rows[0]) == {"Description", "Quantity", "Unit price", "Line total"}
    assert rows[0]["Description"] == valid_invoice.line_items[0].description
