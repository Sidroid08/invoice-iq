"""FastAPI endpoint tests (Phase 6)."""

from __future__ import annotations

import random
from pathlib import Path
from uuid import uuid4

import pytest
from config.settings import Settings
from fastapi.testclient import TestClient

from invoice_iq.classifier import LABELS, Predictor
from invoice_iq.classifier.train import TrainConfig, train_classifier
from invoice_iq.extraction import RuleBasedExtractor
from invoice_iq.rag import ChromaVectorStore, HashingEmbedder, RAGPipeline
from invoice_iq.schemas.invoice import Invoice
from invoice_iq.serving.app import create_app
from invoice_iq.serving.deps import AppDeps
from invoice_iq.synthetic import random_contract_lines, render_invoice_pdf, render_text_pdf


@pytest.fixture(scope="module")
def predictor() -> Predictor:
    model, vocab, _ = train_classifier(
        TrainConfig(n_train_per_class=20, n_test_per_class=5, epochs=10, seed=11)
    )
    return Predictor(model=model, vocab=vocab, labels=list(LABELS))


@pytest.fixture
def client(tmp_path: Path, predictor: Predictor) -> TestClient:
    settings = Settings(chroma_dir=str(tmp_path / "chroma"), model_dir=str(tmp_path / "models"))
    store = ChromaVectorStore(collection_name=f"api_{uuid4().hex[:8]}")
    deps = AppDeps(
        settings=settings,
        predictor=predictor,
        extractor=RuleBasedExtractor(),
        rag=RAGPipeline(HashingEmbedder(), store),
    )
    return TestClient(create_app(deps))


def _invoice_pdf(tmp_path: Path, invoice: Invoice) -> Path:
    pdf = tmp_path / "invoice.pdf"
    render_invoice_pdf(pdf, invoice)
    return pdf


def _contract_pdf(tmp_path: Path) -> Path:
    pdf = tmp_path / "contract.pdf"
    render_text_pdf(pdf, random_contract_lines(random.Random(3)))
    return pdf


def _upload(path: Path) -> dict[str, tuple[str, bytes, str]]:
    return {"file": (path.name, path.read_bytes(), "application/pdf")}


def test_health_reports_local_providers(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["providers"]["ocr"] == "local"
    assert body["providers"]["extractor"] == "rule_based"
    assert body["vector_count"] == 0


def test_ingest_returns_raw_document_and_ocr(
    client: TestClient, tmp_path: Path, valid_invoice: Invoice
) -> None:
    response = client.post("/ingest", files=_upload(_invoice_pdf(tmp_path, valid_invoice)))
    assert response.status_code == 200
    body = response.json()
    assert body["raw_document"]["filename"] == "invoice.pdf"
    assert body["ocr"]["provider"] == "local"
    assert "INVOICE" in body["ocr"]["full_text"]


def test_classify_invoice(client: TestClient, tmp_path: Path, valid_invoice: Invoice) -> None:
    response = client.post("/classify", files=_upload(_invoice_pdf(tmp_path, valid_invoice)))
    assert response.status_code == 200
    body = response.json()
    assert body["document_type"] == "invoice"
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["page_count"] == 1


def test_extract_invoice(client: TestClient, tmp_path: Path, valid_invoice: Invoice) -> None:
    response = client.post("/extract", files=_upload(_invoice_pdf(tmp_path, valid_invoice)))
    assert response.status_code == 200
    body = response.json()
    assert body["extractor"] == "rule_based"
    assert body["invoice"]["invoice_number"] == valid_invoice.invoice_number
    assert body["invoice"]["total"]["amount"] == "1210.00"


def test_ask_indexes_invoice_and_returns_answer(
    client: TestClient, tmp_path: Path, valid_invoice: Invoice
) -> None:
    response = client.post(
        "/ask",
        files=_upload(_invoice_pdf(tmp_path, valid_invoice)),
        data={"question": "What is the total?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert "1210.00" in body["answer"]["answer"]
    assert body["answer"]["sources"]


def test_agent_invoice_flow(client: TestClient, tmp_path: Path, valid_invoice: Invoice) -> None:
    response = client.post(
        "/agent",
        files=_upload(_invoice_pdf(tmp_path, valid_invoice)),
        data={"question": "What is the total?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["document_type"] == "invoice"
    assert body["recommendation"] == "approve"
    assert body["answer"] is not None


def test_agent_routes_non_invoice(client: TestClient, tmp_path: Path) -> None:
    response = client.post("/agent", files=_upload(_contract_pdf(tmp_path)))
    assert response.status_code == 200
    body = response.json()
    assert body["document_type"] == "contract"
    assert body["recommendation"] == "route_non_invoice"
    assert body["invoice"] is None


def test_extract_contract_returns_422(client: TestClient, tmp_path: Path) -> None:
    response = client.post("/extract", files=_upload(_contract_pdf(tmp_path)))
    assert response.status_code == 422
    assert "missing required field" in response.json()["detail"]


def test_rejects_non_pdf_upload(client: TestClient) -> None:
    response = client.post(
        "/ingest",
        files={"file": ("note.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "only PDF uploads are supported"


def test_metrics_tracks_requests_and_predictions(
    client: TestClient, tmp_path: Path, valid_invoice: Invoice
) -> None:
    pdf = _invoice_pdf(tmp_path, valid_invoice)
    assert client.post("/classify", files=_upload(pdf)).status_code == 200
    assert client.post("/agent", files=_upload(pdf)).status_code == 200

    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.json()
    assert body["total_requests"] >= 2
    assert body["prediction_counts"]["invoice"] >= 2
    assert body["error_rate"] == 0.0
