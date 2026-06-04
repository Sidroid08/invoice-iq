"""Mock-tested GCP swap-in contracts (Phase 7).

These tests do not import Google SDKs, authenticate, or call live services.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from config.settings import Settings

from invoice_iq.ingestion.docai import DocAIOCRProvider
from invoice_iq.schemas.documents import DocumentType, OCRResult
from invoice_iq.serving.deps import build_app_deps
from invoice_iq.vertex.client import VertexClassifierClient
from invoice_iq.vertex.deploy_classifier import build_plan
from invoice_iq.vertex.embeddings import VertexEmbedder


class _FakePredictor:
    def predict(self, text: str) -> tuple[DocumentType, float]:
        return DocumentType.INVOICE, 0.9

    def predict_ocr(self, ocr: OCRResult) -> tuple[DocumentType, float]:
        return self.predict(ocr.full_text)


class _FakeDocAIClient:
    def __init__(self) -> None:
        self.request: dict[str, object] | None = None

    def process_document(self, *, request: dict[str, object]) -> SimpleNamespace:
        self.request = request
        text = "Page one text\nPage two text"
        page_1 = SimpleNamespace(
            layout=SimpleNamespace(
                confidence=0.8,
                text_anchor=SimpleNamespace(
                    text_segments=[SimpleNamespace(start_index=0, end_index=13)]
                ),
            )
        )
        page_2 = SimpleNamespace(
            layout=SimpleNamespace(
                confidence=1.0,
                text_anchor=SimpleNamespace(
                    text_segments=[SimpleNamespace(start_index=14, end_index=len(text))]
                ),
            )
        )
        document = SimpleNamespace(text=text, pages=[page_1, page_2])
        return SimpleNamespace(document=document)


class _FakePredictionClient:
    def __init__(self, prediction: dict[str, object]) -> None:
        self.prediction = prediction
        self.requests: list[dict[str, Any]] = []

    def predict(self, *, request: dict[str, Any]) -> SimpleNamespace:
        self.requests.append(request)
        return SimpleNamespace(predictions=[self.prediction])


class _FakeEmbeddingClient:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def predict(self, *, request: dict[str, Any]) -> SimpleNamespace:
        self.requests.append(request)
        instances = request["instances"]
        assert isinstance(instances, list)
        predictions = [
            {"embeddings": {"values": [float(i), float(i + 1)]}}
            for i, _ in enumerate(instances)
        ]
        return SimpleNamespace(predictions=predictions)


def test_docai_provider_builds_process_document_request(tmp_path: Path) -> None:
    pdf = tmp_path / "invoice.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    client = _FakeDocAIClient()
    provider = DocAIOCRProvider(
        project_id="proj",
        location="us",
        processor_id="processor-1",
        client=client,
    )

    ocr = provider.extract(pdf, doc_id="doc-1")

    assert client.request is not None
    assert client.request["name"] == "projects/proj/locations/us/processors/processor-1"
    assert ocr.provider == "gcp_docai"
    assert ocr.pages == ["Page one text", "Page two text"]
    assert ocr.confidence == pytest.approx(0.9)


def test_vertex_classifier_client_parses_prediction() -> None:
    client = _FakePredictionClient({"document_type": "invoice", "confidence": 0.97})
    predictor = VertexClassifierClient(
        project_id="proj", location="us-central1", endpoint_id="123", client=client
    )

    document_type, confidence = predictor.predict("Invoice text")

    assert document_type is DocumentType.INVOICE
    assert confidence == pytest.approx(0.97)
    assert client.requests[0]["endpoint"] == "projects/proj/locations/us-central1/endpoints/123"
    assert client.requests[0]["instances"] == [{"text": "Invoice text"}]


def test_vertex_classifier_client_parses_class_scores() -> None:
    client = _FakePredictionClient(
        {"classes": ["contract", "invoice", "receipt"], "scores": [0.1, 0.8, 0.1]}
    )
    predictor = VertexClassifierClient(
        project_id="proj", location="us-central1", endpoint_id="123", client=client
    )
    assert predictor.predict("Invoice text") == (DocumentType.INVOICE, 0.8)


def test_vertex_embedder_builds_predict_requests() -> None:
    client = _FakeEmbeddingClient()
    embedder = VertexEmbedder(
        project_id="proj", location="us-central1", model_name="text-embedding-005", client=client
    )

    vectors = embedder.embed_documents(["one", "two"])
    query = embedder.embed_query("question")

    assert vectors == [[0.0, 1.0], [1.0, 2.0]]
    assert query == [0.0, 1.0]
    assert client.requests[0]["endpoint"].endswith("/publishers/google/models/text-embedding-005")
    assert client.requests[0]["instances"][0]["task_type"] == "RETRIEVAL_DOCUMENT"
    assert client.requests[1]["instances"][0]["task_type"] == "RETRIEVAL_QUERY"


def test_vertex_deploy_plan_is_no_execute_and_uses_api_routes() -> None:
    plan = build_plan(
        project_id="proj",
        region="us-central1",
        image_uri="us-central1-docker.pkg.dev/proj/invoice-iq/api:latest",
    )
    commands = plan.commands()
    assert any("--container-predict-route=/predict" in command for command in commands)
    assert any("--container-health-route=/health" in command for command in commands)
    assert any("deploy-model" in command for command in commands)


def test_vector_search_flag_remains_cost_gated(tmp_path: Path) -> None:
    settings = Settings(vector_store="gcp", model_dir=str(tmp_path), chroma_dir=str(tmp_path))
    with pytest.raises(NotImplementedError, match="cost approval"):
        build_app_deps(settings=settings, predictor=_FakePredictor())
