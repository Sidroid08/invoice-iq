"""Vertex AI endpoint client for document-type classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config.settings import Settings

from invoice_iq.schemas.documents import DocumentType, OCRResult


@dataclass
class VertexClassifierClient:
    """Document-type predictor backed by a Vertex AI endpoint.

    The endpoint is expected to expose the Phase 7 `/predict` JSON contract:
    `{"instances": [{"text": "..."}]}` -> predictions with `document_type` and
    `confidence`.
    """

    project_id: str
    location: str
    endpoint_id: str
    client: Any | None = None

    @property
    def endpoint_name(self) -> str:
        return (
            f"projects/{self.project_id}/locations/{self.location}/"
            f"endpoints/{self.endpoint_id}"
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> VertexClassifierClient:
        if not settings.google_cloud_project:
            raise RuntimeError("GOOGLE_CLOUD_PROJECT is required for CLASSIFIER_BACKEND=vertex")
        if not settings.vertex_classifier_endpoint_id:
            raise RuntimeError(
                "VERTEX_CLASSIFIER_ENDPOINT_ID is required for CLASSIFIER_BACKEND=vertex"
            )
        return cls(
            project_id=settings.google_cloud_project,
            location=settings.google_cloud_region,
            endpoint_id=settings.vertex_classifier_endpoint_id,
        )

    def _client(self) -> Any:  # noqa: ANN401 - Google client is optional/untyped here
        if self.client is None:
            try:
                from google.cloud import aiplatform_v1  # noqa: PLC0415
            except ImportError as exc:  # pragma: no cover - only without [gcp]
                raise RuntimeError("VertexClassifierClient requires the '[gcp]' extra") from exc
            self.client = aiplatform_v1.PredictionServiceClient(
                client_options={"api_endpoint": f"{self.location}-aiplatform.googleapis.com"}
            )
        return self.client

    def predict(self, text: str) -> tuple[DocumentType, float]:
        if not text.strip():
            return DocumentType.UNKNOWN, 0.0
        response = self._client().predict(
            request={
                "endpoint": self.endpoint_name,
                "instances": [{"text": text}],
            }
        )
        prediction = _first_prediction(response)
        return _parse_document_prediction(prediction)

    def predict_ocr(self, ocr: OCRResult) -> tuple[DocumentType, float]:
        return self.predict(ocr.full_text)


def _first_prediction(response: Any) -> dict[str, Any]:  # noqa: ANN401
    predictions = getattr(response, "predictions", None)
    if predictions is None and isinstance(response, dict):
        predictions = response.get("predictions")
    if not predictions:
        raise RuntimeError("Vertex classifier response contained no predictions")
    first = predictions[0]
    if isinstance(first, dict):
        return first
    try:
        return dict(first)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"unsupported Vertex prediction payload: {first!r}") from exc


def _parse_document_prediction(payload: dict[str, Any]) -> tuple[DocumentType, float]:
    if "document_type" in payload:
        return DocumentType(str(payload["document_type"])), float(payload.get("confidence", 0.0))
    if "label" in payload:
        return DocumentType(str(payload["label"])), float(payload.get("confidence", 0.0))
    classes = payload.get("classes")
    scores = payload.get("scores")
    if isinstance(classes, list) and isinstance(scores, list) and classes and scores:
        best = max(range(len(scores)), key=lambda i: float(scores[i]))
        return DocumentType(str(classes[best])), float(scores[best])
    raise RuntimeError(f"unsupported Vertex classifier prediction shape: {payload!r}")
