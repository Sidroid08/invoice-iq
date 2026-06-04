"""Vertex AI text embedding provider."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from config.settings import Settings

TaskType = Literal["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]


@dataclass
class VertexEmbedder:
    """Embedding provider backed by Vertex AI publisher models."""

    project_id: str
    location: str
    model_name: str = "text-embedding-005"
    client: Any | None = None
    _dimension: int | None = None

    @property
    def name(self) -> str:
        return "vertex"

    @property
    def model_endpoint(self) -> str:
        return (
            f"projects/{self.project_id}/locations/{self.location}/"
            f"publishers/google/models/{self.model_name}"
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> VertexEmbedder:
        if not settings.google_cloud_project:
            raise RuntimeError("GOOGLE_CLOUD_PROJECT is required for EMBEDDING_PROVIDER=gcp")
        return cls(
            project_id=settings.google_cloud_project,
            location=settings.google_cloud_region,
            model_name=settings.vertex_embedding_model,
        )

    def _client(self) -> Any:  # noqa: ANN401 - Google client is optional/untyped here
        if self.client is None:
            try:
                from google.cloud import aiplatform_v1  # noqa: PLC0415
            except ImportError as exc:  # pragma: no cover - only without [gcp]
                raise RuntimeError("VertexEmbedder requires the '[gcp]' extra") from exc
            self.client = aiplatform_v1.PredictionServiceClient(
                client_options={"api_endpoint": f"{self.location}-aiplatform.googleapis.com"}
            )
        return self.client

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = len(self.embed_query("dimension probe"))
        return self._dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, task_type="RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], task_type="RETRIEVAL_QUERY")[0]

    def _embed(self, texts: list[str], *, task_type: TaskType) -> list[list[float]]:
        response = self._client().predict(
            request={
                "endpoint": self.model_endpoint,
                "instances": [
                    {"content": text, "task_type": task_type}
                    for text in texts
                ],
            }
        )
        predictions = getattr(response, "predictions", None)
        if predictions is None and isinstance(response, dict):
            predictions = response.get("predictions")
        if predictions is None:
            raise RuntimeError("Vertex embedding response contained no predictions")
        return [_parse_embedding(prediction) for prediction in predictions]


def _parse_embedding(payload: Any) -> list[float]:  # noqa: ANN401
    if not isinstance(payload, dict):
        try:
            payload = dict(payload)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"unsupported Vertex embedding payload: {payload!r}") from exc
    embedding = payload.get("embeddings", payload)
    values = embedding.get("values") if isinstance(embedding, dict) else None
    if not isinstance(values, list):
        raise RuntimeError(f"unsupported Vertex embedding shape: {payload!r}")
    return [float(value) for value in values]
