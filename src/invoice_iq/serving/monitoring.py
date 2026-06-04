"""Small in-process metrics collector for the FastAPI service."""

from __future__ import annotations

import logging
from collections import Counter
from threading import Lock

from pydantic import BaseModel, ConfigDict, Field

from invoice_iq.schemas.documents import DocumentType

logger = logging.getLogger(__name__)


class MetricsSnapshot(BaseModel):
    """Serializable view returned by `/metrics`."""

    model_config = ConfigDict(extra="forbid")

    total_requests: int = Field(ge=0)
    total_errors: int = Field(ge=0)
    error_rate: float = Field(ge=0.0, le=1.0)
    avg_latency_ms: float = Field(ge=0.0)
    endpoint_counts: dict[str, int]
    endpoint_errors: dict[str, int]
    prediction_counts: dict[str, int]
    vector_count: int | None = Field(default=None, ge=0)


class MetricsCollector:
    """Thread-safe, process-local request and prediction counters."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._total_requests = 0
        self._total_errors = 0
        self._latency_sum_ms = 0.0
        self._endpoint_counts: Counter[str] = Counter()
        self._endpoint_errors: Counter[str] = Counter()
        self._prediction_counts: Counter[str] = Counter()

    def record_request(
        self, *, method: str, path: str, status_code: int, latency_ms: float
    ) -> None:
        """Record one HTTP request after a response has been produced."""
        key = f"{method.upper()} {path}"
        is_error = status_code >= 400
        with self._lock:
            self._total_requests += 1
            self._latency_sum_ms += latency_ms
            self._endpoint_counts[key] += 1
            if is_error:
                self._total_errors += 1
                self._endpoint_errors[key] += 1
        logger.info(
            "request method=%s path=%s status=%s latency_ms=%.2f",
            method.upper(),
            path,
            status_code,
            latency_ms,
        )

    def record_prediction(self, document_type: DocumentType) -> None:
        """Record the classifier's predicted class distribution."""
        with self._lock:
            self._prediction_counts[document_type.value] += 1
        logger.info("prediction document_type=%s", document_type.value)

    def snapshot(self, *, vector_count: int | None = None) -> MetricsSnapshot:
        """Return a consistent metrics snapshot."""
        with self._lock:
            avg_latency = (
                self._latency_sum_ms / self._total_requests if self._total_requests else 0.0
            )
            error_rate = self._total_errors / self._total_requests if self._total_requests else 0.0
            return MetricsSnapshot(
                total_requests=self._total_requests,
                total_errors=self._total_errors,
                error_rate=error_rate,
                avg_latency_ms=round(avg_latency, 3),
                endpoint_counts=dict(self._endpoint_counts),
                endpoint_errors=dict(self._endpoint_errors),
                prediction_counts=dict(self._prediction_counts),
                vector_count=vector_count,
            )
