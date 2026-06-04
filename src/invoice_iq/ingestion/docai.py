"""GCP Document AI OCR provider (lazy, opt-in Phase 7 swap-in)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config.settings import Settings

from invoice_iq.schemas.documents import OCRResult


@dataclass
class DocAIOCRProvider:
    """Extract text with a Document AI processor.

    The Google SDK is imported only when no test client is injected. No network
    call happens until `extract` is invoked.
    """

    project_id: str
    location: str
    processor_id: str
    client: Any | None = None
    mime_type: str = "application/pdf"

    @property
    def name(self) -> str:
        return "gcp_docai"

    @property
    def processor_name(self) -> str:
        return (
            f"projects/{self.project_id}/locations/{self.location}/"
            f"processors/{self.processor_id}"
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> DocAIOCRProvider:
        if not settings.google_cloud_project:
            raise RuntimeError("GOOGLE_CLOUD_PROJECT is required for OCR_PROVIDER=gcp")
        if not settings.docai_processor_id:
            raise RuntimeError("DOCAI_PROCESSOR_ID is required for OCR_PROVIDER=gcp")
        return cls(
            project_id=settings.google_cloud_project,
            location=settings.google_cloud_region,
            processor_id=settings.docai_processor_id,
        )

    def _client(self) -> Any:  # noqa: ANN401 - Google client is optional/untyped here
        if self.client is None:
            try:
                from google.cloud import documentai_v1  # noqa: PLC0415
            except ImportError as exc:  # pragma: no cover - only without [gcp]
                raise RuntimeError("DocAIOCRProvider requires the '[gcp]' extra") from exc
            self.client = documentai_v1.DocumentProcessorServiceClient()
        return self.client

    def extract(self, pdf_path: Path, doc_id: str) -> OCRResult:
        path = Path(pdf_path)
        request = {
            "name": self.processor_name,
            "raw_document": {
                "content": path.read_bytes(),
                "mime_type": self.mime_type,
            },
        }
        response = self._client().process_document(request=request)
        document = response.document
        full_text = str(getattr(document, "text", "") or "")
        pages = self._page_texts(document, full_text)
        if not pages and full_text:
            pages = [full_text]
        confidence = self._mean_confidence(document)
        return OCRResult(
            doc_id=doc_id,
            full_text=full_text or "\f".join(pages),
            pages=pages,
            provider=self.name,
            confidence=confidence,
        )

    @classmethod
    def _page_texts(cls, document: Any, full_text: str) -> list[str]:  # noqa: ANN401
        pages: list[str] = []
        for page in getattr(document, "pages", []) or []:
            page_text = cls._layout_text(getattr(page, "layout", None), full_text)
            if page_text:
                pages.append(page_text)
        return pages

    @staticmethod
    def _layout_text(layout: Any, full_text: str) -> str:  # noqa: ANN401
        anchor = getattr(layout, "text_anchor", None)
        segments = getattr(anchor, "text_segments", None) or []
        pieces: list[str] = []
        for segment in segments:
            start = int(getattr(segment, "start_index", 0) or 0)
            end = int(getattr(segment, "end_index", 0) or 0)
            pieces.append(full_text[start:end])
        return "".join(pieces).strip()

    @staticmethod
    def _mean_confidence(document: Any) -> float | None:  # noqa: ANN401
        values = [
            float(confidence)
            for page in (getattr(document, "pages", []) or [])
            if (confidence := getattr(getattr(page, "layout", None), "confidence", None))
            is not None
        ]
        if not values:
            return None
        return round(sum(values) / len(values), 4)
