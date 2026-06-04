"""Ingestion interfaces shared by local and (future) Document AI providers."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from invoice_iq.schemas.documents import OCRResult, RawDocument

# Page-break sentinel used to join per-page text into `OCRResult.full_text`.
PAGE_BREAK = "\f"


@runtime_checkable
class OCRProvider(Protocol):
    """Extracts text + light layout from a PDF.

    Implementations: `LocalOCRProvider` (pdfplumber, Phase 2) and a future
    `DocAIProvider` (GCP Document AI, Phase 7). Both return the same `OCRResult`
    contract so the rest of the pipeline is provider-agnostic.
    """

    @property
    def name(self) -> str:
        """Short provider id recorded on `OCRResult.provider` (e.g. 'local')."""
        ...

    def extract(self, pdf_path: Path, doc_id: str) -> OCRResult:
        """Parse `pdf_path`, returning an `OCRResult` linked to `doc_id`."""
        ...


def build_raw_document(pdf_path: Path) -> RawDocument:
    """Construct a `RawDocument` (metadata only) from a file on disk."""
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"no such file: {path}")
    return RawDocument(
        filename=path.name,
        content_type="application/pdf",
        size_bytes=path.stat().st_size,
    )
