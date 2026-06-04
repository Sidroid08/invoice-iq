"""Ingestion: turn source PDFs into text + light layout (`OCRResult`)."""

from invoice_iq.ingestion.base import OCRProvider, build_raw_document
from invoice_iq.ingestion.local_ocr import LocalOCRProvider, ingest_pdf

__all__ = [
    "LocalOCRProvider",
    "OCRProvider",
    "build_raw_document",
    "ingest_pdf",
]
