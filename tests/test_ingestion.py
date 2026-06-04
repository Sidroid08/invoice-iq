"""Tests for local ingestion / OCR (Phase 2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from invoice_iq.ingestion import LocalOCRProvider, build_raw_document, ingest_pdf
from invoice_iq.ingestion.base import OCRProvider
from invoice_iq.schemas.invoice import Invoice
from invoice_iq.synthetic import render_invoice_pdf


@pytest.fixture
def invoice_pdf(tmp_path: Path, valid_invoice: Invoice) -> Path:
    path = tmp_path / "acme.pdf"
    render_invoice_pdf(path, valid_invoice)
    return path


def test_local_provider_satisfies_protocol() -> None:
    assert isinstance(LocalOCRProvider(), OCRProvider)
    assert LocalOCRProvider().name == "local"


def test_build_raw_document(invoice_pdf: Path) -> None:
    raw = build_raw_document(invoice_pdf)
    assert raw.filename == "acme.pdf"
    assert raw.content_type == "application/pdf"
    assert raw.size_bytes > 0


def test_build_raw_document_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_raw_document(tmp_path / "nope.pdf")


def test_ingest_pdf_extracts_text(invoice_pdf: Path) -> None:
    raw, ocr = ingest_pdf(invoice_pdf)
    assert ocr.doc_id == raw.doc_id
    assert ocr.provider == "local"
    assert raw.num_pages == ocr.page_count >= 1
    # Key content survived the PDF -> text round-trip.
    assert "INVOICE" in ocr.full_text
    assert "Acme Corp" in ocr.full_text
    assert "Consulting" in ocr.full_text
    assert "1210.00" in ocr.full_text


def test_extract_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        LocalOCRProvider().extract(Path("does-not-exist.pdf"), doc_id="x")
