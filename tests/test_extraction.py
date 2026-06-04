"""Tests for invoice extraction (Phase 2)."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from invoice_iq.extraction import ExtractionError, RuleBasedExtractor
from invoice_iq.extraction.base import Extractor
from invoice_iq.extraction.llm_extractor import LLMExtractor
from invoice_iq.ingestion import ingest_pdf
from invoice_iq.schemas.documents import OCRResult
from invoice_iq.schemas.invoice import Invoice
from invoice_iq.synthetic import random_invoice, render_invoice_pdf

EXTRACTOR = RuleBasedExtractor()


def _ocr(text: str) -> OCRResult:
    return OCRResult(doc_id="t", full_text=text, pages=[text])


def _assert_same_invoice(parsed: Invoice, original: Invoice) -> None:
    assert parsed.invoice_number == original.invoice_number
    assert parsed.invoice_date == original.invoice_date
    assert parsed.due_date == original.due_date
    assert parsed.currency == original.currency
    assert parsed.vendor == original.vendor
    assert parsed.line_items == original.line_items
    assert parsed.subtotal == original.subtotal
    assert parsed.tax == original.tax
    assert parsed.total == original.total


def test_rule_based_satisfies_protocol() -> None:
    assert isinstance(EXTRACTOR, Extractor)
    assert EXTRACTOR.name == "rule_based"


def test_round_trip_fixture_invoice(tmp_path: Path, valid_invoice: Invoice) -> None:
    """Invoice -> PDF -> OCR -> Invoice preserves every field."""
    pdf = tmp_path / "inv.pdf"
    render_invoice_pdf(pdf, valid_invoice)
    _, ocr = ingest_pdf(pdf)
    parsed = EXTRACTOR.extract(ocr)
    _assert_same_invoice(parsed, valid_invoice)
    assert parsed.extraction_confidence is not None


@pytest.mark.parametrize("seed", [1, 7, 13, 99, 2026])
def test_round_trip_random_invoices(tmp_path: Path, seed: int) -> None:
    original = random_invoice(random.Random(seed))
    pdf = tmp_path / f"inv_{seed}.pdf"
    render_invoice_pdf(pdf, original)
    _, ocr = ingest_pdf(pdf)
    _assert_same_invoice(EXTRACTOR.extract(ocr), original)


def test_missing_required_field_raises() -> None:
    text = "INVOICE\nInvoice Number: INV-1\nVendor: Acme\n"  # no Currency/Total/etc.
    with pytest.raises(ExtractionError, match="missing required field"):
        EXTRACTOR.extract(_ocr(text))


def test_malformed_line_item_raises() -> None:
    text = (
        "INVOICE\nInvoice Number: INV-1\nInvoice Date: 2026-01-01\nCurrency: USD\n"
        "Vendor: Acme\nLine Items:\nItem: Bad item | 10\nSubtotal: 100.00\nTotal: 100.00\n"
    )
    with pytest.raises(ExtractionError, match="malformed line item"):
        EXTRACTOR.extract(_ocr(text))


def test_inconsistent_totals_raise_via_validation() -> None:
    text = (
        "INVOICE\nInvoice Number: INV-1\nInvoice Date: 2026-01-01\nCurrency: USD\n"
        "Vendor: Acme\nLine Items:\nItem: Widget | 1 | 100.00 | 100.00\n"
        "Subtotal: 999.00\nTotal: 999.00\n"
    )
    with pytest.raises(ExtractionError, match="failed validation"):
        EXTRACTOR.extract(_ocr(text))


# --------------------------------------------------------------------------- #
# LLM extractor — guards only (no live API call, no key needed)
# --------------------------------------------------------------------------- #
def test_llm_extractor_requires_key() -> None:
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        LLMExtractor(api_key="")


def test_llm_from_settings_requires_enabled() -> None:
    from config.settings import Settings

    settings = Settings(_env_file=None)
    with pytest.raises(RuntimeError, match="not enabled"):
        LLMExtractor.from_settings(settings)


def test_llm_from_settings_ready() -> None:
    from config.settings import Settings

    settings = Settings(_env_file=None, enable_llm_extraction=True, anthropic_api_key="sk-test")
    extractor = LLMExtractor.from_settings(settings)
    assert extractor.name == "llm"
