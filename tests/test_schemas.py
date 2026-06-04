"""Tests for the Pydantic schema contracts (Phase 1)."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from invoice_iq.schemas import (
    DocumentType,
    Invoice,
    LineItem,
    Money,
    OCRResult,
    RawDocument,
    Vendor,
)


def usd(amount: str) -> Money:
    return Money(amount=amount, currency="USD")


# --------------------------------------------------------------------------- #
# DocumentType
# --------------------------------------------------------------------------- #
def test_document_type_values() -> None:
    assert {d.value for d in DocumentType} == {"invoice", "receipt", "contract", "unknown"}
    assert DocumentType("invoice") is DocumentType.INVOICE


# --------------------------------------------------------------------------- #
# RawDocument
# --------------------------------------------------------------------------- #
def test_raw_document_defaults_and_id() -> None:
    doc = RawDocument(filename="acme.pdf")
    assert doc.content_type == "application/pdf"
    assert len(doc.doc_id) == 32  # uuid4 hex
    assert doc.ingested_at.tzinfo is not None  # timezone-aware


def test_raw_document_blank_filename_rejected() -> None:
    with pytest.raises(ValidationError):
        RawDocument(filename="   ")


def test_raw_document_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        RawDocument(filename="a.pdf", bogus="x")  # type: ignore[call-arg]


# --------------------------------------------------------------------------- #
# OCRResult
# --------------------------------------------------------------------------- #
def test_ocr_result_page_count() -> None:
    r = OCRResult(doc_id="d1", full_text="hello", pages=["hello", "world"])
    assert r.page_count == 2


def test_ocr_result_empty_text_rejected() -> None:
    with pytest.raises(ValidationError):
        OCRResult(doc_id="d1", full_text="   ")


def test_ocr_confidence_bounds() -> None:
    with pytest.raises(ValidationError):
        OCRResult(doc_id="d1", full_text="x", confidence=1.5)


# --------------------------------------------------------------------------- #
# Money
# --------------------------------------------------------------------------- #
def test_money_quantizes_and_normalizes_currency() -> None:
    m = Money(amount="100.1", currency="usd")
    assert m.amount == Decimal("100.10")
    assert m.currency == "USD"


def test_money_rejects_negative() -> None:
    with pytest.raises(ValidationError):
        Money(amount="-1.00", currency="USD")


@pytest.mark.parametrize("bad", ["US", "USDD", "12$", "U2D"])
def test_money_rejects_bad_currency(bad: str) -> None:
    with pytest.raises(ValidationError):
        Money(amount="1.00", currency=bad)


def test_money_is_frozen() -> None:
    m = Money(amount="1.00", currency="USD")
    with pytest.raises(ValidationError):
        m.amount = Decimal("2.00")  # type: ignore[misc]


def test_money_str() -> None:
    assert str(Money(amount="9.5", currency="eur")) == "9.50 EUR"


# --------------------------------------------------------------------------- #
# Vendor
# --------------------------------------------------------------------------- #
def test_vendor_blank_name_rejected() -> None:
    with pytest.raises(ValidationError):
        Vendor(name="  ")


def test_vendor_invalid_email_rejected() -> None:
    with pytest.raises(ValidationError):
        Vendor(name="Acme", email="not-an-email")


# --------------------------------------------------------------------------- #
# LineItem
# --------------------------------------------------------------------------- #
def test_line_item_valid_math() -> None:
    li = LineItem(
        description="Widget",
        quantity=Decimal("3"),
        unit_price=usd("4.00"),
        line_total=usd("12.00"),
    )
    assert li.line_total.amount == Decimal("12.00")


def test_line_item_wrong_total_rejected() -> None:
    with pytest.raises(ValidationError, match="does not match line_total"):
        LineItem(
            description="Widget",
            quantity=Decimal("3"),
            unit_price=usd("4.00"),
            line_total=usd("13.00"),
        )


def test_line_item_currency_mismatch_rejected() -> None:
    with pytest.raises(ValidationError, match="currency"):
        LineItem(
            description="Widget",
            quantity=Decimal("1"),
            unit_price=usd("4.00"),
            line_total=Money(amount="4.00", currency="EUR"),
        )


def test_line_item_zero_quantity_rejected() -> None:
    with pytest.raises(ValidationError):
        LineItem(
            description="Widget",
            quantity=Decimal("0"),
            unit_price=usd("4.00"),
            line_total=usd("0.00"),
        )


# --------------------------------------------------------------------------- #
# Invoice
# --------------------------------------------------------------------------- #
def test_valid_invoice_builds(valid_invoice: Invoice) -> None:
    assert valid_invoice.total.amount == Decimal("1210.00")
    assert valid_invoice.document_type is DocumentType.INVOICE
    assert len(valid_invoice.line_items) == 2


def test_invoice_json_round_trip(valid_invoice: Invoice) -> None:
    rebuilt = Invoice.model_validate_json(valid_invoice.model_dump_json())
    assert rebuilt == valid_invoice


def test_invoice_subtotal_mismatch_rejected(valid_invoice: Invoice) -> None:
    data = valid_invoice.model_dump()
    # Break subtotal vs. line totals, but keep total == subtotal+tax so the
    # subtotal-vs-lines check is the one that fires.
    data["subtotal"] = {"amount": "999.00", "currency": "USD"}
    data["total"] = {"amount": "1109.00", "currency": "USD"}
    with pytest.raises(ValidationError, match="subtotal"):
        Invoice.model_validate(data)


def test_invoice_total_mismatch_rejected(valid_invoice: Invoice) -> None:
    data = valid_invoice.model_dump()
    data["total"] = {"amount": "9999.00", "currency": "USD"}
    with pytest.raises(ValidationError, match="total"):
        Invoice.model_validate(data)


def test_invoice_currency_mismatch_rejected(valid_invoice: Invoice) -> None:
    data = valid_invoice.model_dump()
    data["tax"] = {"amount": "110.00", "currency": "EUR"}
    with pytest.raises(ValidationError, match="currency"):
        Invoice.model_validate(data)


def test_invoice_due_before_invoice_date_rejected(valid_invoice: Invoice) -> None:
    data = valid_invoice.model_dump()
    data["due_date"] = date(2026, 3, 1)
    with pytest.raises(ValidationError, match="due_date"):
        Invoice.model_validate(data)


def test_invoice_requires_at_least_one_line_item(valid_invoice: Invoice) -> None:
    data = valid_invoice.model_dump()
    data["line_items"] = []
    with pytest.raises(ValidationError):
        Invoice.model_validate(data)


def test_invoice_without_tax_reconciles() -> None:
    inv = Invoice(
        invoice_number="INV-1",
        vendor=Vendor(name="Acme"),
        invoice_date=date(2026, 1, 1),
        currency="USD",
        line_items=[
            LineItem(
                description="Item",
                quantity=Decimal("1"),
                unit_price=usd("50.00"),
                line_total=usd("50.00"),
            )
        ],
        subtotal=usd("50.00"),
        total=usd("50.00"),
    )
    assert inv.tax is None
    assert inv.total.amount == Decimal("50.00")
