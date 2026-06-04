"""Shared pytest fixtures."""

from __future__ import annotations

from datetime import date

import pytest

from invoice_iq.schemas import Invoice, LineItem, Money, Vendor


def money(amount: str, currency: str = "USD") -> Money:
    return Money(amount=amount, currency=currency)


@pytest.fixture
def valid_invoice() -> Invoice:
    """A fully consistent two-line invoice used across schema tests."""
    return Invoice(
        invoice_number="INV-2026-0042",
        vendor=Vendor(name="Acme Corp", email="billing@acme.example"),
        invoice_date=date(2026, 4, 1),
        due_date=date(2026, 5, 1),
        currency="USD",
        line_items=[
            LineItem(
                description="Consulting",
                quantity=10,
                unit_price=money("100.00"),
                line_total=money("1000.00"),
            ),
            LineItem(
                description="License",
                quantity=2,
                unit_price=money("50.00"),
                line_total=money("100.00"),
            ),
        ],
        subtotal=money("1100.00"),
        tax=money("110.00"),
        total=money("1210.00"),
    )
