"""Invoice schema: the validated, structured output of extraction.

This is the contract every downstream stage (RAG, agent, API) depends on. The
validators encode real invoice business rules — currency consistency, line-item
math, and total reconciliation — so a constructed `Invoice` is guaranteed
internally consistent, not merely well-typed.
"""

from __future__ import annotations

from datetime import date
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Annotated

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator, model_validator

from invoice_iq.schemas.documents import DocumentType

# Absolute tolerance (in currency minor units) for reconciliation checks.
# Real invoices accumulate rounding across line items; a strict equality would
# reject valid documents, so we allow a small slack that scales with line count.
_CENT = Decimal("0.01")
_PER_LINE_TOLERANCE = Decimal("0.01")


def _quantize(value: Decimal) -> Decimal:
    """Round a Decimal to 2 places (half-up), the convention for most currencies."""
    return value.quantize(_CENT, rounding=ROUND_HALF_UP)


# A non-negative monetary amount, stored exactly as Decimal (never float).
Amount = Annotated[Decimal, Field(max_digits=14, decimal_places=2, ge=0)]


class Money(BaseModel):
    """A currency-tagged amount. Immutable value object.

    Amounts are quantized to 2 decimal places and currency is normalized to an
    upper-case ISO-4217-style 3-letter code.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    amount: Amount
    currency: str = Field(min_length=3, max_length=3, description="ISO 4217 code, e.g. 'USD'.")

    @field_validator("amount", mode="before")
    @classmethod
    def _coerce_amount(cls, v: object) -> Decimal:
        try:
            dec = Decimal(str(v))
        except (InvalidOperation, ValueError) as exc:
            raise ValueError(f"invalid monetary amount: {v!r}") from exc
        return _quantize(dec)

    @field_validator("currency")
    @classmethod
    def _normalize_currency(cls, v: str) -> str:
        v = v.strip().upper()
        if len(v) != 3 or not v.isalpha():
            raise ValueError("currency must be a 3-letter ISO 4217 alphabetic code")
        return v

    def __str__(self) -> str:
        return f"{self.amount} {self.currency}"


class Vendor(BaseModel):
    """The party issuing the invoice."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, description="Vendor/supplier legal or trading name.")
    address: str | None = Field(default=None, description="Full postal address, if present.")
    tax_id: str | None = Field(default=None, description="VAT / tax registration number.")
    email: EmailStr | None = Field(default=None, description="Billing contact email, if present.")

    @field_validator("name")
    @classmethod
    def _strip_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("vendor name must not be blank")
        return v


class LineItem(BaseModel):
    """A single billed line. Enforces unit_price * quantity == line_total."""

    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=1)
    quantity: Decimal = Field(gt=0, description="Units billed (supports fractional, e.g. hours).")
    unit_price: Money
    line_total: Money

    @model_validator(mode="after")
    def _check_line_math(self) -> LineItem:
        if self.unit_price.currency != self.line_total.currency:
            raise ValueError(
                f"line '{self.description}': unit_price currency "
                f"{self.unit_price.currency} != line_total currency {self.line_total.currency}"
            )
        expected = _quantize(self.quantity * self.unit_price.amount)
        if abs(expected - self.line_total.amount) > _PER_LINE_TOLERANCE:
            raise ValueError(
                f"line '{self.description}': quantity*unit_price={expected} "
                f"does not match line_total={self.line_total.amount}"
            )
        return self


class Invoice(BaseModel):
    """A fully extracted, internally consistent invoice.

    Guarantees enforced after construction:
      * all amounts share the invoice's `currency`;
      * each line item's math is correct (see `LineItem`);
      * subtotal == sum(line totals) within tolerance;
      * total == subtotal + tax (tax optional) within tolerance;
      * due_date, if present, is on/after invoice_date.
    """

    model_config = ConfigDict(extra="forbid")

    document_type: DocumentType = Field(default=DocumentType.INVOICE)
    source_doc_id: str | None = Field(default=None, description="Links to RawDocument.doc_id.")

    invoice_number: str = Field(min_length=1)
    vendor: Vendor
    invoice_date: date
    due_date: date | None = None

    currency: str = Field(min_length=3, max_length=3)
    line_items: list[LineItem] = Field(min_length=1)
    subtotal: Money
    tax: Money | None = None
    total: Money

    extraction_confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("currency")
    @classmethod
    def _normalize_currency(cls, v: str) -> str:
        v = v.strip().upper()
        if len(v) != 3 or not v.isalpha():
            raise ValueError("currency must be a 3-letter ISO 4217 alphabetic code")
        return v

    @field_validator("invoice_number")
    @classmethod
    def _strip_number(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("invoice_number must not be blank")
        return v

    @model_validator(mode="after")
    def _check_consistency(self) -> Invoice:
        cur = self.currency

        # 1. Currency consistency across every monetary field.
        monies: list[tuple[str, Money]] = [
            ("subtotal", self.subtotal),
            ("total", self.total),
        ]
        if self.tax is not None:
            monies.append(("tax", self.tax))
        for name, money in monies:
            if money.currency != cur:
                raise ValueError(f"{name} currency {money.currency} != invoice currency {cur}")
        for i, item in enumerate(self.line_items):
            if item.line_total.currency != cur:
                raise ValueError(
                    f"line_items[{i}] currency {item.line_total.currency} != invoice currency {cur}"
                )

        # 2. Subtotal reconciles with line totals (tolerance scales with line count).
        line_sum = _quantize(sum((it.line_total.amount for it in self.line_items), Decimal("0")))
        line_tol = _PER_LINE_TOLERANCE * len(self.line_items)
        if abs(line_sum - self.subtotal.amount) > line_tol:
            raise ValueError(
                f"subtotal {self.subtotal.amount} != sum(line_totals) {line_sum} "
                f"(tolerance {line_tol})"
            )

        # 3. Total reconciles with subtotal + tax.
        tax_amount = self.tax.amount if self.tax is not None else Decimal("0")
        expected_total = _quantize(self.subtotal.amount + tax_amount)
        if abs(expected_total - self.total.amount) > _CENT:
            raise ValueError(f"total {self.total.amount} != subtotal+tax {expected_total}")

        # 4. Date ordering.
        if self.due_date is not None and self.due_date < self.invoice_date:
            raise ValueError(f"due_date {self.due_date} is before invoice_date {self.invoice_date}")
        return self
