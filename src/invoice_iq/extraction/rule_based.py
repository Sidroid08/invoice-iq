"""Deterministic, dependency-free invoice extraction from canonical text.

Parses the field labels produced by `invoice_iq.synthetic` (and any real invoice
that follows the same labelled convention). This is the always-on default path:
it needs no API key and no network, so the test-suite and CI never depend on a
paid service. The optional `LLMExtractor` handles messier, free-form layouts.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal, InvalidOperation

from invoice_iq.extraction.base import ExtractionError
from invoice_iq.schemas.documents import OCRResult
from invoice_iq.schemas.invoice import Invoice, LineItem, Money, Vendor
from invoice_iq.synthetic import (
    ITEM_PREFIX,
    ITEM_SEP,
    LABEL_ADDRESS,
    LABEL_CURRENCY,
    LABEL_DUE_DATE,
    LABEL_EMAIL,
    LABEL_INV_DATE,
    LABEL_NUMBER,
    LABEL_SUBTOTAL,
    LABEL_TAX,
    LABEL_TAX_ID,
    LABEL_TOTAL,
    LABEL_VENDOR,
)

# Optional fields that, when present, raise extraction confidence.
_OPTIONAL_LABELS = (LABEL_DUE_DATE, LABEL_ADDRESS, LABEL_TAX_ID, LABEL_EMAIL, LABEL_TAX)


class RuleBasedExtractor:
    """Parses labelled invoice text into a validated `Invoice`."""

    name = "rule_based"

    def extract(self, ocr: OCRResult) -> Invoice:
        lines = [ln.strip() for ln in ocr.full_text.replace("\f", "\n").splitlines()]
        lines = [ln for ln in lines if ln]

        currency = self._require(lines, LABEL_CURRENCY, "currency")
        invoice_number = self._require(lines, LABEL_NUMBER, "invoice_number")
        invoice_date = self._parse_date(self._require(lines, LABEL_INV_DATE, "invoice_date"))

        due_raw = self._find(lines, LABEL_DUE_DATE)
        due_date = self._parse_date(due_raw) if due_raw else None

        vendor = Vendor(
            name=self._require(lines, LABEL_VENDOR, "vendor"),
            address=self._find(lines, LABEL_ADDRESS),
            tax_id=self._find(lines, LABEL_TAX_ID),
            email=self._find(lines, LABEL_EMAIL),
        )

        items = self._parse_line_items(lines, currency)
        if not items:
            raise ExtractionError("no line items found")

        subtotal = self._money(self._require(lines, LABEL_SUBTOTAL, "subtotal"), currency)
        tax_raw = self._find(lines, LABEL_TAX)
        tax = self._money(tax_raw, currency) if tax_raw else None
        total = self._money(self._require(lines, LABEL_TOTAL, "total"), currency)

        confidence = self._confidence(lines)
        try:
            return Invoice(
                invoice_number=invoice_number,
                vendor=vendor,
                invoice_date=invoice_date,
                due_date=due_date,
                currency=currency,
                line_items=items,
                subtotal=subtotal,
                tax=tax,
                total=total,
                extraction_confidence=confidence,
            )
        except ValueError as exc:  # pydantic ValidationError is a ValueError
            raise ExtractionError(f"parsed fields failed validation: {exc}") from exc

    # ----------------------------- helpers ----------------------------- #
    @staticmethod
    def _find(lines: list[str], label: str) -> str | None:
        for ln in lines:
            if ln.startswith(label):
                return ln[len(label) :].strip()
        return None

    def _require(self, lines: list[str], label: str, field: str) -> str:
        value = self._find(lines, label)
        if value is None or not value:
            raise ExtractionError(f"missing required field: {field} (label '{label}')")
        return value

    @staticmethod
    def _parse_date(raw: str) -> date:
        try:
            return date.fromisoformat(raw)
        except ValueError as exc:
            raise ExtractionError(f"unparseable date: {raw!r}") from exc

    @staticmethod
    def _money(raw: str, currency: str) -> Money:
        cleaned = raw.replace("$", "").replace(",", "").strip()
        try:
            amount = Decimal(cleaned)
        except (InvalidOperation, ValueError) as exc:
            raise ExtractionError(f"unparseable amount: {raw!r}") from exc
        return Money(amount=amount, currency=currency)

    def _parse_line_items(self, lines: list[str], currency: str) -> list[LineItem]:
        items: list[LineItem] = []
        for ln in lines:
            if not ln.startswith(ITEM_PREFIX):
                continue
            rest = ln[len(ITEM_PREFIX) :].strip()
            # rsplit keeps any separators inside the description intact.
            parts = [p.strip() for p in rest.rsplit(ITEM_SEP, 3)]
            if len(parts) != 4:
                raise ExtractionError(f"malformed line item: {ln!r}")
            desc, qty_s, unit_s, total_s = parts
            try:
                quantity = Decimal(qty_s)
            except (InvalidOperation, ValueError) as exc:
                raise ExtractionError(f"unparseable quantity in {ln!r}") from exc
            items.append(
                LineItem(
                    description=desc,
                    quantity=quantity,
                    unit_price=self._money(unit_s, currency),
                    line_total=self._money(total_s, currency),
                )
            )
        return items

    @staticmethod
    def _confidence(lines: list[str]) -> float:
        present = sum(1 for label in _OPTIONAL_LABELS if any(ln.startswith(label) for ln in lines))
        # Base 0.8 for a parseable doc, +0.04 per optional field present (max 1.0).
        return round(min(1.0, 0.8 + 0.04 * present), 2)
