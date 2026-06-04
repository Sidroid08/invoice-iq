"""Chunking: turn an `Invoice` into retrievable, field-aware text chunks.

Field-aware chunks (summary / vendor / per-line-item) retrieve far better than
naively splitting raw text, because each chunk is a self-contained, queryable
fact carrying structured metadata for citation and filtering.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from invoice_iq.schemas.invoice import Invoice


class Chunk(BaseModel):
    """A unit of indexed text with citation metadata."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Unique chunk id, e.g. '<doc_id>:summary'.")
    text: str = Field(min_length=1)
    metadata: dict[str, str] = Field(default_factory=dict)


def _money(value: object) -> str:
    return str(value)


def chunk_invoice(invoice: Invoice, doc_id: str) -> list[Chunk]:
    """Produce field-aware chunks for one invoice."""
    vendor = invoice.vendor
    base_meta = {
        "doc_id": doc_id,
        "invoice_number": invoice.invoice_number,
        "vendor": vendor.name,
    }

    tax_str = _money(invoice.tax) if invoice.tax is not None else "0.00"
    summary = (
        f"Invoice {invoice.invoice_number} from {vendor.name} "
        f"dated {invoice.invoice_date.isoformat()}"
        + (f", due {invoice.due_date.isoformat()}" if invoice.due_date else "")
        + f". Subtotal {_money(invoice.subtotal)}, tax {tax_str}, "
        f"total {_money(invoice.total)}."
    )

    vendor_parts = [f"Vendor {vendor.name}"]
    if vendor.address:
        vendor_parts.append(f"address {vendor.address}")
    if vendor.tax_id:
        vendor_parts.append(f"tax id {vendor.tax_id}")
    if vendor.email:
        vendor_parts.append(f"email {vendor.email}")
    vendor_text = ", ".join(vendor_parts) + "."

    chunks = [
        Chunk(id=f"{doc_id}:summary", text=summary, metadata={**base_meta, "kind": "summary"}),
        Chunk(id=f"{doc_id}:vendor", text=vendor_text, metadata={**base_meta, "kind": "vendor"}),
    ]
    for i, item in enumerate(invoice.line_items):
        text = (
            f"Line item on invoice {invoice.invoice_number} from {vendor.name}: "
            f"{item.description}, quantity {item.quantity} at {item.unit_price} each, "
            f"line total {item.line_total}."
        )
        chunks.append(
            Chunk(id=f"{doc_id}:item:{i}", text=text, metadata={**base_meta, "kind": "line_item"})
        )
    return chunks


def chunk_text(text: str, *, size: int = 400, overlap: int = 50) -> list[str]:
    """Sliding-window chunker for free-form text (non-invoice documents)."""
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")
    words = text.split()
    if not words:
        return []
    step = size - overlap
    chunks: list[str] = []
    for start in range(0, len(words), step):
        window = words[start : start + size]
        chunks.append(" ".join(window))
        if start + size >= len(words):
            break
    return chunks
