"""Synthetic document generation: realistic invoice / receipt / contract PDFs.

This module is the *single source of truth* for the canonical on-page text layout.
The rule-based extractor parses exactly this layout, and the same `render_*`
functions are used by both the corpus generator (`scripts/generate_synthetic_data.py`)
and the test-suite — so the generator and extractor can never silently drift.

The PDFs are text-based (a real text layer), so local parsing is exact and no OCR
binary is needed.
"""

from __future__ import annotations

import random
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

from invoice_iq.schemas.documents import DocumentType
from invoice_iq.schemas.invoice import Invoice, LineItem, Money, Vendor

# Field labels shared by the generator and the rule-based extractor.
LABEL_NUMBER = "Invoice Number:"
LABEL_INV_DATE = "Invoice Date:"
LABEL_DUE_DATE = "Due Date:"
LABEL_CURRENCY = "Currency:"
LABEL_VENDOR = "Vendor:"
LABEL_ADDRESS = "Address:"
LABEL_TAX_ID = "Tax ID:"
LABEL_EMAIL = "Email:"
LABEL_SUBTOTAL = "Subtotal:"
LABEL_TAX = "Tax:"
LABEL_TOTAL = "Total:"
ITEM_PREFIX = "Item:"
ITEM_SEP = "|"


def _money_str(amount: Decimal) -> str:
    return f"{amount:.2f}"


def _qty_str(quantity: Decimal) -> str:
    # Drop trailing zeros so 10.00 -> "10" but 1.5 -> "1.5".
    normalized = quantity.normalize()
    return f"{normalized:f}"


# --------------------------------------------------------------------------- #
# Invoice -> canonical text lines
# --------------------------------------------------------------------------- #
def invoice_to_lines(invoice: Invoice) -> list[str]:
    """Render an `Invoice` to the canonical list of text lines."""
    lines: list[str] = ["INVOICE", ""]
    lines.append(f"{LABEL_NUMBER} {invoice.invoice_number}")
    lines.append(f"{LABEL_INV_DATE} {invoice.invoice_date.isoformat()}")
    if invoice.due_date is not None:
        lines.append(f"{LABEL_DUE_DATE} {invoice.due_date.isoformat()}")
    lines.append(f"{LABEL_CURRENCY} {invoice.currency}")
    lines.append("")

    lines.append(f"{LABEL_VENDOR} {invoice.vendor.name}")
    if invoice.vendor.address:
        lines.append(f"{LABEL_ADDRESS} {invoice.vendor.address}")
    if invoice.vendor.tax_id:
        lines.append(f"{LABEL_TAX_ID} {invoice.vendor.tax_id}")
    if invoice.vendor.email:
        lines.append(f"{LABEL_EMAIL} {invoice.vendor.email}")
    lines.append("")

    lines.append("Line Items:")
    for item in invoice.line_items:
        lines.append(
            f"{ITEM_PREFIX} {item.description} {ITEM_SEP} {_qty_str(item.quantity)} "
            f"{ITEM_SEP} {_money_str(item.unit_price.amount)} "
            f"{ITEM_SEP} {_money_str(item.line_total.amount)}"
        )
    lines.append("")

    lines.append(f"{LABEL_SUBTOTAL} {_money_str(invoice.subtotal.amount)}")
    if invoice.tax is not None:
        lines.append(f"{LABEL_TAX} {_money_str(invoice.tax.amount)}")
    lines.append(f"{LABEL_TOTAL} {_money_str(invoice.total.amount)}")
    return lines


# --------------------------------------------------------------------------- #
# Text lines -> PDF
# --------------------------------------------------------------------------- #
def render_text_pdf(path: Path, lines: list[str]) -> None:
    """Write `lines` to a single-/multi-page text PDF at `path`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(path), pagesize=LETTER)
    width, height = LETTER
    x, top, line_height = 50, height - 60, 16
    y = top
    pdf.setFont("Helvetica", 11)
    for line in lines:
        if y < 60:
            pdf.showPage()
            pdf.setFont("Helvetica", 11)
            y = top
        pdf.drawString(x, y, line)
        y -= line_height
    pdf.showPage()
    pdf.save()


def render_invoice_pdf(path: Path, invoice: Invoice) -> None:
    """Render an `Invoice` to a PDF using the canonical layout."""
    render_text_pdf(path, invoice_to_lines(invoice))


# --------------------------------------------------------------------------- #
# Receipt / contract text (contrast classes for the Phase 3 classifier)
# --------------------------------------------------------------------------- #
def receipt_lines(store: str, items: list[tuple[str, Decimal]], total: Decimal) -> list[str]:
    lines = ["RECEIPT", "", f"Store: {store}", "Thank you for your purchase!", ""]
    lines += [f"{name}  ${_money_str(price)}" for name, price in items]
    lines += ["", f"TOTAL  ${_money_str(total)}", "Payment: VISA ****1234", "Cash back: $0.00"]
    return lines


def contract_lines(party_a: str, party_b: str, effective: date) -> list[str]:
    return [
        "CONTRACT AGREEMENT",
        "",
        f"This Agreement is entered into as of {effective.isoformat()}",
        f"by and between {party_a} ('Provider')",
        f"and {party_b} ('Client').",
        "",
        "1. SCOPE OF SERVICES. The Provider shall render professional services",
        "   as described in Exhibit A, subject to the terms herein.",
        "2. TERM. This Agreement remains in effect for twelve (12) months",
        "   unless terminated earlier per Section 7.",
        "3. CONFIDENTIALITY. Each party shall protect the other's confidential",
        "   information and not disclose it to third parties.",
        "4. GOVERNING LAW. This Agreement is governed by the laws of the State.",
        "",
        "IN WITNESS WHEREOF, the parties have executed this Agreement.",
        "",
        "_______________________        _______________________",
        f"{party_a}                       {party_b}",
    ]


# --------------------------------------------------------------------------- #
# Randomized corpus generation
# --------------------------------------------------------------------------- #
_VENDORS = [
    ("Acme Corp", "123 Main Street, Springfield", "US123456789", "billing@acme.example"),
    ("Globex LLC", "500 Industrial Ave, Metropolis", "US987654321", "ar@globex.example"),
    ("Initech", "742 Evergreen Terrace, Ogdenville", "US555000111", "accounts@initech.example"),
    ("Umbrella Co", "1 Raccoon Plaza, Raccoon City", "US222333444", "pay@umbrella.example"),
]
_PRODUCTS = [
    "Consulting services",
    "Software license",
    "Cloud hosting",
    "Support retainer",
    "Implementation",
    "Training session",
]
_STORES = ["QuickMart", "FreshGrocer", "ByteCafe", "CityBooks"]
_PARTIES = ["Wayne Enterprises", "Stark Industries", "Wonka Inc", "Hooli"]


def random_invoice(rng: random.Random, *, currency: str = "USD") -> Invoice:
    """Build a valid, internally consistent random `Invoice`."""
    vendor_name, address, tax_id, email = rng.choice(_VENDORS)
    inv_date = date(2026, rng.randint(1, 12), rng.randint(1, 28))
    due = inv_date + timedelta(days=rng.choice([15, 30, 45]))
    n_items = rng.randint(1, 4)

    items: list[LineItem] = []
    subtotal = Decimal("0")
    for _ in range(n_items):
        desc = rng.choice(_PRODUCTS)
        qty = Decimal(rng.randint(1, 12))
        unit = Decimal(rng.randint(20, 500)) + Decimal("0.00")
        line_total = (qty * unit).quantize(Decimal("0.01"))
        subtotal += line_total
        items.append(
            LineItem(
                description=desc,
                quantity=qty,
                unit_price=Money(amount=unit, currency=currency),
                line_total=Money(amount=line_total, currency=currency),
            )
        )
    subtotal = subtotal.quantize(Decimal("0.01"))
    tax = (subtotal * Decimal("0.10")).quantize(Decimal("0.01"))
    total = (subtotal + tax).quantize(Decimal("0.01"))
    inv_number = f"INV-2026-{rng.randint(1000, 9999)}"

    return Invoice(
        invoice_number=inv_number,
        vendor=Vendor(name=vendor_name, address=address, tax_id=tax_id, email=email),
        invoice_date=inv_date,
        due_date=due,
        currency=currency,
        line_items=items,
        subtotal=Money(amount=subtotal, currency=currency),
        tax=Money(amount=tax, currency=currency),
        total=Money(amount=total, currency=currency),
    )


def random_receipt_lines(rng: random.Random) -> list[str]:
    store = rng.choice(_STORES)
    items = [
        (rng.choice(_PRODUCTS), Decimal(rng.randint(2, 40)) + Decimal("0.99"))
        for _ in range(rng.randint(2, 5))
    ]
    total = sum((p for _, p in items), Decimal("0")).quantize(Decimal("0.01"))
    return receipt_lines(store, items, total)


def random_contract_lines(rng: random.Random) -> list[str]:
    a, b = rng.sample(_PARTIES, 2)
    effective = date(2026, rng.randint(1, 12), rng.randint(1, 28))
    return contract_lines(a, b, effective)


def generate_corpus(out_dir: Path, n_per_class: int = 10, seed: int = 42) -> dict[str, str]:
    """Generate a labelled corpus of PDFs under `out_dir`.

    Returns a manifest mapping relative PDF path -> `DocumentType` value.
    """
    rng = random.Random(seed)
    out_dir = Path(out_dir)
    manifest: dict[str, str] = {}

    for i in range(n_per_class):
        inv_path = out_dir / "invoice" / f"invoice_{i:03d}.pdf"
        render_invoice_pdf(inv_path, random_invoice(rng))
        manifest[str(inv_path.relative_to(out_dir))] = DocumentType.INVOICE.value

        rcpt_path = out_dir / "receipt" / f"receipt_{i:03d}.pdf"
        render_text_pdf(rcpt_path, random_receipt_lines(rng))
        manifest[str(rcpt_path.relative_to(out_dir))] = DocumentType.RECEIPT.value

        ctr_path = out_dir / "contract" / f"contract_{i:03d}.pdf"
        render_text_pdf(ctr_path, random_contract_lines(rng))
        manifest[str(ctr_path.relative_to(out_dir))] = DocumentType.CONTRACT.value

    return manifest
