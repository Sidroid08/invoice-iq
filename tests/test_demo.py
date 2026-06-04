"""Tests for the one-command demo packaging (Phase 8)."""

from __future__ import annotations

from pathlib import Path

import pytest

from invoice_iq.agent import NextAction
from invoice_iq.demo import DemoSummary, render_human, run_demo
from invoice_iq.schemas.documents import DocumentType
from invoice_iq.schemas.invoice import Invoice
from invoice_iq.synthetic import render_invoice_pdf


def test_run_demo_invoice_flow(tmp_path: Path, valid_invoice: Invoice) -> None:
    pdf = tmp_path / "invoice.pdf"
    render_invoice_pdf(pdf, valid_invoice)

    summary = run_demo(
        pdf,
        checkpoint=tmp_path / "missing.pt",
        question="What is the total?",
        train_if_missing=True,
    )

    assert summary.predictor_source == "ephemeral-trained"
    assert summary.document_type is DocumentType.INVOICE
    assert summary.invoice_number == valid_invoice.invoice_number
    assert summary.total == "1210.00 USD"
    assert summary.answer is not None
    assert "1210.00" in summary.answer
    assert summary.recommendation is NextAction.APPROVE


def test_run_demo_can_require_existing_checkpoint(tmp_path: Path, valid_invoice: Invoice) -> None:
    pdf = tmp_path / "invoice.pdf"
    render_invoice_pdf(pdf, valid_invoice)

    with pytest.raises(FileNotFoundError, match="no classifier checkpoint"):
        run_demo(pdf, checkpoint=tmp_path / "missing.pt", train_if_missing=False)


def test_render_human_summary() -> None:
    summary = DemoSummary(
        pdf_path="data/samples/sample_invoice.pdf",
        predictor_source="checkpoint",
        document_type=DocumentType.INVOICE,
        classification_confidence=0.99,
        invoice_number="INV-1",
        vendor="Acme Corp",
        total="10.00 USD",
        question="What is the total?",
        answer="Total: 10.00 USD",
        recommendation=NextAction.APPROVE,
        trace=["classified", "extracted", "answered", "recommend"],
    )

    rendered = render_human(summary)

    assert "invoice-iq demo" in rendered
    assert "Classification: invoice (0.99)" in rendered
    assert "Recommendation: approve" in rendered
